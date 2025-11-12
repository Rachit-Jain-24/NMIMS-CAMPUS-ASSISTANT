import os
import logging
import uuid
from flask import Flask, render_template, request, jsonify, url_for
from flask_cors import CORS
from dotenv import load_dotenv
import boto3

from rag_backend import RAGBackend

from deep_translator import GoogleTranslator
import whisper
import tempfile
import base64

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app, resources={r"/api/": {"origins": ""}})

# --- AWS and Model Configuration ---
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# --- Load Secret Key for Refresh Endpoint ---
REFRESH_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "default_secret_key_fallback")
if REFRESH_SECRET_KEY == "default_secret_key_fallback":
    logger.warning("FLASK_SECRET_KEY is not set. Refresh endpoint is using a default key.")

# --- Feedback persistence configuration ---
FEEDBACK_FILE = os.getenv("FEEDBACK_FILE", os.path.join(os.path.dirname(__file__), "feedback.jsonl"))

if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, BUCKET_NAME]):
    logger.critical("Missing AWS config. Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, BUCKET_NAME in environment/.env")
    
# --- Initialize Boto3 Clients ---
try:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    bedrock_client = boto3.client(
        service_name='bedrock-runtime',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    logger.info("Boto3 clients initialized successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize Boto3 clients: {e}")
    s3_client = None
    bedrock_client = None

# --- Load Whisper Model ---
try:
    whisper_model = whisper.load_model("base")
    logger.info("Whisper 'base' model loaded successfully.")
except Exception as e:
    logger.error(f"Could not load Whisper model: {e}. Voice transcription will fail.")
    whisper_model = None

# --- Initialize the RAGBackend Class ---
try:
    if not all([s3_client, bedrock_client, BUCKET_NAME]):
        raise RuntimeError("AWS clients or bucket name not configured. RAGBackend cannot start.")
        
    rag_system = RAGBackend(
        bucket=BUCKET_NAME,
        s3_client=s3_client,
        bedrock_client=bedrock_client
    )
    logger.info("RAGBackend class initialized and vector stores loaded.")
except Exception as e:
    logger.critical(f"Failed to initialize RAGBackend: {e}")
    rag_system = None

# --- Routes ---

@app.route('/')
def index():
    """Serves the main page that links to the widget."""
    return render_template('index.html')

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    if not whisper_model:
        return jsonify({"error": "Transcription service is not available."}), 503

    try:
        data = request.json
        if 'audio_data' not in data:
            return jsonify({"error": "No audio data provided."}), 400
        
        # Expect a data URL, e.g. "data:audio/webm;base64,<BASE64>"
        try:
            header, encoded = data['audio_data'].split(',', 1)
        except ValueError:
            return jsonify({"error": "Malformed audio data."}), 400

        try:
            audio_bytes = base64.b64decode(encoded)
        except Exception:
            return jsonify({"error": "Invalid base64 audio payload."}), 400

        # Windows NOTE: NamedTemporaryFile keeps the handle open -> ffmpeg can't read (Permission denied)
        # Use mkstemp, close handle, write bytes, then let ffmpeg read freely.
        fd, temp_path = tempfile.mkstemp(suffix=".webm")
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(audio_bytes)
            logger.info(f"Transcribing audio file (size={len(audio_bytes)} bytes): {temp_path}")
            # Call whisper; if you want faster latency consider tiny model
            result = whisper_model.transcribe(temp_path)
            transcript = result.get("text", "").strip()
            logger.info(f"Transcription result: {transcript}")
            return jsonify({"transcript": transcript})
        finally:
            # Ensure cleanup even if transcription fails
            try:
                os.remove(temp_path)
            except OSError as rm_err:
                logger.warning(f"Could not remove temp audio file {temp_path}: {rm_err}")

    except Exception as e:
        logger.exception("Error during transcription")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    if not rag_system or not rag_system.vector_stores:
        return jsonify({
            'answer': 'Error: The chatbot backend is not initialized. Please check server logs or ask the admin to upload documents.', 
            'sources': [],
            'request_id': 'error',
            'confidence': 0.0
        }), 500

    try:
        data = request.json
        query = data.get('query')
        language_code = data.get('language', 'en') 
        chat_history = data.get('history', [])
        user_name = data.get('user_name')  # Track user's name if provided

        if not query:
            return jsonify({'answer': 'No query provided.', 'sources': [], 'request_id': 'error', 'confidence': 0.0}), 400

        translated_query = query
        translated_history = []

        try:
            if language_code != 'en':
                translated_query = GoogleTranslator(source='auto', target='en').translate(query)
                logger.info(f"Translated query ({language_code} -> en): '{query}' -> '{translated_query}'")
                
                for turn in chat_history:
                    trans_q = GoogleTranslator(source='auto', target='en').translate(turn.get('query', ''))
                    trans_a = GoogleTranslator(source='auto', target='en').translate(turn.get('answer', ''))
                    translated_history.append({'query': trans_q, 'answer': trans_a})
                
                logger.info(f"Translated {len(translated_history)} history turns to English.")
            
            else:
                translated_query = query
                translated_history = chat_history
        
        except Exception as e:
            logger.error(f"Translation failed: {e}. Using original query/history.")
            translated_query = query
            translated_history = chat_history
        
        response_data = rag_system.get_rag_response(
            translated_query, 
            chat_history=translated_history,
            user_name=user_name
        )
        
        answer_text = response_data.get("answer", "An error occurred.")
        sources = response_data.get("sources", [])
        conversation_type = response_data.get("conversation_type")
        captured_name = response_data.get("user_name")
        
        try:
            if language_code != 'en':
                final_answer = GoogleTranslator(source='en', target=language_code).translate(answer_text)
                logger.info(f"Translated answer (en -> {language_code})")
            else:
                final_answer = answer_text
        except Exception as e:
            logger.error(f"Translation from English failed: {e}. Sending English answer.")
            final_answer = answer_text

        response_payload = {
            'answer': final_answer,
            'sources': sources,
            'request_id': f"req_{uuid.uuid4()}", 
            'confidence': 1.0
        }
        
        # Include conversation metadata if present
        if conversation_type:
            response_payload['conversation_type'] = conversation_type
        if captured_name:
            response_payload['user_name'] = captured_name

        return jsonify(response_payload)

    except Exception as e:
        logger.exception("Error in /api/chat endpoint")
        return jsonify({'answer': f'An error occurred: {str(e)}', 'sources': [], 'request_id': 'error', 'confidence': 0.0}), 500
        
@app.route('/api/feedback', methods=['POST'])
def feedback():
    data = request.json or {}
    # Enrich with server-side metadata
    record = {
        "id": f"fb_{uuid.uuid4()}",
        "user_query": data.get("query"),
        "assistant_answer": data.get("answer"),
        "rating": data.get("rating"),  # e.g., up/down or 1-5
        "comment": data.get("comment"),
        "sources": data.get("sources", []),
        "created_at": _import_('datetime').datetime.utcnow().isoformat() + "Z"
    }
    try:
        # Append as JSONL for easy later analysis
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            import json
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Feedback stored to {FEEDBACK_FILE}: {record}")
        return jsonify({"status": "success", "message": "Feedback recorded"}), 200
    except Exception as e:
        logger.exception("Failed to persist feedback")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/sources', methods=['GET'])
def get_source():
    file_id = request.args.get('file_id') 
    page = request.args.get('page')
    logger.info(f"Source request for: {file_id}, page {page}")

    if not rag_system:
            return jsonify({"snippet": "Error: The RAG system is not initialized."}), 500
        
    if not file_id or not page:
        return jsonify({"snippet": "Error: Missing file_id or page parameter."}), 400

    try:
        snippet = rag_system.get_source_snippet(file_id, page)
        
        return jsonify({
            "snippet": snippet
        }), 200
        
    except Exception as e:
        logger.exception(f"Error retrieving snippet for {file_id}")
        return jsonify({"snippet": f"An error occurred while fetching the source: {str(e)}"}), 500

# --- *** ADD THIS NEW ROUTE *** ---
@app.route('/api/refresh-knowledge-base', methods=['POST'])
def refresh_knowledge_base():
    """
    A secure endpoint to trigger a reload of the in-memory vector stores.
    """
    # 1. Check for the secret key
    auth_header = request.headers.get("Authorization")
    if not auth_header or auth_header != REFRESH_SECRET_KEY:
        logger.warning("Unauthorized attempt to refresh knowledge base.")
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    
    # 2. Check if the RAG system is initialized
    if not rag_system:
        logger.error("Refresh triggered, but RAG system is not initialized.")
        return jsonify({"status": "error", "message": "RAG system not initialized"}), 500
        
    # 3. Trigger the reload
    try:
        logger.info("Authorized refresh request received. Reloading stores...")
        success = rag_system.reload_all_stores()
        if success:
            logger.info("Reload successful.")
            return jsonify({"status": "success", "message": "Knowledge base reloaded."}), 200
        else:
            logger.error("Reload failed. Check RAGBackend logs.")
            return jsonify({"status": "error", "message": "Reload failed, check user logs."}), 500
    except Exception as e:
        logger.exception("Exception during knowledge base refresh.")
        return jsonify({"status": "error", "message": str(e)}), 500
# --- *** END OF NEW ROUTE *** ---


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8086)) 
    app.run(debug=True, host='0.0.0.0', port=port)