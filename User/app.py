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

# --- NEW DB IMPORTS ---
from flask_sqlalchemy import SQLAlchemy
from models import db, QueryLog # Assumes models.py defines db and QueryLog
# --- END NEW DB IMPORTS ---


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app, resources={r"/api/": {"origins": ""}})

# --- NEW: Database Configuration ---
db_uri = os.getenv("DATABASE_URL")
if not db_uri:
    logger.warning("DATABASE_URL is not set in .env file. Query logging will not work.")
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    # --- FIX 2: Added pool recycling to prevent 'Connection timed out' ---
    app.config['SQLALCHEMY_POOL_RECYCLE'] = 280 
    # --- END FIX 2 ---
    db.init_app(app)
# --- END NEW ---


# --- AWS and Model Configuration ---
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# --- Load Secret Key for Refresh Endpoint ---
REFRESH_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "default_secret_key_fallback")
if REFRESH_SECRET_KEY == "default_secret_key_fallback":
    logger.warning("FLASK_SECRET_KEY is not set. Refresh endpoint is using a default key.")

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
        
        try:
            header, encoded = data['audio_data'].split(',', 1)
        except ValueError:
            return jsonify({"error": "Malformed audio data."}), 400

        try:
            audio_bytes = base64.b64decode(encoded)
        except Exception:
            return jsonify({"error": "Invalid base64 audio payload."}), 400

        fd, temp_path = tempfile.mkstemp(suffix=".webm")
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(audio_bytes)
            logger.info(f"Transcribing audio file (size={len(audio_bytes)} bytes): {temp_path}")
            result = whisper_model.transcribe(temp_path)
            transcript = result.get("text", "").strip()
            logger.info(f"Transcription result: {transcript}")
            return jsonify({"transcript": transcript})
        finally:
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
        user_name = data.get('user_name') 

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

        # --- FIX 1: Removed 'req_' prefix. UUID is exactly 36 chars. ---
        req_id = str(uuid.uuid4())
        # --- END FIX 1 ---

        if db_uri:
            try:
                new_log = QueryLog(
                    id=req_id,
                    query_text=translated_query,
                    response_text=answer_text, 
                    classified_context=response_data.get("classified_context", "unknown"),
                    was_ambiguous=response_data.get("was_ambiguous", False),
                    was_no_docs=response_data.get("was_no_docs", False),
                    feedback=0 
                )
                db.session.add(new_log)
                db.session.commit()
                logger.info(f"Query logged to DB with id: {req_id}")
            except Exception as e:
                logger.error(f"Failed to log query to DB: {e}", exc_info=True)
                db.session.rollback()
        
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
            'request_id': req_id, 
            'confidence': 1.0
        }
        
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
    """
    REWRITTEN: This endpoint now updates the PostgreSQL database.
    """
    data = request.json or {}
    request_id = data.get("request_id") 
    rating_str = data.get("rating")     
    comment = data.get("comment", "") 

    if not request_id:
        logger.warning("Feedback request missing request_id")
        return jsonify({"status": "error", "message": "Missing request_id"}), 400
    
    if not db_uri:
        logger.error("Feedback request failed: Database not configured")
        return jsonify({"status": "error", "message": "Database not configured"}), 500

    feedback_value = 0
    if rating_str == "up":
        feedback_value = 1
    elif rating_str == "down":
        feedback_value = -1

    try:
        log_entry = db.session.query(QueryLog).filter_by(id=request_id).first()
        
        if log_entry:
            log_entry.feedback = feedback_value
            log_entry.comment = comment
            db.session.commit()
            logger.info(f"Feedback stored to DB for request_id: {request_id}")
            return jsonify({"status": "success", "message": "Feedback recorded"}), 200
        else:
            logger.warning(f"Feedback received for unknown request_id: {request_id}")
            return jsonify({"status": "error", "message": "Invalid request_id"}), 404
            
    except Exception as e:
        logger.exception("Failed to persist feedback to DB")
        db.session.rollback()
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

@app.route('/api/refresh-knowledge-base', methods=['POST'])
def refresh_knowledge_base():
    """
    A secure endpoint to trigger a reload of the in-memory vector stores.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or auth_header != REFRESH_SECRET_KEY:
        logger.warning("Unauthorized attempt to refresh knowledge base.")
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    
    if not rag_system:
        logger.error("Refresh triggered, but RAG system is not initialized.")
        return jsonify({"status": "error", "message": "RAG system not initialized"}), 500
        
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


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8086)) 
    app.run(debug=True, host='0.0.0.0', port=port)