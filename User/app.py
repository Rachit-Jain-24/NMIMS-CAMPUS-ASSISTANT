import os
import logging
import uuid
from flask import Flask, render_template, request, jsonify, url_for
from flask_cors import CORS
from dotenv import load_dotenv
import boto3

# --- NEW: Import the RAGBackend class ---
from rag_backend import RAGBackend
# ----------------------------------------

# --- Imports for other features ---
from deep_translator import GoogleTranslator
import whisper
import tempfile
import base64
# --------------------------------

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# --- AWS and Model Configuration ---
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
BUCKET_NAME = os.getenv("BUCKET_NAME")

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

# --- NEW: Initialize the RAGBackend Class ---
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
# ------------------------------------------

# --- Routes ---

@app.route('/')
def index():
    """Serves the main page that links to the widget."""
    return render_template('index.html')

# --- Whisper Transcription Endpoint ---
@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    if not whisper_model:
        return jsonify({"error": "Transcription service is not available."}), 503

    try:
        data = request.json
        if 'audio_data' not in data:
            return jsonify({"error": "No audio data provided."}), 400
        
        header, encoded = data['audio_data'].split(',', 1)
        audio_bytes = base64.b64decode(encoded)

        with tempfile.NamedTemporaryFile(delete=True, suffix=".webm") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio.flush()
            
            logger.info(f"Transcribing temporary audio file: {temp_audio.name}")
            result = whisper_model.transcribe(temp_audio.name)
            transcript = result.get("text", "").strip()
            
            logger.info(f"Transcription result: {transcript}")
            return jsonify({"transcript": transcript})

    except Exception as e:
        logger.exception("Error during transcription")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# --- Main Chat Endpoint (Updated) ---
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
        
        # --- ENHANCEMENT: Get chat history from payload ---
        # We expect history to be a list of dicts: [{'query': '...', 'answer': '...'}]
        chat_history = data.get('history', []) 

        if not query:
            return jsonify({'answer': 'No query provided.', 'sources': [], 'request_id': 'error', 'confidence': 0.0}), 400

        # --- ENHANCEMENT: Translation logic now handles query AND history ---
        translated_query = query
        translated_history = []

        try:
            if language_code != 'en':
                # 1. Translate current query
                translated_query = GoogleTranslator(source='auto', target='en').translate(query)
                logger.info(f"Translated query ({language_code} -> en): '{query}' -> '{translated_query}'")
                
                # 2. Translate history to English for the RAG model
                for turn in chat_history:
                    trans_q = GoogleTranslator(source='auto', target='en').translate(turn.get('query', ''))
                    trans_a = GoogleTranslator(source='auto', target='en').translate(turn.get('answer', ''))
                    translated_history.append({'query': trans_q, 'answer': trans_a})
                
                logger.info(f"Translated {len(translated_history)} history turns to English.")
            
            else:
                # If lang is 'en', assume history is already in English
                translated_query = query
                translated_history = chat_history
        
        except Exception as e:
            logger.error(f"Translation failed: {e}. Using original query/history.")
            translated_query = query
            translated_history = chat_history # Pass as-is
        
        # --- 2. Get RAG Response (pass translated query and history) ---
        response_data = rag_system.get_rag_response(
            translated_query, 
            chat_history=translated_history
        )
        
        answer_text = response_data.get("answer", "An error occurred.")
        sources = response_data.get("sources", [])
        
        # --- 3. Translation Logic (from English) ---
        try:
            if language_code != 'en':
                final_answer = GoogleTranslator(source='en', target=language_code).translate(answer_text)
                logger.info(f"Translated answer (en -> {language_code})")
            else:
                final_answer = answer_text
        except Exception as e:
            logger.error(f"Translation from English failed: {e}. Sending English answer.")
            final_answer = answer_text
        # ---------------------------

        return jsonify({
            'answer': final_answer,
            'sources': sources,
            'request_id': f"req_{uuid.uuid4()}", 
            'confidence': 1.0 
        })

    except Exception as e:
        logger.exception("Error in /api/chat endpoint")
        return jsonify({'answer': f'An error occurred: {str(e)}', 'sources': [], 'request_id': 'error', 'confidence': 0.0}), 500
        
# --- Other API routes (unchanged) ---

@app.route('/api/feedback', methods=['POST'])
def feedback():
    """Handles feedback submission from the widget."""
    data = request.json
    logger.info(f"Feedback received: {data}")
    # TODO: Add logic to store this feedback
    return jsonify({"status": "success", "message": "Feedback received"}), 200

@app.route('/api/sources', methods=['GET'])
def get_source():
    """Handles source lookups from the widget."""
    file_id = request.args.get('file_id')
    page = request.args.get('page')
    logger.info(f"Source request for: {file_id}, page {page}")
    # TODO: Add logic to fetch a specific page snippet from S3
    return jsonify({
        "snippet": f"This is placeholder content for {file_id}, page {page}. Build this logic to fetch real data."
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8086)) 
    app.run(debug=True, host='0.0.0.0', port=port)