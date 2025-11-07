import os
import logging
from flask import Flask, render_template, request, jsonify, url_for
from flask_cors import CORS  # <-- IMPORT THE NEW LIBRARY
from rag_backend import (
    load_vector_store, 
    get_llm, 
    get_rag_response,
    _format_answer_for_lists
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# --- ADD THIS LINE TO ENABLE CORS FOR YOUR API ---
CORS(app, resources={r"/api/*": {"origins": "*"}})
# --------------------------------------------------

# --- Load models once on startup ---
logger.info("Starting Flask app... Loading RAG backend...")
vector_store = load_vector_store()
llm = get_llm()

if not vector_store or not llm:
    logger.critical("Failed to load vector store or LLM. The chatbot will not be functional.")
else:
    logger.info("RAG backend loaded successfully.")

# --- Routes ---

@app.route('/')
def index():
    """Serves the main page that links to the widget."""
    return render_template('index.html')

@app.route('/widget')
def chat_widget():
    """Serves the chat widget HTML file."""
    return render_template('chat_widget.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handles the chat API requests from the widget."""
    if not vector_store or not llm:
        return jsonify({
            'answer': 'Error: The chatbot backend is not initialized. Please check server logs.', 
            'sources': [],
            'request_id': 'error',
            'confidence': 0.0
        }), 500

    try:
        data = request.json
        query = data.get('query')
        
        if not query:
            return jsonify({'answer': 'No query provided.', 'sources': [], 'request_id': 'error', 'confidence': 0.0}), 400

        # Get RAG response (assuming English, as widget has no selector)
        answer_text, sources, req_id, confidence = get_rag_response(llm, vector_store, query, k=3)
            
        # Format answer for clean display
        formatted_answer = _format_answer_for_lists(answer_text)

        return jsonify({
            'answer': formatted_answer, 
            'sources': sources,
            'request_id': req_id,
            'confidence': float(confidence) # <-- FIX: Cast to standard Python float
        })

    except Exception as e:
        logger.exception("Error in /api/chat endpoint")
        return jsonify({'answer': f'An error occurred: {str(e)}', 'sources': [], 'request_id': 'error', 'confidence': 0.0}), 500
        
# --- Other API routes from your widget (e.g., feedback, sources) ---

@app.route('/api/feedback', methods=['POST'])
def feedback():
    """Handles feedback submission from the widget."""
    data = request.json
    logger.info(f"Feedback received: {data}")
    # TODO: Add logic to store this feedback (e.g., in a database or S3 file)
    return jsonify({"status": "success", "message": "Feedback received"}), 200

@app.route('/api/sources', methods=['GET'])
def get_source():
    """Handles source lookups from the widget."""
    file_id = request.args.get('file_id')
    page = request.args.get('page')
    logger.info(f"Source request for: {file_id}, page {page}")
    # TODO: Add logic to fetch a specific page snippet from S3
    # For now, return a placeholder
    return jsonify({
        "snippet": f"This is placeholder content for {file_id}, page {page}. Build this logic to fetch real data."
    }), 200


if __name__ == '__main__':
    # Run on port 8086 as specified in your README
    port = int(os.environ.get('PORT', 8086)) 
    app.run(debug=True, host='0.0.0.0', port=port)