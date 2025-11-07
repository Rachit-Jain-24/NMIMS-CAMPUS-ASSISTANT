import os
import tempfile
import logging
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Import the processing logic
import backend_processor as bp

load_dotenv()

# --- Flask App Configuration ---
app = Flask(__name__, static_folder='static', template_folder='templates')

app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'a_very_secret_fallback_key_CHANGE_ME')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB file upload limit

ALLOWED_EXTENSIONS = {'pdf', 'csv', 'xlsx', 'docx', 'pptx', 'txt'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Routes ---

@app.route('/', methods=['GET'])
def index():
    """
    Handles displaying the main page and listing the files.
    """
    config_info = {
        "S3_BUCKET": bp.BUCKET_NAME,
        "S3_KB_PATH": bp.KB_ROOT_PREFIX,
        "S3_SOURCE_PREFIX": bp.SOURCE_DOCS_PREFIX,
        "CHUNK_SIZE": bp.TEXT_CHUNK_SIZE,
        "CHUNK_OVERLAP": bp.TEXT_CHUNK_OVERLAP,
        "EMBEDDING_MODEL": bp.BEDROCK_MODEL_ID,
    }
    
    try:
        source_files = bp.list_source_files()
    except Exception as e:
        bp.logger.error(f"Error listing files from S3: {e}", exc_info=True)
        flash(f'Error listing files from S3: {e}', 'danger')
        source_files = []
        
    return render_template(
        'index.html', 
        config_info=config_info, 
        files=source_files,
        school_contexts=bp.ALL_SCHOOL_CONTEXTS,
        doc_types=bp.ALL_DOC_TYPES
    )


@app.route('/upload', methods=['POST'])
def upload_file_router():
    """
    --- *** NEW SMART UPLOAD LOGIC *** ---
    Handles uploading files AND incrementally updating the vector store.
    This is now the FAST path.
    """
    school_context = request.form.get('school_context')
    doc_type = request.form.get('doc_type')

    if not school_context or not doc_type:
        flash('Missing school context or document type.', 'danger')
        return redirect(url_for('index'))
        
    files = request.files.getlist('file')

    if not files or files[0].filename == '':
        flash('No files selected.', 'danger')
        return redirect(url_for('index'))

    uploaded_count = 0
    errors = []
    total_chunks = 0

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, filename)
                try:
                    file.save(temp_file_path)
                    bp.logger.info(f"File saved to temp path: {temp_file_path}")
                    
                    # 1. Upload the raw file to S3 (for backup and rebuilds)
                    standardized_filename = bp.upload_source_file(temp_file_path, filename, school_context, doc_type)
                    if not standardized_filename:
                         raise Exception("S3 upload failed.")

                    # 2. Add the file to the live knowledge base (fast, incremental)
                    # --- SOLVED: Pass S3 key and display name ---
                    chunks_added = bp.add_file_to_knowledge_base(
                        standardized_filename, # The full S3 key
                        filename,              # The original display name
                        school_context, 
                        doc_type
                    )
                    
                    if chunks_added > 0:
                        uploaded_count += 1
                        total_chunks += chunks_added
                    else:
                        errors.append(filename)
                        flash(f'❌ File "{filename}" uploaded but failed to be processed (0 chunks created). Check logs.', 'warning')

                except Exception as e:
                    bp.logger.error(f"Critical error during upload/processing of {filename}: {e}", exc_info=True)
                    errors.append(filename)
                    flash(f'An unexpected error occurred with {filename}: {e}', 'danger')
            
        elif file:
            errors.append(file.filename)
            flash(f'Invalid file type: "{file.filename}". Allowed: {", ".join(ALLOWED_EXTENSIONS)}', 'danger')

    if uploaded_count > 0:
        flash(f'✅ Successfully uploaded and added {uploaded_count} file(s) ({total_chunks} chunks) to the knowledge base.', 'success')
    if len(errors) > 0:
        flash(f'❌ Failed to upload/process {len(errors)} file(s).', 'danger')

    return redirect(url_for('index'))


@app.route('/rebuild', methods=['POST'])
def rebuild_knowledge_base_router():
    """
    (SLOW) Handles the request to rebuild the vector store from all source files.
    This is now only needed after deleting files.
    """
    try:
        bp.logger.info("Rebuild request received. Starting...")
        total_chunks = bp.rebuild_knowledge_base()  
        flash(f'✅ (SLOW REBUILD) Knowledge base rebuilt successfully ({total_chunks} chunks).', 'success')
    except Exception as e:
        bp.logger.error(f"Critical error during /rebuild route: {e}", exc_info=True)
        flash(f'An unexpected error occurred during rebuild: {e}', 'danger')
    
    return redirect(url_for('index'))


@app.route('/delete', methods=['POST'])
def delete_file_router():
    """
    Handles the request to delete a single source file.
    """
    filename = request.form.get('filename')
    if not filename:
        flash('No filename provided for deletion.', 'danger')
        return redirect(url_for('index'))
    
    try:
        if bp.delete_source_file(filename):
            flash(f'✅ File "{filename}" deleted from S3. You MUST click "Re-build" to remove it from the chatbot.', 'warning')
        else:
            flash(f'❌ Error deleting "{filename}" from S3.', 'danger')
    except Exception as e:
        bp.logger.error(f"Critical error during /delete route: {e}", exc_info=True)
        flash(f'An unexpected error occurred: {e}', 'danger')
    
    return redirect(url_for('index'))


@app.route('/clear', methods=['POST'])
def clear_knowledge_base_router():
    """
    (DANGEROUS) Handles the request to delete ALL federated vector stores AND all source files.
    """
    try:
        if bp.delete_vector_store():
            flash('✅ Entire knowledge base and all source files cleared successfully.', 'success')
        else:
            flash('❌ Could not clear knowledge base. Check logs.', 'danger')
    except Exception as e:
        bp.logger.error(f"Critical error during /clear route: {e}", exc_info=True)
        flash(f'An unexpected error occurred: {e}', 'danger')
    
    return redirect(url_for('index'))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)