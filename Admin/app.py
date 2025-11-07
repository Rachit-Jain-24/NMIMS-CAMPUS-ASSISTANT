import os
import tempfile
import logging
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Import the processing logic from our refactored module
# This assumes backend_processor.py is in the same folder
import backend_processor as bp

load_dotenv()

# --- Flask App Configuration ---
# 1. FIX: Use __name__ (double underscores)
app = Flask(__name__, static_folder='static', template_folder='templates')

app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'a_very_secret_fallback_key_CHANGE_ME')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB file upload limit (cleaned)

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
        # 2. REVERTED FIX: Kept your original KB_ROOT_PREFIX, as it's correct
        "S3_KB_PATH": bp.KB_ROOT_PREFIX,
        "S3_SOURCE_PREFIX": bp.SOURCE_DOCS_PREFIX,
        "CHUNK_SIZE": bp.TEXT_CHUNK_SIZE,
        "CHUNK_OVERLAP": bp.TEXT_CHUNK_OVERLAP,
        "EMBEDDING_MODEL": bp.BEDROCK_MODEL_ID,
    }
    
    # Fetch the list of files from S3
    try:
        source_files = bp.list_source_files()
    except Exception as e:
        bp.logger.error(f"Error listing files from S3: {e}", exc_info=True)
        flash(f'Error listing files from S3: {e}', 'danger')
        source_files = []
        
    return render_template('index.html', config_info=config_info, files=source_files)


@app.route('/upload', methods=['POST'])
def upload_file_router():
    """
    Handles uploading multiple files to the S3 staging folder.
    """
    # Use getlist to handle multiple files
    files = request.files.getlist('file')

    if not files or files[0].filename == '':
        flash('No files selected.', 'danger')
        return redirect(url_for('index'))

    uploaded_count = 0
    errors = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, filename)
                try:
                    file.save(temp_file_path)
                    bp.logger.info(f"File saved to temp path: {temp_file_path}")
                    
                    # Upload the file to S3 source prefix (staging)
                    if bp.upload_source_file(temp_file_path, filename):
                        uploaded_count += 1
                    else:
                        errors.append(filename)
                        flash(f'❌ Error uploading "{filename}" to S3.', 'danger')

                except Exception as e:
                    bp.logger.error(f"Critical error during upload of {filename}: {e}", exc_info=True)
                    errors.append(filename)
                    flash(f'An unexpected error occurred with {filename}: {e}', 'danger')
            
        elif file:
            errors.append(file.filename)
            flash(f'Invalid file type: "{file.filename}". Allowed types are: {", ".join(ALLOWED_EXTENSIONS)}', 'danger')

    # Provide a summary flash message
    if uploaded_count > 0:
        flash(f'✅ Successfully uploaded {uploaded_count} file(s). Click "Re-build" to apply changes.', 'success')
    if len(errors) > 0:
        flash(f'❌ Failed to upload {len(errors)} file(s).', 'danger')

    return redirect(url_for('index'))


@app.route('/rebuild', methods=['POST'])
def rebuild_knowledge_base_router():
    """
    Handles the request to rebuild the vector store from all source files.
    """
    try:
        bp.logger.info("Rebuild request received. Starting...")
        # 3. REVERTED FIX: Kept your original total_chunks logic, as it's correct
        total_chunks = bp.rebuild_knowledge_base()  
        flash(f'✅ Knowledge base rebuilt successfully ({total_chunks} chunks).', 'success')
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
            flash(f'✅ File "{filename}" deleted. Click "Re-build" to apply changes.', 'success')
        else:
            flash(f'❌ Error deleting "{filename}" from S3.', 'danger')
    except Exception as e:
        bp.logger.error(f"Critical error during /delete route: {e}", exc_info=True)
        flash(f'An unexpected error occurred: {e}', 'danger')
    
    return redirect(url_for('index'))


@app.route('/clear', methods=['POST'])
def clear_knowledge_base_router():
    """
    Handles the request to delete ALL federated vector stores AND all source files.
    """
    try:
        if bp.delete_vector_store():
            flash('✅ Entire knowledge base and all source files cleared successfully.', 'success')
        else:
            flash('❌ Could not clear knowledge base. Check S3 permissions or server logs.', 'danger')
    except Exception as e:
        bp.logger.error(f"Critical error during /clear route: {e}", exc_info=True)
        flash(f'An unexpected error occurred: {e}', 'danger')
    
    return redirect(url_for('index'))


# 4. FIX: Use __name__ and __main__ (double underscores)
if __name__ == '__main__':
    # Run with Gunicorn in production, not this.
    port = int(os.environ.get('PORT', 5000)) # Changed to 5000
    app.run(debug=True, host='0.0.0.0', port=port)