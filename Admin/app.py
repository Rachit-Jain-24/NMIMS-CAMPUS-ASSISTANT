import os
import tempfile
import logging
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# --- NEW IMPORTS for Dashboard ---
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from models import db, QueryLog  # Assumes models.py is in Admin/
# --- END NEW IMPORTS ---

# --- IMPORTS for Login ---
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
# -----------------------------

# Import the processing logic
import backend_processor as bp

load_dotenv()

# --- Flask App Configuration ---
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'a_very_secret_fallback_key_CHANGE_ME')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB file upload limit

# --- NEW: Database Configuration ---
# Get the database URL from your .env file
db_uri = os.getenv("DATABASE_URL")
if not db_uri:
    logging.warning("DATABASE_URL is not set in .env file. Dashboard will not work.")
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
# --- END NEW ---

# --- Load Admin Credentials ---
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_HASHED_PASSWORD = os.getenv("ADMIN_HASHED_PASSWORD")
if not ADMIN_HASHED_PASSWORD:
    logging.warning("ADMIN_HASHED_PASSWORD is not set. Admin login will fail.")
# ---------------------------------

# --- Initialize Extensions ---
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login' 
login_manager.login_message_category = 'info'
# ---------------------------------

ALLOWED_EXTENSIONS = {'pdf', 'csv', 'xlsx', 'docx', 'pptx', 'txt'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- User Model and Loader ---
class AdminUser(UserMixin):
    """Simple user class for the Admin."""
    def __init__(self, id):
        self.id = id
        self.username = ADMIN_USERNAME

@login_manager.user_loader
def load_user(user_id):
    """Required by Flask-Login to load the user from session."""
    if user_id == ADMIN_USERNAME:
        return AdminUser(ADMIN_USERNAME)
    return None
# ---------------------------------

# --- Login and Logout Routes ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles the login page."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == ADMIN_USERNAME and ADMIN_HASHED_PASSWORD and bcrypt.check_password_hash(ADMIN_HASHED_PASSWORD, password):
            user = AdminUser(username)
            login_user(user, remember=True) 
            flash('Logged in successfully.', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            flash('Login failed. Please check username and password.', 'danger')
            
    return render_template('login.html') 

@app.route('/logout')
@login_required 
def logout():
    """Handles logging out the user."""
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))
# -----------------------------------


# --- Main Admin Routes ---

@app.route('/', methods=['GET'])
@login_required
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

# --- NEW: Analytics Dashboard Route ---
@app.route('/dashboard')
@login_required
def dashboard():
    if not db_uri:
        flash('Database not configured. Cannot load dashboard.', 'danger')
        return redirect(url_for('index'))

    try:
        # --- 1. KPI Cards ---
        total_queries = db.session.query(QueryLog).count()
        total_failures = db.session.query(QueryLog).filter(
            (QueryLog.was_ambiguous == True) | 
            (QueryLog.was_no_docs == True) | 
            (QueryLog.feedback == -1)
        ).count()
        
        failure_rate = (total_failures / total_queries) * 100 if total_queries > 0 else 0
        
        good_feedback_count = db.session.query(QueryLog).filter(QueryLog.feedback == 1).count()
        total_feedback_count = db.session.query(QueryLog).filter(QueryLog.feedback != 0).count()
        good_feedback_rate = (good_feedback_count / total_feedback_count) * 100 if total_feedback_count > 0 else 0

        # --- 2. Top Queries Table ---
        top_queries = db.session.query(
            QueryLog.query_text, 
            func.count(QueryLog.query_text).label('count')
        ).group_by(QueryLog.query_text).order_by(func.count(QueryLog.query_text).desc()).limit(10).all()

        # --- 3. Failed Queries Table ---
        failed_queries = db.session.query(QueryLog).filter(
            (QueryLog.was_ambiguous == True) | 
            (QueryLog.was_no_docs == True) | 
            (QueryLog.feedback == -1)
        ).order_by(QueryLog.timestamp.desc()).limit(20).all()

        # --- 4. Queries by School Chart ---
        queries_by_school = db.session.query(
            QueryLog.classified_context,
            func.count(QueryLog.classified_context).label('count')
        ).group_by(QueryLog.classified_context).order_by(func.count(QueryLog.classified_context).desc()).all()

        school_chart_labels = [row.classified_context for row in queries_by_school]
        school_chart_data = [row.count for row in queries_by_school]

        # --- 5. Query Status Chart ---
        ambiguous_count = db.session.query(QueryLog).filter(QueryLog.was_ambiguous == True).count()
        no_docs_count = db.session.query(QueryLog).filter(QueryLog.was_no_docs == True).count()
        bad_feedback_count = db.session.query(QueryLog).filter(QueryLog.feedback == -1).count()
        
        # Calculate successful queries (total minus all *non-disliked* failures)
        # A query can be successful AND disliked.
        total_non_feedback_failures = ambiguous_count + no_docs_count
        successful_queries = total_queries - total_non_feedback_failures

        status_chart_labels = ['Successful', 'Disliked', 'Ambiguous', 'No Docs Found']
        # We show bad_feedback_count, but don't subtract it, as a "successful" query can be disliked.
        # This makes the pie chart more intuitive.
        status_chart_data = [successful_queries - bad_feedback_count, bad_feedback_count, ambiguous_count, no_docs_count]
        
        return render_template(
            'dashboard.html',
            total_queries=total_queries,
            total_failures=total_failures,
            failure_rate=failure_rate,
            good_feedback_rate=good_feedback_rate,
            top_queries=top_queries,
            failed_queries=failed_queries,
            school_chart_labels=school_chart_labels,
            school_chart_data=school_chart_data,
            status_chart_labels=status_chart_labels,
            status_chart_data=status_chart_data
        )

    except Exception as e:
        logger.error(f"Error loading dashboard: {e}", exc_info=True)
        flash(f'Error loading dashboard: {e}', 'danger')
        return redirect(url_for('index'))
# --- END NEW ROUTE ---


@app.route('/upload', methods=['POST'])
@login_required
def upload_file_router():
    """
    Handles uploading files AND incrementally updating the vector store.
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
                    
                    standardized_filename = bp.upload_source_file(temp_file_path, filename, school_context, doc_type)
                    if not standardized_filename:
                         raise Exception("S3 upload failed.")

                    chunks_added = bp.add_file_to_knowledge_base(
                        standardized_filename, 
                        filename,              
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
@login_required
def rebuild_knowledge_base_router():
    """
    (SLOW) Handles the request to rebuild the vector store from all source files.
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
@login_required
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
@login_required
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