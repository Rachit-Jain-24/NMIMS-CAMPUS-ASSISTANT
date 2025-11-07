import boto3
import streamlit as st
import os
import uuid
import tempfile
import logging
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- LangChain Components ---
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings
from pypdf import PdfReader

## Pdf Loader
## import FAISS
# --- Setup Logging ---
# Configure logging to provide insights into the application's execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
# Centralize configuration for easy updates
BEDROCK_MODEL_ID = "amazon.titan-embed-text-v2:0"  # Using the correct Bedrock model ID for embeddings

# Load AWS configuration from environment variables or Streamlit secrets
try:
    # Try Streamlit secrets first (for cloud deployment)
    AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
    AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
    AWS_REGION = st.secrets["AWS_DEFAULT_REGION"]
    BUCKET_NAME = st.secrets["BUCKET_NAME"]
except:
    # Fallback to environment variables (for local deployment)
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
    BUCKET_NAME = os.getenv("BUCKET_NAME")

# Validate required environment variables
if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, BUCKET_NAME]):
    st.error("""Please set the following environment variables in your .env file:
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - BUCKET_NAME""")
    st.stop()

# Configure AWS clients with credentials
boto3.setup_default_session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# Using a specific path for the university's vector store
FAISS_S3_KEY = "nmims_rag/vector_store.faiss"
PKL_S3_KEY = "nmims_rag/vector_store.pkl"
TEXT_CHUNK_SIZE = 1000
TEXT_CHUNK_OVERLAP = 200

# --- AWS Clients ---
try:
    # Initialize S3 client
    s3_client = boto3.client('s3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    
    # Initialize Bedrock runtime client
    bedrock_client = boto3.client(
        service_name='bedrock-runtime',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    
    logger.info("AWS clients initialized successfully")
except Exception as e:
    st.error(f"Failed to initialize AWS clients: {str(e)}")
    logger.error(f"AWS client initialization error: {str(e)}")
    st.stop()
# Initialize Bedrock Embeddings
try:
    bedrock_embeddings = BedrockEmbeddings(
        model_id=BEDROCK_MODEL_ID,
        client=bedrock_client,
        region_name=AWS_REGION
    )
    logger.info("Bedrock embeddings initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Bedrock embeddings: {e}")
    st.error(f"Error initializing Bedrock embeddings: {e}. Please check model ID and region.")
    st.stop()

# --- Helper Functions ---

def get_unique_id() -> str:
    """Generates a unique identifier (UUID)."""
    return str(uuid.uuid4())

def split_text(pages: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    Splits a list of Document pages into character-based chunks with overlap.

    This avoids importing optional splitters that may pull heavy binary deps (e.g., spaCy).
    """
    try:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

        chunks: List[Document] = []
        for page in pages:
            text = page.page_content or ""
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                content = text[start:end]
                metadata = dict(page.metadata or {})
                metadata["chunk_start"] = start
                metadata["chunk_end"] = end
                chunks.append(Document(page_content=content, metadata=metadata))
                if end == len(text):
                    break
                start = end - chunk_overlap
        return chunks
    except Exception as e:
        logger.error(f"Error splitting text: {e}")
        st.error(f"An error occurred while splitting the document: {e}")
        return []

def load_pdf_pages(pdf_path: str) -> List[Document]:
    """Loads a PDF via pypdf and returns one Document per page."""
    try:
        reader = PdfReader(pdf_path)
        docs: List[Document] = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            docs.append(Document(page_content=text, metadata={"page": i + 1}))
        return docs
    except Exception as e:
        logger.error(f"Failed to read PDF: {e}")
        st.error(f"Unable to read PDF: {e}")
        return []

def create_and_upload_vector_store(request_id: str, documents: List[Document]) -> bool:
    """
    Creates a FAISS vector store from documents and uploads it to S3.

    Args:
        request_id: A unique ID for this processing request (used for temp naming).
        documents: The list of document chunks to embed.

    Returns:
        True if successful, False otherwise.
    """
    if not documents:
        logger.warning("No documents provided to create vector store.")
        st.warning("Cannot create vector store: No text was successfully processed from the PDF.")
        return False

    try:
        logger.info("Creating FAISS vector store in memory...")
        # Create the vector store from the document chunks and embeddings
        vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
        logger.info("FAISS vector store created successfully.")

        # Use a temporary directory for saving local files
        with tempfile.TemporaryDirectory() as temp_dir:
            local_index_name = f"{request_id}_index"
            vectorstore_faiss.save_local(index_name=local_index_name, folder_path=temp_dir)
            
            local_faiss_path = os.path.join(temp_dir, f"{local_index_name}.faiss")
            local_pkl_path = os.path.join(temp_dir, f"{local_index_name}.pkl")

            # Check if files were created before uploading
            if not os.path.exists(local_faiss_path) or not os.path.exists(local_pkl_path):
                logger.error("FAISS local save failed. Files not found.")
                st.error("Failed to save vector store locally before upload.")
                return False

            logger.info(f"Uploading FAISS index to S3: {BUCKET_NAME}/{FAISS_S3_KEY}")
            s3_client.upload_file(
                Filename=local_faiss_path,
                Bucket=BUCKET_NAME,
                Key=FAISS_S3_KEY
            )
            
            logger.info(f"Uploading PKL file to S3: {BUCKET_NAME}/{PKL_S3_KEY}")
            s3_client.upload_file(
                Filename=local_pkl_path,
                Bucket=BUCKET_NAME,
                Key=PKL_S3_KEY
            )
        
        logger.info("Vector store uploaded to S3 successfully.")
        return True

    except Exception as e:
        logger.error(f"Error creating or uploading vector store: {e}")
        st.error(f"An error occurred during vector store creation/upload: {e}")
        return False

# --- Main Streamlit Application ---

def main():
    """
    The main function to run the Streamlit application.
    """
    # Set the page configuration with the NMIMS branding
    st.set_page_config(page_title="NMIMS Admin Portal", layout="wide", page_icon="🎓")
    
    # Custom header with NMIMS branding
    st.markdown(
        """
        <style>
        .nmims-header {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 50%, #dee2e6 100%);
            border-bottom: 3px solid #d32f2f;
            padding: 25px 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            border-radius: 8px;
            position: relative;
            overflow: hidden;
        }
        .nmims-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="%23d32f2f" opacity="0.05"/><circle cx="75" cy="75" r="1" fill="%23d32f2f" opacity="0.05"/><circle cx="50" cy="10" r="0.5" fill="%23d32f2f" opacity="0.03"/><circle cx="10" cy="60" r="0.5" fill="%23d32f2f" opacity="0.03"/><circle cx="90" cy="40" r="0.5" fill="%23d32f2f" opacity="0.03"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
            pointer-events: none;
        }
        .nmims-logo-section {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            position: relative;
            z-index: 1;
        }
        .nmims-shield {
            width: 100px;
            height: 100px;
            margin-right: 12px;
            display: inline-block;
            border-radius: 4px;
            background: linear-gradient(45deg, #000000 0%, #000000 33%, #ffffff 33%, #ffffff 66%, #666666 66%, #666666 100%);
            position: relative;
        }
        .nmims-shield::before {
            content: "NMIMS";
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #d32f2f;
            font-weight: bold;
            font-size: 12px;
            text-align: center;
            line-height: 1.2;
        }
        .nmims-title {
            font-size: 18px;
            font-weight: 600;
            color: #333333;
            margin: 0;
            line-height: 1.2;
        }
        .nmims-subtitle {
            font-size: 24px;
            font-weight: 700;
            color: #d32f2f;
            margin: 0;
            letter-spacing: 1px;
        }
        .nmims-admin-badge {
            background: linear-gradient(45deg, #d32f2f, #f44336);
            color: white;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            display: inline-block;
            margin-top: 8px;
            box-shadow: 0 2px 8px rgba(211, 47, 47, 0.3);
            position: relative;
            z-index: 1;
        }
        .admin-info-box {
            background: #f8f9fa;
            border-left: 4px solid #d32f2f;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .upload-section {
            background: #ffffff;
            border: 2px dashed #d32f2f;
            border-radius: 8px;
            padding: 30px;
            text-align: center;
            margin: 20px 0;
            transition: all 0.3s ease;
        }
        .upload-section:hover {
            border-color: #f44336;
            background: #fafafa;
        }
        .success-box {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .error-box {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
        }
        </style>
        <div class="nmims-header">
          <div class="nmims-logo-section">
            <div class="nmims-shield"></div>
            <div>
              <div class="nmims-title">SVKM'S NMIMS Deemed to be UNIVERSITY</div>
              <div class="nmims-subtitle">HYDERABAD</div>
            </div>
          </div>
          <div class="nmims-admin-badge">⚙️ NMIMS Admin Portal</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Create two columns for layout - sidebar on left, main content on right
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Sidebar with admin information
        st.header("📋 Admin Information")
        st.markdown("**🏠 NMIMS Admin Portal**")
        st.markdown("Manage the chatbot's knowledge base by uploading official documents.")
        
        st.divider()
        
        # Processing Settings
        st.subheader("⚙️ Processing Settings")
        st.markdown(f"**Chunk Size:** {TEXT_CHUNK_SIZE}")
        st.markdown(f"**Chunk Overlap:** {TEXT_CHUNK_OVERLAP}")
        st.markdown(f"**Embedding Model:** {BEDROCK_MODEL_ID}")
        
        st.divider()
        
        # Technical Information
        st.subheader("🔧 Technical Details")
        st.markdown("**AWS Services:**")
        st.markdown("• Amazon Bedrock (Embeddings)")
        st.markdown("• Amazon S3 (Vector Store)")
        st.markdown("• FAISS (Vector Search)")
        
        st.markdown("**File Formats:**")
        st.markdown("• PDF Documents")
        st.markdown("• Text Extraction")
        st.markdown("• Vector Embeddings")
        
        st.divider()
        
        # # Contact Information
        # st.subheader("📞 Support")
        # st.markdown("**NMIMS Hyderabad**")
        # st.markdown("📍 Survey No. 102, Shamirpet")
        # st.markdown("📧 admin@nmims.edu")
        # st.markdown("🌐 hyderabad.nmims.edu")

    with col2:
        # Main content area
        st.markdown("### 📚 Knowledge Base Management")
        st.markdown("Upload official NMIMS documents to update the chatbot's knowledge base.")

        if not BUCKET_NAME:
            st.markdown('<div class="error-box">S3 BUCKET_NAME environment variable is not set. The application cannot proceed.</div>', unsafe_allow_html=True)
            logger.error("S3 BUCKET_NAME is not set.")
            st.stop()
        
        st.markdown(f'<div class="admin-info-box">📦 <strong>S3 Bucket:</strong> {BUCKET_NAME}<br>🔗 <strong>Vector Store Path:</strong> nmims_rag/</div>', unsafe_allow_html=True)

        # Enhanced upload section
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("📄 Upload an NMIMS Document (PDF)", type="pdf", help="Upload course catalogs, student handbooks, policy documents, etc.", key="admin_pdf_uploader")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Use a temporary file to safely handle the upload
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(uploaded_file.getvalue())
                temp_pdf_path = temp_pdf.name
            
            request_id = get_unique_id()
            st.markdown(f'<div class="admin-info-box">🆔 <strong>Processing Request ID:</strong> {request_id}</div>', unsafe_allow_html=True)
            logger.info(f"Started processing file: {uploaded_file.name} (Request ID: {request_id})")

            try:
                with st.spinner("📄 Loading and splitting PDF pages..."):
                    pages = load_pdf_pages(temp_pdf_path)
                
                if not pages:
                    st.markdown('<div class="error-box">⚠️ Could not extract any pages from the PDF.</div>', unsafe_allow_html=True)
                    logger.warning(f"No pages extracted from {uploaded_file.name}")
                    return

                st.markdown(f'<div class="admin-info-box">📊 <strong>Total Pages Extracted:</strong> {len(pages)}</div>', unsafe_allow_html=True)

                with st.spinner(f"✂️ Splitting document into chunks (Size: {TEXT_CHUNK_SIZE}, Overlap: {TEXT_CHUNK_OVERLAP})..."):
                    splitted_docs = split_text(pages, TEXT_CHUNK_SIZE, TEXT_CHUNK_OVERLAP)
                
                if not splitted_docs:
                    st.markdown('<div class="error-box">❌ Failed to split the document into chunks.</div>', unsafe_allow_html=True)
                    logger.error(f"split_text returned no docs for {request_id}")
                    return

                st.markdown(f'<div class="admin-info-box">📝 <strong>Total Text Chunks Created:</strong> {len(splitted_docs)}</div>', unsafe_allow_html=True)
                
                # Display a preview of the first few chunks
                with st.expander("🔍 Show document chunk preview"):
                    st.write("**Chunk 1:**")
                    st.text(splitted_docs[0].page_content)
                    st.write("---")
                    if len(splitted_docs) > 1:
                        st.write("**Chunk 2:**")
                        st.text(splitted_docs[1].page_content)

                with st.spinner("🤖 Creating text embeddings and building knowledge base... This may take a moment."):
                    result = create_and_upload_vector_store(request_id, splitted_docs)

                if result:
                    st.markdown('<div class="success-box">🎉 Success! The NMIMS document has been processed and the chatbot\'s knowledge base is updated.</div>', unsafe_allow_html=True)
                    logger.info(f"Successfully completed processing for {request_id}")
                else:
                    st.markdown('<div class="error-box">❌ Processing failed. Please check the logs for details.</div>', unsafe_allow_html=True)
                    logger.error(f"Processing failed for {request_id}")

            except Exception as e:
                st.markdown(f'<div class="error-box">💥 An unexpected error occurred: {e}</div>', unsafe_allow_html=True)
                logger.error(f"Critical error during main processing for {request_id}: {e}", exc_info=True)
            
            finally:
                # Clean up the temporary PDF file
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)
                    logger.info(f"Cleaned up temporary file: {temp_pdf_path}")

if __name__ == "__main__":
    main()

