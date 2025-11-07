import boto3
import os
import uuid
import tempfile
import logging
from typing import List
from dotenv import load_dotenv

# --- LangChain Components ---
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings

# --- LangChain Document Loaders ---
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_community.document_loaders.powerpoint import UnstructuredPowerPointLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

# Load environment variables from .env file
load_dotenv()

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
BEDROCK_MODEL_ID = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
BUCKET_NAME = os.getenv("BUCKET_NAME")

if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, BUCKET_NAME]):
    logger.error("Missing critical AWS environment variables. Check .env file.")
    raise RuntimeError("Missing critical AWS environment variables.")
    
# --- S3 Key Constants ---
FAISS_S3_KEY = "nmims_rag/vector_store.faiss"
PKL_S3_KEY = "nmims_rag/vector_store.pkl"
SOURCE_DOCS_PREFIX = "source_documents/" # <-- NEW: Staging folder

TEXT_CHUNK_SIZE = 1000
TEXT_CHUNK_OVERLAP = 200

# --- AWS Clients & Embeddings (Initialized once on module load) ---
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
    
    bedrock_embeddings = BedrockEmbeddings(
        model_id=BEDROCK_MODEL_ID,
        client=bedrock_client,
        region_name=AWS_REGION
    )
    logger.info("AWS clients and Bedrock embeddings initialized successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize AWS clients or Bedrock: {e}")
    raise RuntimeError(f"AWS/Bedrock initialization failed: {e}") from e

# --- NEW: Source File Management Functions ---

def list_source_files() -> List[dict]:
    """
    Lists all files in the S3 source document prefix.
    """
    files = []
    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=SOURCE_DOCS_PREFIX)
        if 'Contents' in response:
            for obj in response['Contents']:
                # Don't list the "folder" itself
                if obj['Key'] == SOURCE_DOCS_PREFIX:
                    continue
                files.append({
                    'key': os.path.basename(obj['Key']),
                    'last_modified': obj['LastModified'],
                    'size': obj['Size']
                })
        # Sort by last modified date, newest first
        files.sort(key=lambda x: x['last_modified'], reverse=True)
        return files
    except Exception as e:
        logger.error(f"Error listing source files from S3: {e}", exc_info=True)
        return []

def upload_source_file(temp_file_path: str, filename: str) -> bool:
    """
    Uploads a single source file to the S3 staging prefix.
    """
    s3_key = f"{SOURCE_DOCS_PREFIX}{filename}"
    try:
        s3_client.upload_file(
            Filename=temp_file_path,
            Bucket=BUCKET_NAME,
            Key=s3_key
        )
        logger.info(f"Successfully uploaded source file to S3: {s3_key}")
        return True
    except Exception as e:
        logger.error(f"Error uploading source file to S3: {e}", exc_info=True)
        return False

def delete_source_file(filename: str) -> bool:
    """
    Deletes a single source file from the S3 staging prefix.
    """
    s3_key = f"{SOURCE_DOCS_PREFIX}{filename}"
    try:
        s3_client.delete_object(Bucket=BUCKET_NAME, Key=s3_key)
        logger.info(f"Successfully deleted source file from S3: {s3_key}")
        return True
    except Exception as e:
        logger.error(f"Error deleting source file from S3: {e}", exc_info=True)
        return False

# --- Main Data Loading Function (Unchanged) ---
def load_document(file_path: str, filename: str) -> List[Document]:
    """
    Detects file type and uses the appropriate LangChain loader.
    """
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    
    loader = None
    
    try:
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.csv':
            loader = CSVLoader(file_path, source_column='source', autodetect_encoding=True)
        elif ext == '.xlsx':
            loader = UnstructuredExcelLoader(file_path)
        elif ext == '.docx':
            loader = Docx2txtLoader(file_path)
        elif ext == '.pptx':
            loader = UnstructuredPowerPointLoader(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path, autodetect_encoding=True)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        logger.info(f"Using {loader.__class__.__name__} for file {filename}")
        pages = loader.load() # This single call does all the work
        
        for page in pages:
            if 'source' not in page.metadata:
                page.metadata['source'] = filename
            if ext in ['.csv', '.xlsx'] and 'row' not in page.metadata:
                page.metadata['row'] = page.metadata.get('__index__', 0) + 1
            if ext == '.pdf' and 'page' not in page.metadata:
                 page.metadata['page'] = page.metadata.get('page_number', 0) + 1
        return pages
    except Exception as e:
        logger.error(f"Failed to load document {filename} with loader: {e}")
        raise

# --- Core Helper Functions (Unchanged) ---
def get_unique_id() -> str:
    """Generates a unique identifier (UUID)."""
    return str(uuid.uuid4())

def split_text(pages: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    # ... (This function is unchanged) ...
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
                metadata.update(chunk_start=start, chunk_end=end)
                chunks.append(Document(page_content=content, metadata=metadata))
                if end == len(text):
                    break
                start = end - chunk_overlap
        return chunks
    except Exception as e:
        logger.error(f"Error splitting text: {e}")
        raise

def create_and_upload_vector_store(request_id: str, documents: List[Document]) -> bool:
    """
    Creates a FAISS vector store from documents and uploads it to S3.
    """
    if not documents:
        logger.warning("No documents provided to create vector store. Creating empty store.")
        # Create an empty index to wipe the old one
        documents = [Document(page_content="empty", metadata={"source": "empty"})]

    try:
        logger.info("Creating FAISS vector store in memory...")
        vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
        logger.info("FAISS vector store created successfully.")

        with tempfile.TemporaryDirectory() as temp_dir:
            local_index_name = f"{request_id}_index"
            vectorstore_faiss.save_local(index_name=local_index_name, folder_path=temp_dir)
            
            local_faiss_path = os.path.join(temp_dir, f"{local_index_name}.faiss")
            local_pkl_path = os.path.join(temp_dir, f"{local_index_name}.pkl")

            if not os.path.exists(local_faiss_path) or not os.path.exists(local_pkl_path):
                logger.error("FAISS local save failed. Files not found.")
                return False

            logger.info(f"Uploading FAISS index to S3: {BUCKET_NAME}/{FAISS_S3_KEY}")
            s3_client.upload_file(
                Filename=local_faiss_path, Bucket=BUCKET_NAME, Key=FAISS_S3_KEY
            )
            
            logger.info(f"Uploading PKL file to S3: {BUCKET_NAME}/{PKL_S3_KEY}")
            s3_client.upload_file(
                Filename=local_pkl_path, Bucket=BUCKET_NAME, Key=PKL_S3_KEY
            )
        
        logger.info("Vector store uploaded to S3 successfully.")
        return True
    except Exception as e:
        logger.error(f"Error creating or uploading vector store: {e}")
        raise

# --- MODIFIED & NEW: Knowledge Base Management Functions ---

def rebuild_knowledge_base() -> int:
    """
    Downloads ALL source files, processes them, and builds a new vector store.
    """
    logger.info("Starting knowledge base rebuild...")
    source_files = list_source_files()
    all_documents = []
    
    if not source_files:
        logger.warning("No source files found. Building an empty knowledge base.")
        create_and_upload_vector_store(get_unique_id(), [])
        return 0

    with tempfile.TemporaryDirectory() as temp_dir:
        for file in source_files:
            try:
                filename = file['key']
                temp_file_path = os.path.join(temp_dir, filename)
                s3_key = f"{SOURCE_DOCS_PREFIX}{filename}"
                
                logger.info(f"Downloading: {s3_key}")
                s3_client.download_file(BUCKET_NAME, s3_key, temp_file_path)
                
                logger.info(f"Loading: {filename}")
                docs = load_document(temp_file_path, filename)
                all_documents.extend(docs)
            except Exception as e:
                logger.error(f"Failed to process file {filename}: {e}", exc_info=True)
                # Continue to next file
    
    logger.info(f"Total documents loaded: {len(all_documents)}")
    if not all_documents:
        logger.error("No documents could be loaded, though source files exist. Check file formats.")
        # Build empty store to wipe old one
        create_and_upload_vector_store(get_unique_id(), [])
        return 0

    splitted_docs = split_text(all_documents, TEXT_CHUNK_SIZE, TEXT_CHUNK_OVERLAP)
    logger.info(f"Total text chunks created: {len(splitted_docs)}")
    
    request_id = get_unique_id()
    create_and_upload_vector_store(request_id, splitted_docs)
    
    logger.info("Knowledge base rebuild complete.")
    return len(source_files)


def delete_vector_store() -> bool:
    """
    Deletes the FAISS files AND all source documents from S3.
    """
    logger.warning(f"Attempting to clear entire knowledge base...")
    try:
        # Delete all source files
        source_files = list_source_files()
        if source_files:
            keys_to_delete = [{'Key': f"{SOURCE_DOCS_PREFIX}{f['key']}"} for f in source_files]
            s3_client.delete_objects(Bucket=BUCKET_NAME, Delete={'Objects': keys_to_delete})
            logger.info(f"Deleted {len(keys_to_delete)} source files.")
        
        # Delete the .faiss file
        s3_client.delete_object(Bucket=BUCKET_NAME, Key=FAISS_S3_KEY)
        logger.info(f"Deleted S3 object: {FAISS_S3_KEY}")
        
        # Delete the .pkl file
        s3_client.delete_object(Bucket=BUCKET_NAME, Key=PKL_S3_KEY)
        logger.info(f"Deleted S3 object: {PKL_S3_KEY}")
        
        logger.info("Entire knowledge base cleared successfully.")
        return True
    except Exception as e:
        logger.error(f"Error clearing knowledge base from S3: {e}", exc_info=True)
        return False