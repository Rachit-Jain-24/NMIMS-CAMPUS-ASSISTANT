import boto3
import os
import uuid
import tempfile
import logging
from typing import List
from dotenv import load_dotenv
import pandas as pd

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
# 1. FIX: Use __name__ (double underscores)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
BEDROCK_MODEL_ID = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# 2. FIX: Removed invisible whitespace characters
if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, BUCKET_NAME]):
    logger.error("Missing critical AWS environment variables. Check .env file.")
    raise RuntimeError("Missing critical AWS environment variables.")
    
# --- S3 Key Constants ---
KB_ROOT_PREFIX = "nmims_rag/"
SOURCE_DOCS_PREFIX = "source_documents/"
ALL_SCHOOL_CONTEXTS = ["SBM", "SOL", "SOC", "STME", "SPTM", "general"]

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

# --- Source File Management Functions ---

def list_source_files() -> List[dict]:
    """
    Lists all files in the S3 source document prefix.
    """
    files = []
    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=SOURCE_DOCS_PREFIX)
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'] == SOURCE_DOCS_PREFIX:
                    continue
                files.append({
                    'key': os.path.basename(obj['Key']),
                    'last_modified': obj['LastModified'],
                    'size': obj['Size']
                })
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

# --- Main Data Loading Function ---
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
            try:
                # Try to find a good source column
                df = pd.read_csv(file_path, nrows=1)
                if 'Title' in df.columns:
                    loader = CSVLoader(file_path, source_column='Title', autodetect_encoding=True)
                else:
                    loader = CSVLoader(file_path, autodetect_encoding=True)
            except Exception:
                loader = CSVLoader(file_path, autodetect_encoding=True)
        elif ext in ['.xlsx', '.xls']:
            loader = UnstructuredExcelLoader(file_path, mode="elements")
        elif ext == '.docx':
            loader = Docx2txtLoader(file_path)
        elif ext == '.pptx':
            loader = UnstructuredPowerPointLoader(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path, autodetect_encoding=True)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        # 3. FIX: Use __class__.__name__
        logger.info(f"Using {loader.__class__.__name__} for file {filename}")
        pages = loader.load()
        
        # Standardize metadata
        for i, page in enumerate(pages):
            # 4. CRITICAL FIX: Overwrite the source path
            # This ensures the clean filename is saved, not the temp path
            page.metadata['source'] = filename
            
            if ext in ['.csv', '.xlsx'] and 'row' not in page.metadata:
                page.metadata['row'] = page.metadata.get('_index_', i) + 1
            if ext == '.pdf' and 'page' not in page.metadata:
                page.metadata['page'] = page.metadata.get('page_number', i) + 1
        return pages
    except Exception as e:
        logger.error(f"Failed to load document {filename} with loader: {e}")
        raise

# --- Core Helper Functions ---
def get_unique_id() -> str:
    """Generates a unique identifier (UUID)."""
    return str(uuid.uuid4())

def get_file_metadata(filename: str) -> dict:
    """
    Extracts metadata tags (like school) from the filename.
    e.g., "SOL-Placements.pdf" -> {"school": "SOL", "doc_type": "placement"}
    """
    filename_lower = filename.lower()
    metadata = {"school": "general", "doc_type": "unknown"}
    
    if "sbm" in filename_lower:
        metadata["school"] = "SBM"
    elif "sol" in filename_lower:
        metadata["school"] = "SOL"
    elif "soc" in filename_lower:
        metadata["school"] = "SOC"
    elif "stme" in filename_lower:
        metadata["school"] = "STME"
    elif "sptm" in filename_lower:
        metadata["school"] = "SPTM"
    
    if "placement" in filename_lower:
        metadata["doc_type"] = "placement"
    elif "calendar" in filename_lower:
        metadata["doc_type"] = "calendar"
    elif "srb" in filename_lower:
        metadata["doc_type"] = "srb"
    elif "books" in filename_lower:
        metadata["doc_type"] = "book_list"
    
    logger.info(f"File: {filename} -> Metadata: {metadata}")
    return metadata

def split_text(pages: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    Splits a list of Document pages into character-based chunks with overlap.
    """
    # This implementation is fine and does not need to change.
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

#
# --- (MODIFIED FUNCTION) ---
#
def create_and_upload_vector_store(request_id: str, documents: List[Document], school: str) -> bool:
    """
    Creates a FAISS vector store FOR A SPECIFIC SCHOOL and uploads it to S3.
    This is now optimized for BATCH EMBEDDING.
    """
    
    # 2. FIX: Removed invisible whitespace
    s3_faiss_key = f"{KB_ROOT_PREFIX}{school}/vector_store.faiss"
    s3_pkl_key = f"{KB_ROOT_PREFIX}{school}/vector_store.pkl"

    if not documents:
        logger.warning(f"No documents provided for {school}. Creating empty store.")
        documents = [Document(page_content="empty", metadata={"source": "empty", "school": school})]

    try:
        # --- BATCHING SOLUTION ---
        # 1. Separate texts and metadata
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        logger.info(f"Generating {len(texts)} embeddings in batches for {school}...")
        
        # 2. Call embed_documents. This is highly optimized and uses parallel batch calls.
        # This will be MUCH faster than FAISS.from_documents()
        embeddings = bedrock_embeddings.embed_documents(texts)
        
        logger.info(f"Embeddings generated. Building local FAISS index for {school}...")

        # 3. Create FAISS index from the pre-computed embeddings
        # We use from_embeddings, which pairs texts and their vectors
        vectorstore_faiss = FAISS.from_embeddings(
            text_embeddings=zip(texts, embeddings), # FAISS.from_embeddings expects (text, vector) tuples
            embedding=bedrock_embeddings, # Pass the embedding function (needed by FAISS)
            metadatas=metadatas # Pass the metadata
        )
        # --- END BATCHING SOLUTION ---
        
        logger.info(f"FAISS vector store for {school} created successfully.")

        with tempfile.TemporaryDirectory() as temp_dir:
            local_index_name = f"{request_id}_index"
            vectorstore_faiss.save_local(index_name=local_index_name, folder_path=temp_dir)
            
            local_faiss_path = os.path.join(temp_dir, f"{local_index_name}.faiss")
            local_pkl_path = os.path.join(temp_dir, f"{local_index_name}.pkl")

            if not os.path.exists(local_faiss_path) or not os.path.exists(local_pkl_path):
                logger.error(f"FAISS local save failed for {school}.")
                return False

            logger.info(f"Uploading FAISS index to S3: {BUCKET_NAME}/{s3_faiss_key}")
            s3_client.upload_file(
                Filename=local_faiss_path, Bucket=BUCKET_NAME, Key=s3_faiss_key
            )
            
            logger.info(f"Uploading PKL file to S3: {BUCKET_NAME}/{s3_pkl_key}")
            s3_client.upload_file(
                Filename=local_pkl_path, Bucket=BUCKET_NAME, Key=s3_pkl_key
            )
        
        logger.info(f"Vector store for {school} uploaded to S3 successfully.")
        return True
    except Exception as e:
        logger.error(f"Error creating/uploading vector store for {school}: {e}")
        raise

# --- Knowledge Base Management Functions (Unchanged) ---

def rebuild_knowledge_base() -> int:
    """
    Downloads ALL source files, groups them by school, processes them, 
    and builds a SEPARATE vector store for each school.
    """
    logger.info("Starting federated knowledge base rebuild...")
    source_files = list_source_files()
    
    all_docs_by_school = {school: [] for school in ALL_SCHOOL_CONTEXTS}
    
    if not source_files:
        logger.warning("No source files found. Clearing all knowledge bases.")
        for school in ALL_SCHOOL_CONTEXTS:
            create_and_upload_vector_store(get_unique_id(), [], school)
        return 0

    with tempfile.TemporaryDirectory() as temp_dir:
        for file in source_files:
            try:
                filename = file['key']
                temp_file_path = os.path.join(temp_dir, filename)
                s3_key = f"{SOURCE_DOCS_PREFIX}{filename}"
                
                logger.info(f"Downloading: {s3_key}")
                s3_client.download_file(BUCKET_NAME, s3_key, temp_file_path)
                
                metadata = get_file_metadata(filename)
                school_context = metadata.get("school", "general")
                
                logger.info(f"Loading: {filename} for school: {school_context}")
                docs = load_document(temp_file_path, filename)
                
                for page in docs:
                    page.metadata.update(metadata)
                
                if school_context in all_docs_by_school:
                    all_docs_by_school[school_context].extend(docs)
                else:
                    all_docs_by_school["general"].extend(docs)

            except Exception as e:
                logger.error(f"Failed to process file {filename}: {e}", exc_info=True)
    
    total_chunks_processed = 0
    for school, school_docs in all_docs_by_school.items():
        if not school_docs:
            logger.info(f"No documents found for {school}. Building empty index to clear old data.")
            create_and_upload_vector_store(get_unique_id(), [], school)
            continue

        logger.info(f"Total documents loaded for {school}: {len(school_docs)}")
        
        splitted_docs = split_text(school_docs, TEXT_CHUNK_SIZE, TEXT_CHUNK_OVERLAP)
        logger.info(f"Total text chunks created for {school}: {len(splitted_docs)}")
        
        if not splitted_docs:
            logger.warning(f"No chunks created for {school}. Building empty index.")
            create_and_upload_vector_store(get_unique_id(), [], school)
            continue
            
        request_id = get_unique_id()
        create_and_upload_vector_store(request_id, splitted_docs, school)
        
        total_chunks_processed += len(splitted_docs)
    
    logger.info("Federated knowledge base rebuild complete.")
    return total_chunks_processed

def delete_vector_store() -> bool:
    """
    Deletes ALL school-specific FAISS files AND all source documents from S3.
    """
    logger.warning(f"Attempting to clear entire federated knowledge base...")
    try:
        # 1. Delete all source files
        source_files = list_source_files()
        if source_files:
            keys_to_delete = []
            for f in source_files:
                keys_to_delete.append({'Key': f"{SOURCE_DOCS_PREFIX}{f['key']}"})
            
            for i in range(0, len(keys_to_delete), 1000):
                s3_client.delete_objects(Bucket=BUCKET_NAME, Delete={'Objects': keys_to_delete[i:i+1000]})
            logger.info(f"Deleted {len(keys_to_delete)} source files.")
        
        # 2. Delete all federated indexes
        for school in ALL_SCHOOL_CONTEXTS:
            s3_faiss_key = f"{KB_ROOT_PREFIX}{school}/vector_store.faiss"
            s3_pkl_key = f"{KB_ROOT_PREFIX}{school}/vector_store.pkl"
            try:
                s3_client.delete_object(Bucket=BUCKET_NAME, Key=s3_faiss_key)
                s3_client.delete_object(Bucket=BUCKET_NAME, Key=s3_pkl_key)
                logger.info(f"Deleted index for {school}.")
            except s3_client.exceptions.ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    logger.info(f"No index to delete for {school}.")
                else:
                    raise # Re-raise other S3 errors
            except Exception as e:
                logger.error(f"Error deleting index for {school}: {e}", exc_info=True)
                
        logger.info("Entire federated knowledge base cleared successfully.")
        return True
    except Exception as e:
        logger.error(f"Error clearing knowledge base from S3: {e}", exc_info=True)
        return False