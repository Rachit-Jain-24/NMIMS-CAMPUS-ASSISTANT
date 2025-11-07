import boto3
import os
import uuid
import time
import tempfile
import logging
import re
from typing import List
from dotenv import load_dotenv
import pandas as pd

# --- LangChain Components ---
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings

# --- LangChain Document Loaders ---
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_community.document_loaders.powerpoint import UnstructuredPowerPointLoader

# Load environment variables
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
KB_ROOT_PREFIX = "nmims_rag/"
SOURCE_DOCS_PREFIX = "source_documents/"
ALL_SCHOOL_CONTEXTS = ["SBM", "SOL", "SOC", "STME", "SPTM", "general"]
ALL_DOC_TYPES = ["placement", "calendar", "srb", "book_list", "other"]

TEXT_CHUNK_SIZE = 1000
TEXT_CHUNK_OVERLAP = 200

# --- Utility Functions ---
def get_unique_id() -> str:
    """Generate a unique ID for tracking purposes."""
    return str(uuid.uuid4())

# --- AWS Clients & Embeddings (Initialized once) ---
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
    textract_client = boto3.client(
        'textract',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    bedrock_embeddings = BedrockEmbeddings(
        model_id=BEDROCK_MODEL_ID,
        client=bedrock_client,
        region_name=AWS_REGION
    )
    logger.info("AWS clients (S3, Bedrock, Textract) and Bedrock embeddings initialized successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize AWS clients or Bedrock: {e}")
    raise RuntimeError(f"AWS/Bedrock initialization failed: {e}") from e

# --- Source File Management Functions ---

def list_source_files() -> List[dict]:
    """ Lists all files in the S3 source document prefix. """
    files = []
    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=SOURCE_DOCS_PREFIX)
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'] == SOURCE_DOCS_PREFIX:
                    continue
                
                filename = os.path.basename(obj['Key'])
                school, doc_type, display_name = _parse_standardized_filename(filename)

                files.append({
                    'key': filename, # The full key
                    'display_name': display_name,
                    'school': school,
                    'doc_type': doc_type,
                    'last_modified': obj['LastModified'],
                    'size': obj['Size']
                })
        files.sort(key=lambda x: x['last_modified'], reverse=True)
        return files
    except Exception as e:
        logger.error(f"Error listing source files from S3: {e}", exc_info=True)
        return []

def upload_source_file(temp_file_path: str, filename: str, school: str, doc_type: str) -> str:
    """
    Uploads a single source file to S3 with a standardized, parsable filename.
    Returns the new standardized filename.
    """
    standardized_filename = f"[{school.upper()}]_[{doc_type.lower()}]__{filename}"
    s3_key = f"{SOURCE_DOCS_PREFIX}{standardized_filename}"
    
    try:
        s3_client.upload_file(
            Filename=temp_file_path,
            Bucket=BUCKET_NAME,
            Key=s3_key
        )
        logger.info(f"Successfully uploaded source file to S3: {s3_key}")
        return standardized_filename
    except Exception as e:
        logger.error(f"Error uploading source file to S3: {e}", exc_info=True)
        return None

def delete_source_file(filename: str) -> bool:
    """ Deletes a single source file from the S3 staging prefix. """
    s3_key = f"{SOURCE_DOCS_PREFIX}{filename}"
    try:
        s3_client.delete_object(Bucket=BUCKET_NAME, Key=s3_key)
        logger.info(f"Successfully deleted source file from S3: {s3_key}")
        return True
    except Exception as e:
        logger.error(f"Error deleting source file from S3: {e}", exc_info=True)
        return False

# --- Core Document Loading Functions ---

def load_document(s3_key: str, display_name: str) -> List[Document]:
    """
    Detects file type and uses the appropriate (and most efficient) loader.
    """
    _, ext = os.path.splitext(display_name)
    ext = ext.lower()
    
    loader = None
    pages: List[Document] = []
    
    try:
        if ext == '.pdf':
            logger.info(f"Using AWS Textract directly for PDF: {s3_key}")
            full_s3_key = f"{SOURCE_DOCS_PREFIX}{s3_key}"
            
            # Use Textract to extract text from PDF
            try:
                response = textract_client.start_document_text_detection(
                    DocumentLocation={'S3Object': {'Bucket': BUCKET_NAME, 'Name': full_s3_key}}
                )
                job_id = response['JobId']
                
                # Wait for job to complete
                import time
                while True:
                    result = textract_client.get_document_text_detection(JobId=job_id)
                    status = result['JobStatus']
                    if status in ['SUCCEEDED', 'FAILED']:
                        break
                    time.sleep(2)
                
                if status == 'SUCCEEDED':
                    # Extract text from all pages
                    page_texts = {}
                    for block in result.get('Blocks', []):
                        if block['BlockType'] == 'LINE':
                            page_num = block.get('Page', 1)
                            if page_num not in page_texts:
                                page_texts[page_num] = []
                            page_texts[page_num].append(block.get('Text', ''))
                    
                    # Handle pagination if needed
                    next_token = result.get('NextToken')
                    while next_token:
                        result = textract_client.get_document_text_detection(JobId=job_id, NextToken=next_token)
                        for block in result.get('Blocks', []):
                            if block['BlockType'] == 'LINE':
                                page_num = block.get('Page', 1)
                                if page_num not in page_texts:
                                    page_texts[page_num] = []
                                page_texts[page_num].append(block.get('Text', ''))
                        next_token = result.get('NextToken')
                    
                    # Create Document objects for each page
                    for page_num in sorted(page_texts.keys()):
                        text = '\n'.join(page_texts[page_num])
                        if text.strip():
                            pages.append(Document(
                                page_content=text,
                                metadata={'source': display_name, 'page': page_num}
                            ))
                else:
                    logger.error(f"Textract job failed for {display_name}")
                    
            except Exception as textract_error:
                logger.error(f"Textract processing failed for {display_name}: {textract_error}")
                # Fallback: try pypdf as alternative
                try:
                    from pypdf import PdfReader
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_file_path = os.path.join(temp_dir, display_name)
                        s3_client.download_file(BUCKET_NAME, full_s3_key, temp_file_path)
                        reader = PdfReader(temp_file_path)
                        for i, page in enumerate(reader.pages):
                            text = page.extract_text()
                            if text.strip():
                                pages.append(Document(
                                    page_content=text,
                                    metadata={'source': display_name, 'page': i + 1}
                                ))
                except Exception as pypdf_error:
                    logger.error(f"PyPDF fallback also failed: {pypdf_error}")

        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, display_name)
                full_s3_key = f"{SOURCE_DOCS_PREFIX}{s3_key}"
                logger.info(f"Downloading {full_s3_key} to {temp_file_path} for local processing...")
                s3_client.download_file(BUCKET_NAME, full_s3_key, temp_file_path)
            
                if ext == '.csv':
                    logger.info(f"Using pandas.read_csv for {display_name}")
                    df = pd.read_csv(temp_file_path, on_bad_lines='skip', encoding='utf-8', encoding_errors='ignore')
                    for i, row in df.iterrows():
                        content = ", ".join([f"{col}: {val}" for col, val in row.astype(str).to_dict().items() if val not in [None, "nan", ""]])
                        metadata = {"source": display_name, "row": i + 1}
                        pages.append(Document(page_content=content, metadata=metadata))
                    
                elif ext in ['.xlsx', '.xls']:
                    logger.info(f"Using pandas.read_excel for {display_name}")
                    df = pd.read_excel(temp_file_path, sheet_name=None) # Load all sheets
                    for sheet_name, sheet_df in df.items():
                        for i, row in sheet_df.iterrows():
                            content = ", ".join([f"{col}: {val}" for col, val in row.astype(str).to_dict().items() if val not in [None, "nan", ""]])
                            metadata = {"source": display_name, "sheet": sheet_name, "row": i + 1}
                            pages.append(Document(page_content=content, metadata=metadata))
                
                elif ext == '.docx':
                    loader = Docx2txtLoader(temp_file_path)
                elif ext == '.pptx':
                    loader = UnstructuredPowerPointLoader(temp_file_path)
                elif ext == '.txt':
                    loader = TextLoader(temp_file_path, autodetect_encoding=True)
                else:
                    raise ValueError(f"Unsupported file type: {ext}")

                if loader:
                    pages = loader.load()

        if not pages:
            logger.warning(f"No pages/rows loaded for {display_name}.")
            return []

        for i, page in enumerate(pages):
            page.metadata['source'] = display_name 
            if ext == '.pdf' and 'page' not in page.metadata:
                page.metadata['page'] = i + 1
        
        return pages
        
    except Exception as e:
        logger.error(f"Failed to load document {display_name} with loader: {e}")
        return []

def split_text(pages: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    Splits a list of Document pages into character-based chunks with overlap.
    """
    chunks: List[Document] = []
    for page in pages:
        if 'row' in page.metadata: # Do not split tabular data
            chunks.append(page)
            continue
        
        text = page.page_content or ""
        
        if not text.strip(): # Skip empty pages
            continue
            
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

# --- Metadata Parsing ---

def _parse_standardized_filename(filename: str) -> tuple[str, str, str]:
    """
    Parses a standardized filename.
    Returns: (school, doc_type, display_name)
    """
    if filename.startswith("[") and "]__" in filename:
        try:
            parts = filename.split("]__", 1)
            tags = parts[0].replace("[", "").split("]_[")
            school = tags[0].upper()
            doc_type = tags[1].lower()
            display_name = parts[1]
            
            if school not in ALL_SCHOOL_CONTEXTS: school = "general"
            if doc_type not in ALL_DOC_TYPES: doc_type = "other"
            
            return school, doc_type, display_name
        except Exception:
            pass
    
    return "Unknown", "Unknown", filename

def get_file_metadata(filename: str) -> dict:
    """ Gets metadata from the standardized filename. """
    school, doc_type, _ = _parse_standardized_filename(filename)
    return {"school": school, "doc_type": doc_type}

# --- *** INCREMENTAL UPDATE LOGIC (FAST PATH) *** ---

def load_vector_store_from_s3(school: str) -> FAISS:
    """
    Downloads an existing FAISS index from S3 or creates a new empty one.
    """
    s3_faiss_key = f"{KB_ROOT_PREFIX}{school}/vector_store.faiss"
    s3_pkl_key = f"{KB_ROOT_PREFIX}{school}/vector_store.pkl"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        local_faiss_path = os.path.join(temp_dir, "vector_store.faiss")
        local_pkl_path = os.path.join(temp_dir, "vector_store.pkl")
        
        try:
            logger.info(f"Attempting to download existing index for {school}...")
            s3_client.download_file(BUCKET_NAME, s3_faiss_key, local_faiss_path)
            s3_client.download_file(BUCKET_NAME, s3_pkl_key, local_pkl_path)
            
            logger.info(f"Successfully downloaded index. Loading.")
            vector_store = FAISS.load_local(
                folder_path=temp_dir, 
                embeddings=bedrock_embeddings, 
                index_name="vector_store",
                allow_dangerous_deserialization=True
            )
            return vector_store
            
        except Exception as e:
            logger.warning(f"Could not load existing index for {school}: {e}. Creating new index.")
            dummy_texts = ["init"]
            dummy_metadatas = [{"source": "init"}]
            vector_store = FAISS.from_texts(
                texts=dummy_texts, 
                embedding=bedrock_embeddings, 
                metadatas=dummy_metadatas
            )
            vector_store.save_local(folder_path=temp_dir, index_name="vector_store")
            return vector_store

def save_vector_store_to_s3(vector_store: FAISS, school: str):
    """
    Saves a FAISS index to a temp folder and uploads it to S3.
    """
    s3_faiss_key = f"{KB_ROOT_PREFIX}{school}/vector_store.faiss"
    s3_pkl_key = f"{KB_ROOT_PREFIX}{school}/vector_store.pkl"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        local_faiss_path = os.path.join(temp_dir, "vector_store.faiss")
        local_pkl_path = os.path.join(temp_dir, "vector_store.pkl")
        
        vector_store.save_local(folder_path=temp_dir, index_name="vector_store")
        
        logger.info(f"Uploading updated index for {school} to S3...")
        s3_client.upload_file(local_faiss_path, BUCKET_NAME, s3_faiss_key)
        s3_client.upload_file(local_pkl_path, BUCKET_NAME, s3_pkl_key)
        logger.info(f"Successfully uploaded index for {school}.")

def add_file_to_knowledge_base(s3_key: str, display_name: str, school: str, doc_type: str) -> int:
    """
    The "smart" function. Loads an existing index, adds new chunks, and re-uploads.
    """
    logger.info(f"Starting incremental update for {display_name} in {school}...")
    
    vector_store = load_vector_store_from_s3(school)
    
    documents = load_document(s3_key, display_name)
    if not documents:
        logger.error(f"Failed to load any documents from {display_name}.")
        return 0
        
    metadata = {"school": school, "doc_type": doc_type}
    for doc in documents:
        doc.metadata.update(metadata)
        doc.metadata['source'] = display_name 

    chunks = split_text(documents, TEXT_CHUNK_SIZE, TEXT_CHUNK_OVERLAP)
    if not chunks:
        logger.error(f"Failed to create any chunks from {display_name}. File might be empty or a scanned image.")
        return 0

    logger.info(f"Adding {len(chunks)} new chunks to {school} vector store...")
    
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    embeddings = bedrock_embeddings.embed_documents(texts)
    
    vector_store.add_embeddings(text_embeddings=zip(texts, embeddings), metadatas=metadatas)
    
    save_vector_store_to_s3(vector_store, school)
    
    logger.info(f"Successfully added {len(chunks)} chunks for {display_name} to {school}.")
    return len(chunks)

# --- *** SLOW REBUILD FUNCTION (FOR DELETES) *** ---

def create_and_upload_vector_store(request_id: str, documents: List[Document], school: str) -> bool:
    """
    (Helper for rebuild) Creates a FAISS vector store from scratch.
    """
    s3_faiss_key = f"{KB_ROOT_PREFIX}{school}/vector_store.faiss"
    s3_pkl_key = f"{KB_ROOT_PREFIX}{school}/vector_store.pkl"

    if not documents:
        logger.warning(f"No documents provided for {school}. Creating empty store.")
        documents = [Document(page_content="empty", metadata={"source": "empty", "school": school})]

    try:
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        logger.info(f"Generating {len(texts)} embeddings in batches for {school}...")
        embeddings = bedrock_embeddings.embed_documents(texts)
        
        vectorstore_faiss = FAISS.from_embeddings(
            text_embeddings=zip(texts, embeddings),
            embedding=bedrock_embeddings,
            metadatas=metadatas
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            vectorstore_faiss.save_local(index_name="vector_store", folder_path=temp_dir)
            
            s3_client.upload_file(os.path.join(temp_dir, "vector_store.faiss"), BUCKET_NAME, s3_faiss_key)
            s3_client.upload_file(os.path.join(temp_dir, "vector_store.pkl"), BUCKET_NAME, s3_pkl_key)
        
        logger.info(f"Vector store for {school} uploaded to S3 successfully.")
        return True
    except Exception as e:
        logger.error(f"Error creating/uploading vector store for {school}: {e}")
        return False

def rebuild_knowledge_base() -> int:
    """
    (SLOW) Downloads ALL source files and rebuilds ALL indexes from scratch.
    """
    logger.warning("Starting SLOW knowledge base rebuild...")
    source_files = list_source_files()
    
    all_docs_by_school = {school: [] for school in ALL_SCHOOL_CONTEXTS}
    
    if not source_files:
        logger.warning("No source files found. Clearing all knowledge bases.")
        for school in ALL_SCHOOL_CONTEXTS:
            create_and_upload_vector_store(get_unique_id(), [], school)
        return 0

    for file in source_files:
        try:
            filename_key = file['key'] # Full standardized key
            display_name = file['display_name']
            
            metadata = get_file_metadata(filename_key)
            school_context = metadata.get("school", "general")
            
            logger.info(f"Loading: {display_name} for school: {school_context}")
            docs = load_document(filename_key, display_name)
            
            for page in docs:
                page.metadata.update(metadata)
                page.metadata['source'] = display_name
            
            all_docs_by_school[school_context].extend(docs)

        except Exception as e:
            logger.error(f"Failed to process file {filename_key}: {e}", exc_info=True)
    
    total_chunks_processed = 0
    for school, school_docs in all_docs_by_school.items():
        if not school_docs:
            logger.info(f"No documents for {school}. Building empty index.")
            create_and_upload_vector_store(get_unique_id(), [], school)
            continue

        splitted_docs = split_text(school_docs, TEXT_CHUNK_SIZE, TEXT_CHUNK_OVERLAP)
        logger.info(f"Total text chunks for {school}: {len(splitted_docs)}")
        
        if not splitted_docs:
            create_and_upload_vector_store(get_unique_id(), [], school)
            continue
            
        request_id = get_unique_id()
        create_and_upload_vector_store(request_id, splitted_docs, school)
        
        total_chunks_processed += len(splitted_docs)
    
    logger.info("Federated knowledge base rebuild complete.")
    return total_chunks_processed

def delete_vector_store() -> bool:
    """
    (DANGEROUS) Deletes ALL indexes AND all source documents.
    """
    logger.warning(f"Attempting to clear entire federated knowledge base...")
    try:
        source_files = list_source_files()
        if source_files:
            keys_to_delete = [{'Key': f"{SOURCE_DOCS_PREFIX}{f['key']}"} for f in source_files]
            for i in range(0, len(keys_to_delete), 1000):
                s3_client.delete_objects(Bucket=BUCKET_NAME, Delete={'Objects': keys_to_delete[i:i+1000]})
            logger.info(f"Deleted {len(keys_to_delete)} source files.")
        
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
                    raise
                
        logger.info("Entire federated knowledge base cleared successfully.")
        return True
    except Exception as e:
        logger.error(f"Error clearing knowledge base from S3: {e}", exc_info=True)
        return False