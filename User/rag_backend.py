import os
import boto3
from botocore.exceptions import ClientError
import logging
import tempfile
import re
import uuid
from dotenv import load_dotenv

# LangChain components
from langchain_core.prompts import PromptTemplate
from langchain_aws import BedrockEmbeddings, BedrockLLM
from langchain_community.vectorstores import FAISS

# Load environment
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
EMBEDDING_MODEL_ID = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
LLM_MODEL_ID = os.getenv("BEDROCK_LLM_MODEL_ID", "mistral.mixtral-8x7b-instruct-v0:1")

# AWS config
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
BUCKET_NAME = os.getenv("BUCKET_NAME")

FAISS_S3_KEY = "nmims_rag/vector_store.faiss"
PKL_S3_KEY = "nmims_rag/vector_store.pkl"

LOCAL_INDEX_DIR = tempfile.gettempdir()
LOCAL_INDEX_NAME = "nmims_rag_index"

# --- Global Clients (Initialized once) ---
try:
    if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, BUCKET_NAME]):
        logger.error("Missing AWS config. Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, BUCKET_NAME in environment/.env")
        raise ValueError("Missing AWS configuration")

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
        model_id=EMBEDDING_MODEL_ID,
        client=bedrock_client,
        region_name=AWS_REGION
    )
    logger.info("AWS clients and Bedrock embeddings initialized successfully.")
except Exception as e:
    logger.critical(f"AWS initialization failed: {e}")
    s3_client = None
    bedrock_client = None
    bedrock_embeddings = None

# --- Main RAG Functions ---

def load_vector_store():
    """
    Downloads and loads the FAISS vector store from S3.
    """
    if not s3_client or not bedrock_embeddings:
        logger.error("AWS clients not initialized. Cannot load vector store.")
        return None
        
    try:
        # Check if files exist in S3
        try:
            s3_client.head_object(Bucket=BUCKET_NAME, Key=FAISS_S3_KEY)
            s3_client.head_object(Bucket=BUCKET_NAME, Key=PKL_S3_KEY)
            logger.info(f"Vector store files found in S3.")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == '404' or error_code == 'NoSuchKey':
                logger.error(f"Vector store files not found in S3 bucket '{BUCKET_NAME}'. Upload a document via Admin portal.")
            else:
                logger.error(f"Error checking S3 files: {error_code} - {str(e)}")
            return None
        
        local_faiss_path = os.path.join(LOCAL_INDEX_DIR, f"{LOCAL_INDEX_NAME}.faiss")
        local_pkl_path = os.path.join(LOCAL_INDEX_DIR, f"{LOCAL_INDEX_NAME}.pkl")

        logger.info(f"Downloading vector store files from S3...")
        s3_client.download_file(BUCKET_NAME, FAISS_S3_KEY, local_faiss_path)
        s3_client.download_file(BUCKET_NAME, PKL_S3_KEY, local_pkl_path)
        
        logger.info("Loading FAISS vector store...")
        faiss_index = FAISS.load_local(
            index_name=LOCAL_INDEX_NAME,
            folder_path=LOCAL_INDEX_DIR,
            embeddings=bedrock_embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("Vector store loaded successfully!")
        return faiss_index
    except Exception as e:
        logger.exception(f"Failed to load vector store: {e}")
        return None

def get_llm():
    """
    Initializes and returns the Bedrock LLM.
    """
    if not bedrock_client:
        logger.error("Bedrock client not initialized. Cannot get LLM.")
        return None
        
    try:
        return BedrockLLM(
            model_id=LLM_MODEL_ID,
            client=bedrock_client,
            region_name=AWS_REGION,
            model_kwargs={
                "max_tokens": 4000,
                "temperature": 0.1,
                "top_p": 0.9
            }
        )
    except Exception as e:
        logger.exception("LLM initialization failed")
        return None

def get_rag_response(llm, vectorstore, question: str, k: int = 3) -> tuple:
    """
    Executes the RAG query.
    Returns: (answer_text, sources_list, request_id, confidence_score)
    """
    guardrail_no_info = "I don't have that information in the NMIMS knowledge base."
    request_id = f"req_{uuid.uuid4()}"
    
    # Using the improved prompt from our last conversation
    prompt_template = """
    Human: You are an expert AI assistant for NMIMS. Your job is to answer the user's question based *only* on the provided context.

    **Context:**
    {context}

    **Question:**
    {question}

    **Strict Rules for your Answer:**
    1.  **Format:** Provide a clean, professional response. Use bullet points or numbered lists if it makes the answer clearer.
    2.  **No Artifacts:** DO NOT include any stray numbers (like "25."), page markers (like "Page 20 of 23"), or repeated headers from the context. Your response must be pure, clean text.
    3.  **No Chat History:** DO NOT output the words "Question:", "Assistant:", or "Human:".
    4.  **Grounded:** Answer *only* using the context. If the answer is not in the context, respond exactly with: "I don't have that information in the NMIMS knowledge base."
    5.  **Citations:** If you use information from the context, add a citation like [Page X] at the end of the sentence, using the page number from the context.

    Assistant:
    """
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    try:
        # Use similarity_search_with_score to get confidence
        docs_with_scores = vectorstore.similarity_search_with_score(question, k=k)
        if not docs_with_scores:
            return guardrail_no_info, [], request_id, 0.0

        # Calculate confidence (Lower distance = higher confidence)
        # FAISS uses L2 distance; scores are 0 (perfect) to higher numbers.
        # Let's invert and normalize this to a 0-1 "confidence"
        # This is a simple heuristic; you can adjust the "scale" factor.
        scale = 1.5 
        top_score = docs_with_scores[0][1]
        confidence = max(0, min(1, (scale - top_score) / scale))

        context_parts = []
        sources_list = []
        
        for d, score in docs_with_scores:
            page = d.metadata.get("page", "?")
            file = d.metadata.get("source", "Unknown")
            snippet = (d.page_content or "").strip()
            
            context_parts.append(f"[Page {page}]\n{snippet}")
            sources_list.append({"file": file, "page": page, "s3_url": None}) # s3_url is in the widget, so we include it

        context = "\n\n".join(context_parts)
        if not context.strip():
            return guardrail_no_info, [], request_id, 0.0
            
        formatted_prompt = PROMPT.format(context=context, question=question)
        logger.info(f"Invoking LLM (model_id='{LLM_MODEL_ID}')")
        
        response = llm.invoke(formatted_prompt)
        text = response if isinstance(response, str) else str(response)
        
        if not text.strip():
            return guardrail_no_info, sources_list, request_id, confidence
            
        return text, sources_list, request_id, confidence
    
    except Exception as e:
        logger.exception("Error during RAG response generation")
        return f"I encountered an error while processing your request: {e}", [], request_id, 0.0

def _format_answer_for_lists(text: str) -> str:
    """Ensure bullets/numbers render with proper line breaks in markdown."""
    try:
        text = re.sub(r"^\s*\d+\.\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"(?<!\n)\s*-\s+", lambda m: "\n" + m.group(0).lstrip(), text)
        text = re.sub(r"(?<!\n)\s*\*\s+", lambda m: "\n" + m.group(0).lstrip(), text)
        text = re.sub(r"(?<!\n)(\s*)(\d{1,2}[\.)]\s+)", lambda m: "\n" + m.group(2), text)
        text = re.sub(r"\n\s*\n", "\n\n", text)
        return text.strip()
    except Exception:
        return text # Fallback