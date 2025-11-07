import boto3
import os
import tempfile
import logging
import re
from dotenv import load_dotenv
from langchain_aws import BedrockLLM, BedrockEmbeddings
from langchain_community.vectorstores import FAISS
# --- THIS IS THE FIX ---
from langchain_core.prompts import PromptTemplate 
# ------------------------

# Setup
load_dotenv()
# 1. FIX: Use __name__ (double underscores)
logger = logging.getLogger(__name__)

# --- Constants ---
BEDROCK_EMBEDDING_MODEL_ID = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
BEDROCK_LLM_MODEL_ID = os.getenv("BEDROCK_LLM_MODEL_ID", "mistral.mixtral-8x7b-instruct-v0:1") # Or your preferred LLM
KB_ROOT_PREFIX = "nmims_rag/"
ALL_SCHOOL_CONTEXTS = ["SBM", "SOL", "SOC", "STME", "SPTM", "general"]

class RAGBackend:
    # 2. FIX: Use __init__ (double underscores)
    def __init__(self, bucket, s3_client, bedrock_client):
        self.bucket = bucket
        self.s3_client = s3_client
        self.bedrock_client = bedrock_client
        
        # Initialize LLM and Embeddings
        self.llm = self._get_llm()
        self.embeddings = self._get_embeddings()
        
        # Load all school-specific vector stores from S3 on startup
        self.vector_stores = self._load_all_vector_stores()

        # Prompt to classify the user's query
        self.classifier_prompt = f"""
        Based on the user's query, identify the most relevant school context from the following list:
        {ALL_SCHOOL_CONTEXTS}
        
        If the query mentions a specific school or program (e.g., "law", "SOL", "BBA-LLB", "business", "SBM", "MBA", "commerce", "SOC", "B.Com", "tech", "STME", "B.Tech", "pharmacy", "SPTM", "B.Pharm"), use that school.
        If the query is general (e.g., "holiday list", "campus address", "who is the vice chancellor?"), use "general".
        
        Return ONLY the matching school name from the list.
        Query: {{query}}
        School:
        """

        # Prompt for the RAG answer generation
        self.rag_prompt_template = """
        Human: You are an official assistant for SVKM's NMIMS, Hyderabad Campus.
        Your job is to answer the user's question using ONLY the provided context.
        Follow these rules:
        - If the answer is not present in the context, respond exactly with: "I'm sorry, I don't have that specific information in my knowledge base."
        - Be concise and professional. Start with a direct answer, then provide a short summary or bullet points if helpful.
        - When citing, use the 'source' and 'page' metadata. Format citations like this:.
        - Never invent information, URLs, or contact details.

        Context:
        {context}

        Question: {question}

        Assistant:
        """
        self.rag_prompt = PromptTemplate(template=self.rag_prompt_template, input_variables=["context", "question"])

    def _get_llm(self):
        return BedrockLLM(
            model_id=BEDROCK_LLM_MODEL_ID,
            client=self.bedrock_client,
            model_kwargs={"max_tokens": 1024, "temperature": 0.1, "top_p": 0.9}
        )

    def _get_embeddings(self):
        return BedrockEmbeddings(
            model_id=BEDROCK_EMBEDDING_MODEL_ID,
            client=self.bedrock_client
        )

    def _load_all_vector_stores(self):
        """
        Loops through all school contexts, downloads their index from S3, 
        and loads them into a dictionary in memory.
        """
        stores = {}
        for school in ALL_SCHOOL_CONTEXTS:
            try:
                s3_faiss_key = f"{KB_ROOT_PREFIX}{school}/vector_store.faiss"
                s3_pkl_key = f"{KB_ROOT_PREFIX}{school}/vector_store.pkl"
                
                # Check if files exist
                self.s3_client.head_object(Bucket=self.bucket, Key=s3_faiss_key)
                
                # Download to a school-specific temp directory
                local_dir = os.path.join(tempfile.gettempdir(), f"faiss_index_{school}")
                os.makedirs(local_dir, exist_ok=True)
                
                # 3. FIX: Download S3 files as "index.faiss" and "index.pkl"
                # This matches the default index_name="index" in load_local
                local_faiss_path = os.path.join(local_dir, "index.faiss")
                local_pkl_path = os.path.join(local_dir, "index.pkl")
                
                logger.info(f"Downloading index for {school}...")
                self.s3_client.download_file(self.bucket, s3_faiss_key, local_faiss_path)
                self.s3_client.download_file(self.bucket, s3_pkl_key, local_pkl_path)
                
                # Load the index from the local temp file
                stores[school] = FAISS.load_local(
                    folder_path=local_dir, 
                    embeddings=self.embeddings, 
                    allow_dangerous_deserialization=True, # Required for FAISS .pkl files
                    index_name="index" # Explicitly state the index name
                )
                logger.info(f"[SUCCESS] Loaded vector store for: {school}")
            except Exception as e:
                # This is not critical, it just means that school has no documents yet
                logger.warning(f"[WARN] Could not load vector store for {school}. This is normal if no documents have been added for it. Error: {e}")
        
        if not stores:
            logger.critical("No vector stores loaded at all. Admin must upload documents.")
            
        return stores

    def _classify_query(self, query: str) -> str:
        """
        Uses the LLM to classify the query and find the correct school context.
        """
        try:
            prompt = self.classifier_prompt.format(query=query)
            response = self.llm.invoke(prompt)
            
            # Clean the response to get just the school name
            # This regex finds the first occurrence of one of the school names
            match = re.search(r'\b(' + '|'.join(ALL_SCHOOL_CONTEXTS) + r')\b', response, re.IGNORECASE)
            
            if match:
                # 4. FIX: Robust matching for case-insensitivity
                raw_match = match.group(0)
                for school in ALL_SCHOOL_CONTEXTS:
                    if school.lower() == raw_match.lower():
                        logger.info(f"Query classified. Context: {school}")
                        return school # Return the correctly cased name (e.g., "general", "SBM")

        except Exception as e:
            logger.error(f"Query classification failed: {e}")
        
        # Fallback to general if classification fails or is ambiguous
        logger.info("Query classification ambiguous. Defaulting to 'general' context.")
        return "general"

    def get_rag_response(self, query: str) -> dict:
        """
        The main RAG pipeline: Classify -> Retrieve -> Generate
        """
        try:
            # 1. Classify the query to find the right school
            school_context = self._classify_query(query)

            # 2. Select the correct vector store
            vector_store = self.vector_stores.get(school_context)
            
            # Fallback logic: If no specific store, try 'general'
            if not vector_store:
                logger.warning(f"No specific store for '{school_context}', falling back to 'general'")
                vector_store = self.vector_stores.get("general")

            # If still no store, the knowledge base is empty
            if not vector_store:
                logger.error("No 'general' vector store found. Knowledge base is empty.")
                return {"answer": "Knowledge base is empty. Please ask the admin to upload documents.", "sources": []}

            # 3. Get relevant documents (chunks) from that specific store
            logger.info(f"Searching index '{school_context}' for: {query}")
            docs = vector_store.similarity_search(query, k=4) # Retrieve 4 chunks
            
            if not docs:
                logger.warning(f"No documents found in '{school_context}' index for query: {query}")
                return {"answer": "I'm sorry, I don't have that specific information in my knowledge base.", "sources": []}

            # 4. Build context and metadata
            context = ""
            sources = []
            source_set = set() # To de-duplicate sources
            
            for doc in docs:
                source_file = doc.metadata.get('source', 'Unknown')
                # Try to get page, fallback to row, else 'N/A'
                page_num = doc.metadata.get('page', doc.metadata.get('row', 'N/A'))
                source_key = f"{source_file}|{page_num}"

                # Add context for the LLM
                context += f"--- START OF CONTEXT (Source: {source_file}, Page: {page_num}) ---\n"
                context += doc.page_content + "\n"
                context += f"--- END OF CONTEXT (Source: {source_file}) ---\n\n"
                
                # Add unique sources for citation
                if source_key not in source_set:
                    # 5. FIX: Send clean metadata to the frontend
                    # The frontend expects 'file' and 'page'
                    sources.append({
                        "file": source_file,
                        "page": page_num
                    })
                    source_set.add(source_key)

            # 5. Generate the final answer
            formatted_prompt = self.rag_prompt.format(context=context, question=query)
            answer_text = self.llm.invoke(formatted_prompt)
            
            # Post-process to remove extra whitespace
            answer_text = re.sub(r'^\s*Assistant:\s*', '', answer_text).strip()
            
            return {"answer": answer_text, "sources": sources}

        except Exception as e:
            logger.error(f"RAG Error: {e}", exc_info=True)
            return {"answer": "I encountered an error while processing your request. Please try again.", "sources": []}