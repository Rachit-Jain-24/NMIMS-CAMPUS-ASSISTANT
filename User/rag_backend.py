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

        # --- ENHANCEMENT: Updated RAG prompt to include chat history ---
        self.rag_prompt_template = """
        Human: You are an official assistant for SVKM's NMIMS, Hyderabad Campus.
        Your job is to answer the user's question using ONLY the provided context and chat history.
        Follow these rules:
        - If the answer is not present in the context, respond exactly with: "I'm sorry, I don't have that specific information in my knowledge base."
        - Be concise and professional. Start with a direct answer, then provide a short summary or bullet points if helpful.
        - When citing, use the 'source' and 'page' metadata.
        - Never invent information, URLs, or contact details.
        - Use the chat history to understand follow-up questions (e.g., if the user asks "what about for SBM?" after asking about placements).

        Chat History:
        {history}

        Context:
        {context}

        Question: {question}

        Assistant:
        """
        # --- ENHANCEMENT: Added "history" to input_variables ---
        self.rag_prompt = PromptTemplate(template=self.rag_prompt_template, input_variables=["history", "context", "question"])

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
                local_faiss_path = os.path.join(local_dir, "index.faiss")
                local_pkl_path = os.path.join(local_dir, "index.pkl")
                
                logger.info(f"Downloading index for {school}...")
                self.s3_client.download_file(self.bucket, s3_faiss_key, local_faiss_path)
                self.s3_client.download_file(self.bucket, s3_pkl_key, local_pkl_path)
                
                # Load the index from the local temp file
                stores[school] = FAISS.load_local(
                    folder_path=local_dir, 
                    embeddings=self.embeddings, 
                    allow_dangerous_deserialization=True, 
                    index_name="index" 
                )
                logger.info(f"[SUCCESS] Loaded vector store for: {school}")
            except Exception as e:
                logger.warning(f"[WARN] Could not load vector store for {school}. Error: {e}")
        
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
            
            match = re.search(r'\b(' + '|'.join(ALL_SCHOOL_CONTEXTS) + r')\b', response, re.IGNORECASE)
            
            if match:
                raw_match = match.group(0)
                for school in ALL_SCHOOL_CONTEXTS:
                    if school.lower() == raw_match.lower():
                        logger.info(f"Query classified. Context: {school}")
                        return school 

        except Exception as e:
            logger.error(f"Query classification failed: {e}")
        
        logger.info("Query classification ambiguous. Defaulting to 'general' context.")
        return "general"

    # --- ENHANCEMENT: Method signature updated to accept chat_history ---
    def get_rag_response(self, query: str, chat_history: list) -> dict:
        """
        The main RAG pipeline: Classify -> Retrieve -> Generate
        """
        try:
            # 1. Classify the query to find the right school
            school_context = self._classify_query(query)

            # --- ENHANCEMENT: "Classify + General" search strategy ---
            
            # 2. Select the correct vector stores
            vector_store = self.vector_stores.get(school_context)
            general_store = self.vector_stores.get("general")
            
            all_docs = []
            
            # 3. Search the specific store
            if vector_store:
                logger.info(f"Searching index '{school_context}' for: {query}")
                all_docs.extend(vector_store.similarity_search(query, k=2)) # Get top 2
            
            # 4. ALWAYS search the 'general' store (if it's not the same store)
            if general_store and school_context != "general":
                logger.info(f"Searching index 'general' for: {query}")
                all_docs.extend(general_store.similarity_search(query, k=2)) # Get top 2
            
            # Fallback if both failed
            if not vector_store and not general_store:
                logger.error("No 'general' or specific vector store found. Knowledge base is empty.")
                return {"answer": "Knowledge base is empty. Please ask the admin to upload documents.", "sources": []}
            # --- End of Enhancement ---

            if not all_docs:
                logger.warning(f"No documents found in '{school_context}' or 'general' index for query: {query}")
                return {"answer": "I'm sorry, I don't have that specific information in my knowledge base.", "sources": []}

            # 5. Build context and metadata
            context = ""
            sources = []
            source_set = set() # To de-duplicate sources
            
            # Use the combined 'all_docs' list
            for doc in all_docs:
                source_file = doc.metadata.get('source', 'Unknown')
                page_num = doc.metadata.get('page', doc.metadata.get('row', 'N/A'))
                source_key = f"{source_file}|{page_num}"

                context += f"--- START OF CONTEXT (Source: {source_file}, Page: {page_num}) ---\n"
                context += doc.page_content + "\n"
                context += f"--- END OF CONTEXT (Source: {source_file}) ---\n\n"
                
                if source_key not in source_set:
                    sources.append({
                        "file": source_file,
                        "page": page_num
                    })
                    source_set.add(source_key)

            # --- ENHANCEMENT: Format chat history for the prompt ---
            formatted_history = "\n".join([f"Human: {turn.get('query', '')}\nAssistant: {turn.get('answer', '')}" for turn in chat_history])
            
            # 6. Generate the final answer
            formatted_prompt = self.rag_prompt.format(history=formatted_history, context=context, question=query)
            answer_text = self.llm.invoke(formatted_prompt)
            
            answer_text = re.sub(r'^\s*Assistant:\s*', '', answer_text).strip()
            
            return {"answer": answer_text, "sources": sources}

        except Exception as e:
            logger.error(f"RAG Error: {e}", exc_info=True)
            return {"answer": "I encountered an error while processing your request. Please try again.", "sources": []}

    # --- NEW METHOD ---
    def get_source_snippet(self, source_file: str, page_num: str) -> str:
        """
        Retrieves the text content for a specific source and page/row.
        """
        logger.info(f"Snippet Request: Searching for {source_file}, page/row {page_num}")

        # We must search all vector stores, as we don't know the school context
        # from the source_file path alone.
        for school, store in self.vector_stores.items():
            try:
                # FAISS vector stores loaded from LangChain often keep an in-memory
                # docstore, which is a dictionary of ID -> Document.
                if hasattr(store, 'docstore') and hasattr(store.docstore, '_dict'):
                    for doc_id, doc in store.docstore._dict.items():
                        meta = doc.metadata
                        doc_source = meta.get('source', '')

                        # Check if this is the file we want
                        if doc_source == source_file:
                            # Now check if the page/row matches.
                            # We check 'page' (for PDFs) and 'row' (for CSVs/Excel)
                            current_page_val = meta.get('page', meta.get('row', 'N/A'))
                            
                            if str(current_page_val) == str(page_num):
                                logger.info(f"Found snippet in store '{school}'")
                                return doc.page_content
                else:
                    logger.warning(f"Store {school} has no accessible .docstore._dict. Cannot retrieve snippet.")

            except Exception as e:
                logger.error(f"Error searching docstore for {school}: {e}", exc_info=True)

        logger.warning(f"Snippet not found for {source_file}, page {page_num}")
        return f"Sorry, the full snippet for {os.path.basename(source_file)} (Page {page_num}) could not be retrieved."