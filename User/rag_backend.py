import boto3
import os
import tempfile
import logging
import re
from dotenv import load_dotenv
from langchain_aws import BedrockLLM, BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate 

# Setup
load_dotenv()
logger = logging.getLogger(__name__)

# --- Constants ---
BEDROCK_EMBEDDING_MODEL_ID = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
BEDROCK_LLM_MODEL_ID = os.getenv("BEDROCK_LLM_MODEL_ID", "mistral.mixtral-8x7b-instruct-v0:1")
KB_ROOT_PREFIX = "nmims_rag/"
ALL_SCHOOL_CONTEXTS = ["SBM", "SOL", "SOC", "STME", "SPTM", "general"]
# --- Define our ambiguity trigger phrase ---
AMBIGUITY_RESPONSE = "That's a good question. Which school are you asking about? (e.g., SBM, SOL, SOC, STME, or SPTM)"

class RAGBackend:
    def __init__(self, bucket, s3_client, bedrock_client):
        self.bucket = bucket
        self.s3_client = s3_client
        self.bedrock_client = bedrock_client
        
        self.llm = self._get_llm()
        self.embeddings = self._get_embeddings()
        
        self.vector_stores = self._load_all_vector_stores()

        # --- *** NEW SIMPLIFIED CLASSIFIER PROMPT *** ---
        # This prompt ONLY looks at the current query.
        # It no longer receives chat history, which was polluting its results.
        self.classifier_prompt = f"""
        You are a strict classifier. Based on the CURRENT QUERY ONLY, you must identify ONE context from this list:
        {ALL_SCHOOL_CONTEXTS + ["AMBIGUOUS"]}

        Follow these rules:
        1.  If the query contains keywords for a specific school, use that school.
            - "SBM" or "Business Management": return "SBM"
            - "SOL" or "Law": return "SOL"
            - "SOC" or "Commerce": return "SOC"
            - "STME" or "Technology Management" or "Engineering": return "STME"
            - "SPTM" or "Pharmacy": return "SPTM"
        2.  If the query is clearly general (e.g., "holiday list", "campus address", "vice chancellor", "disciplinary committee", "hostel"), return "general".
        3.  If the query is ambiguous and could apply to *any* school (e.g., "attendance", "exam dates", "placements", "academic calendar", "srb", "book list", "class timings", "leave rules", "ragging policy"), return "AMBIGUOUS".
        
        Return ONLY the matching word from the list.

        Query: {{query}}
        School:
        """
        # --- *** END NEW PROMPT *** ---

        # --- *** NEW SIMPLIFIED RAG PROMPT *** ---
        # This new prompt is much simpler to prevent
        # the bot from hallucinating follow-up questions.
        self.rag_prompt_template = """
        You are an official assistant for SVKM's NMIMS, Hyderabad Campus.
        Use the "Context" to answer the "Question".
        - Use the "Chat History" to understand the "Question" (e.g., if "Question" is "STME", history might show the user was asking about "exam dates").
        - ONLY answer the single "Question" at the end. Do not add extra information or answer previous questions.
        - If the answer is not in the "Context", respond exactly with: "I'm sorry, I don't have that specific information in my knowledge base."
        - Cite your sources using.

        Chat History:
        {history}

        Context:
        {context}

        Question: {question}

        Assistant:
        """
        # --- *** END NEW PROMPT *** ---
        
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
        stores = {}
        for school in ALL_SCHOOL_CONTEXTS:
            try:
                s3_faiss_key = f"{KB_ROOT_PREFIX}{school}/vector_store.faiss"
                s3_pkl_key = f"{KB_ROOT_PREFIX}{school}/vector_store.pkl"
                
                self.s3_client.head_object(Bucket=self.bucket, Key=s3_faiss_key)
                
                local_dir = os.path.join(tempfile.gettempdir(), f"faiss_index_{school}")
                os.makedirs(local_dir, exist_ok=True)
                
                local_faiss_path = os.path.join(local_dir, "index.faiss")
                local_pkl_path = os.path.join(local_dir, "index.pkl")
                
                logger.info(f"Downloading index for {school}...")
                self.s3_client.download_file(self.bucket, s3_faiss_key, local_faiss_path)
                self.s3_client.download_file(self.bucket, s3_pkl_key, local_pkl_path)
                
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

    def reload_all_stores(self):
        """
        Dumps the current vector stores and re-loads them from S3.
        """
        logger.info("--- RELOADING ALL VECTOR STORES FROM S3 ---")
        try:
            self.vector_stores.clear() 
            self.vector_stores = self._load_all_vector_stores()
            logger.info("--- RELOAD COMPLETE ---")
            return True
        except Exception as e:
            logger.error(f"Failed to reload vector stores: {e}")
            return False

    # --- Classifier now has no history ---
    def _classify_query(self, query: str) -> str:
        """
        Uses the LLM to classify the query (without history)
        and find the correct school context.
        """
        try:
            prompt = self.classifier_prompt.format(query=query)
            response = self.llm.invoke(prompt)
            
            all_valid_contexts = ALL_SCHOOL_CONTEXTS + ["AMBIGUOUS"]
            match = re.search(r'(' + '|'.join(all_valid_contexts) + r')', response, re.IGNORECASE)
            
            if match:
                raw_match = match.group(0)
                for context_name in all_valid_contexts:
                    if context_name.lower() == raw_match.lower():
                        logger.info(f"Query classified. Context: {context_name}")
                        return context_name 

        except Exception as e:
            logger.error(f"Query classification failed: {e}")
        
        logger.info("Query classification failed. Defaulting to 'general'.")
        return "general" # Default to 'general' on failure

    # --- *** NEW RAG PIPELINE (FEDERATED SEARCH) *** ---
    def get_rag_response(self, query: str, chat_history: list) -> dict:
        """
        The main RAG pipeline: Classify -> Retrieve -> Generate
        """
        try:
            # 1. Classify the query (without history)
            school_context = self._classify_query(query)

            # 2. Handle Follow-up Questions
            query_to_search = query
            if chat_history:
                last_bot_answer = chat_history[-1].get('answer', '')
                last_user_query = chat_history[-1].get('query', '')
                
                if last_bot_answer == AMBIGUITY_RESPONSE:
                    potential_school = self._classify_query(query)
                    
                    if potential_school in ALL_SCHOOL_CONTEXTS and potential_school != "general":
                        school_context = potential_school
                        query_to_search = last_user_query 
                        logger.info(f"Follow-up detected. Re-running query '{query_to_search}' for school '{school_context}'")
                    else:
                        school_context = "AMBIGUOUS"

            # 3. Check for ambiguity
            if school_context == "AMBIGUOUS":
                logger.info(f"Ambiguous query detected: {query_to_search}")
                return {
                    "answer": AMBIGUITY_RESPONSE,
                    "sources": []
                }

            # 4. --- NEW FEDERATED SEARCH ---
            # We search BOTH the specific school AND the general index.
            all_docs = []
            stores_to_search = {}
            
            specific_store = self.vector_stores.get(school_context)
            general_store = self.vector_stores.get("general")

            if specific_store:
                stores_to_search[school_context] = specific_store
            
            if general_store and school_context != "general":
                stores_to_search["general"] = general_store
            
            if not stores_to_search:
                 logger.warning(f"No vector stores found for context '{school_context}' or 'general'.")
                 return {"answer": "I'm sorry, I don't have that specific information in my knowledge base.", "sources": []}

            for store_name, store in stores_to_search.items():
                logger.info(f"Searching index '{store_name}' for: '{query_to_search}'")
                # Use k=3 for each, max 6 docs. This is a good balance.
                all_docs.extend(store.similarity_search(query_to_search, k=3))
            # --- END FEDERATED SEARCH ---

            if not all_docs:
                logger.warning(f"No documents found in '{school_context}' or 'general' for query: '{query_to_search}'")
                return {"answer": "I'm sorry, I don't have that specific information in my knowledge base.", "sources": []}

            # 5. Build context and metadata
            context = ""
            sources = []
            source_set = set() 
            
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

            # 6. Format chat history
            formatted_history = "\n".join([f"Human: {turn.get('query', '')}\nAssistant: {turn.get('answer', '')}" for turn in chat_history])
            
            # 7. Generate the final answer
            formatted_prompt = self.rag_prompt.format(history=formatted_history, context=context, question=query)
            answer_text = self.llm.invoke(formatted_prompt)
            
            answer_text = re.sub(r'^\s*Assistant:\s*', '', answer_text).strip()
            
            return {"answer": answer_text, "sources": sources}

        except Exception as e:
            logger.error(f"RAG Error: {e}", exc_info=True)
            return {"answer": "I encountered an error while processing your request. Please try again.", "sources": []}

    def get_source_snippet(self, source_file: str, page_num: str) -> str:
        """
        Retrieves the text content for a specific source and page/row.
        """
        logger.info(f"Snippet Request: Searching for {source_file}, page/row {page_num}")

        for school, store in self.vector_stores.items():
            try:
                if hasattr(store, 'docstore') and hasattr(store.docstore, '_dict'):
                    for doc_id, doc in store.docstore._dict.items():
                        meta = doc.metadata
                        doc_source = meta.get('source', '')

                        if doc_source == source_file:
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