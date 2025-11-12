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

        # --- Deterministic school extraction rules (regex-based) ---
        # Prefer these over LLM classification when possible.
        self._school_patterns = {
            "SBM": [
                r"\bSBM\b", 
                r"\bbusiness\s+management\b", 
                r"\bschool\s+of\s+business\s+management\b",
                r"\bMBA\b",
                r"\bmaster\s+of\s+business\s+administration\b"
            ],
            "SOL": [
                r"\bSOL\b", 
                r"\blaw\b", 
                r"\bschool\s+of\s+law\b",
                r"\bBA\s+LLB\b",
                r"\bBBA\s+LLB\b",
                r"\bLLB\b",
                r"\bLLM\b",
                r"\bbachelor\s+of\s+law\b"
            ],
            "SOC": [
                r"\bSOC\b", 
                r"\bcommerce\b", 
                r"\bschool\s+of\s+commerce\b",
                r"\bBBA\b(?!\s+LLB)",  # BBA but not BBA LLB
                r"\bBBA\s+ho[u]?no[u]?rs\b",
                r"\bBBA\s+hons\b",
                r"\bbachelor\s+of\s+business\s+administration\b(?!\s+law)"
            ],
            "STME": [
                r"\bSTME\b", 
                r"\btechnology\s+management\b", 
                r"\bengineering\b", 
                r"\bschool\s+of\s+technology\s+management\b",
                r"\bB\.?Tech\b",
                r"\bB\.?E\b",
                r"\bbachelor\s+of\s+technology\b",
                r"\bCSE\b",
                r"\bcomputer\s+science\s+and\s+engineering\b",
                r"\bCS\s+&\s+DS\b",
                r"\bCSDS\b",
                r"\bcomputer\s+science\s+and\s+data\s+science\b",
                r"\bdata\s+science\b(?!.*pharmacy)",
                r"\bcomputer\s+engineering\b"
            ],
            "SPTM": [
                r"\bSPTM\b", 
                r"\bpharmacy\b", 
                r"\bschool\s+of\s+pharmacy\b", 
                r"\bschool\s+of\s+pharmacy\s+\&?\s*technology\s+management\b",
                r"\bB\.?Pharm\b",
                r"\bbachelor\s+of\s+pharmacy\b",
                r"\bpharma\s+tech\b",
                r"\bpharmaceutical\b",
                r"\bB\.?Pharm\s*\+\s*MBA\b",
                r"\bintegrated\s+b\.?pharm\b",
                r"\bintegrated\s+pharmacy\b"
            ],
        }
        
        # --- Keywords that MUST trigger ambiguity when school is not mentioned ---
        self._ambiguous_keywords = [
            r"\bacademic\s+calendar\b",
            r"\bcalendar\b",
            r"\bkey\s+dates\b",
            r"\bexam\s+dates\b",
            r"\bexam\s+schedule\b",
            r"\battendance\b",
            r"\bplacements\b",
            r"\bbook\s+list\b",
            r"\bsrb\b",
            r"\bclass\s+timings\b",
            r"\bleave\s+rules\b",
            r"\bacademic\s+year\b",
            r"\bsemester\s+dates\b",
            r"\bfee\s+structure\b",
            r"\bfaculty\b",
            r"\bcourse\s+structure\b",
            r"\bsyllabus\b",
        ]
        
        # --- Keywords that indicate general campus-wide queries ---
        self._general_keywords = [
            r"\bholiday\s+list\b",
            r"\bholidays\b",
            r"\bcampus\s+address\b",
            r"\bvice\s+chancellor\b",
            r"\bdisciplinary\s+committee\b",
            r"\bhostel\b",
            r"\bragging\s+policy\b",
            r"\banti.?ragging\b",
            r"\bgrievance\b",
            r"\bcampus\s+facilities\b",
            r"\blibrary\b",
            r"\btransport\b",
            r"\bcanteen\b",
            r"\bsecurity\b",
        ]
        
        # --- UG/PG program keywords ---
        self._ug_keywords = [
            r"\bUG\b",
            r"\bundergraduate\b",
            r"\bunder.?graduate\b",
            r"\bbachelor\b",
            r"\bB\.?\s*Tech\b",
            r"\bB\.?\s*Com\b",
            r"\bB\.?\s*Sc\b",
            r"\bB\.?\s*A\b",
            r"\bBBA\b",
            r"\bLLB\b",
        ]
        
        self._pg_keywords = [
            r"\bPG\b",
            r"\bpostgraduate\b",
            r"\bpost.?graduate\b",
            r"\bmaster\b",
            r"\bMBA\b",
            r"\bM\.?\s*Tech\b",
            r"\bM\.?\s*Com\b",
            r"\bM\.?\s*Sc\b",
            r"\bLLM\b",
        ]
        
        # UG schools (exclude SBM which is PG only)
        self._ug_schools = ["SOL", "SOC", "STME", "SPTM"]
        # PG schools
        self._pg_schools = ["SBM"]
        
        # --- Keywords for other institutions (should be rejected) ---
        self._other_institution_keywords = [
            r"\bBITS\s+Pilani\b",
            r"\bBITS\b",
            r"\bIIT\b",
            r"\bIndian\s+Institute\s+of\s+Technology\b",
            r"\bNIT\b",
            r"\bNational\s+Institute\s+of\s+Technology\b",
            r"\bIIM\b",
            r"\bIndian\s+Institute\s+of\s+Management\b",
            r"\bDU\b",
            r"\bDelhi\s+University\b",
            r"\bJNU\b",
            r"\bAIIMS\b",
            r"\bIISc\b",
            r"\bXLRI\b",
            r"\bSymbiosis\b",
            r"\bChrist\s+University\b",
            r"\bAmity\b",
            r"\bManipal\b",
            r"\bVIT\b",
            r"\bSRM\b",
            r"\bVIT\b",

        ]
        
        # --- Conversational patterns for greetings, appreciation, farewells ---
        self._greeting_patterns = [
            r"\b(hi|hello|hey|good\s+morning|good\s+afternoon|good\s+evening|greetings)\b",
            r"^(hi|hello|hey)[\s!.,?]*$",
        ]
        
        self._appreciation_patterns = [
            r"\b(thank\s*you|thanks|thank\s*u|thanx|appreciate|helpful|great|awesome|excellent|perfect|wonderful)\b",
            r"\b(that['\s]s?\s(helpful|great|good|perfect|awesome|nice))\b",
        ]
        
        self._farewell_patterns = [
            r"\b(bye|goodbye|see\s+you|take\s+care|later|cheers|farewell)\b",
            r"^(bye|goodbye)[\s!.,?]*$",
        ]
        
        self._name_request_patterns = [
            r"\b(my\s+name\s+is|i['\s]m|call\s+me|this\s+is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+))\b",
            r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)$",  # Just a name
        ]
        
        self._name_decline_patterns = [
            r"\b(no\s+name|don['\s]*t\s+want|skip|pass|rather\s+not|prefer\s+not)\b",
        ]

        # --- *** NEW SIMPLIFIED CLASSIFIER PROMPT *** ---
        # This prompt ONLY looks at the current query.
        # It no longer receives chat history, which was polluting its results.
        self.classifier_prompt = f"""
        You are a strict classifier. Based on the CURRENT QUERY ONLY, identify ONE context from this list:
        {ALL_SCHOOL_CONTEXTS + ["AMBIGUOUS"]}

        Rules:
        1. Map explicit school references and course-specific queries:
           - SBM / School of Business Management / Business Management / MBA -> SBM
           - SOL / School of Law / Law / BA LLB / BBA LLB / LLB / LLM -> SOL (NEVER map to School of Liberal Arts)
           - SOC / School of Commerce / Commerce / BBA (not BBA LLB) / BBA Honours -> SOC
           - STME / Technology Management / Engineering / B.Tech / BTech / CSE / Computer Science / CSDS / Data Science / Computer Engineering -> STME
           - SPTM / Pharmacy / B.Pharm / Pharma Tech / B.Pharm+MBA / Integrated B.Pharm / Pharmaceutical -> SPTM
        2. Course-specific mappings:
           - Queries about MBA, Master of Business Administration -> SBM
           - Queries about BA LLB, BBA LLB, LLB, LLM, law degrees -> SOL
           - Queries about BBA (without LLB), BBA Honours, commerce degrees -> SOC
           - Queries about B.Tech, engineering, CSE, computer science, data science -> STME
           - Queries about B.Pharm, pharmacy, pharmaceutical sciences -> SPTM
        3. General campus-wide queries (holiday list, campus address, vice chancellor, disciplinary committee, hostel, ragging policy, anti-ragging, grievance, library, transport, canteen) -> general
        4. School-specific topics WITHOUT school mention (academic calendar, exam dates, attendance, placements, book list, srb, class timings, leave rules, fee structure, faculty, syllabus) -> AMBIGUOUS
        5. OUTPUT: Return ONLY one token (SBM, SOL, SOC, STME, SPTM, general, AMBIGUOUS). No other words.
        6. IMPORTANT: Do NOT invent or mention "School of Liberal Arts" when SOL appears.
        7. If the query mentions a school-specific topic (like "academic calendar", "exam dates", "book list") but NO school name, you MUST return AMBIGUOUS.

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
        - Format your answer in a clear, readable structure:
          * Use bullet points for lists
          * Use numbered lists for sequential information
          * Use bold formatting for important terms or headings
          * Break long answers into short paragraphs
          * Use line breaks for readability
        - Cite your sources using.
        - Acronym mapping: SOL = School of Law (never liberal arts).

        Chat History:
        {history}

        Context:
        {context}

        Question: {question}

        Assistant:
        """
        # --- *** END NEW PROMPT *** ---
        
        self.rag_prompt = PromptTemplate(template=self.rag_prompt_template, input_variables=["history", "context", "question"])

    # --- Heuristic extractors and helpers ---
    def _extract_school_from_text(self, text: str) -> str | None:
        """Return a school code if text clearly mentions a school, else None.
        Prevent mapping 'liberal arts' to SOL unless 'law' is also present.
        """
        if not text:
            return None
        lowered = text.lower()
        if 'liberal arts' in lowered and re.search(r"\bsol\b", lowered) and 'law' not in lowered and 'school of law' not in lowered:
            return None
        for school, patterns in self._school_patterns.items():
            for pat in patterns:
                if re.search(pat, lowered, re.IGNORECASE):
                    return school
        return None
    
    def _has_ambiguous_keywords(self, text: str) -> bool:
        """Check if query contains keywords that require school context."""
        if not text:
            return False
        lowered = text.lower()
        for pattern in self._ambiguous_keywords:
            if re.search(pattern, lowered, re.IGNORECASE):
                return True
        return False
    
    def _has_general_keywords(self, text: str) -> bool:
        """Check if query contains keywords indicating general campus-wide information."""
        if not text:
            return False
        lowered = text.lower()
        for pattern in self._general_keywords:
            if re.search(pattern, lowered, re.IGNORECASE):
                return True
        return False
    
    def _has_ug_keywords(self, text: str) -> bool:
        """Check if query mentions UG/undergraduate programs."""
        if not text:
            return False
        lowered = text.lower()
        for pattern in self._ug_keywords:
            if re.search(pattern, lowered, re.IGNORECASE):
                return True
        return False
    
    def _has_pg_keywords(self, text: str) -> bool:
        """Check if query mentions PG/postgraduate programs."""
        if not text:
            return False
        lowered = text.lower()
        for pattern in self._pg_keywords:
            if re.search(pattern, lowered, re.IGNORECASE):
                return True
        return False
    
    def _mentions_other_institution(self, text: str) -> bool:
        """Check if query mentions other institutions outside NMIMS."""
        if not text:
            return False
        lowered = text.lower()
        for pattern in self._other_institution_keywords:
            if re.search(pattern, lowered, re.IGNORECASE):
                return True
        return False
    
    def _is_greeting(self, text: str) -> bool:
        """Check if message is a greeting."""
        if not text:
            return False
        lowered = text.lower().strip()
        for pattern in self._greeting_patterns:
            if re.search(pattern, lowered, re.IGNORECASE):
                return True
        return False
    
    def _is_appreciation(self, text: str) -> bool:
        """Check if message expresses appreciation."""
        if not text:
            return False
        lowered = text.lower()
        for pattern in self._appreciation_patterns:
            if re.search(pattern, lowered, re.IGNORECASE):
                return True
        return False
    
    def _is_farewell(self, text: str) -> bool:
        """Check if message is a farewell."""
        if not text:
            return False
        lowered = text.lower().strip()
        for pattern in self._farewell_patterns:
            if re.search(pattern, lowered, re.IGNORECASE):
                return True
        return False
    
    def _extract_name(self, text: str) -> str | None:
        """Extract user's name from text."""
        if not text:
            return None
        for pattern in self._name_request_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Get the last group which should be the name
                groups = match.groups()
                name = groups[-1] if groups else None
                if name and len(name.split()) <= 3:  # Reasonable name length
                    return name.strip()
        return None
    
    def _declines_name(self, text: str) -> bool:
        """Check if user declines to provide name."""
        if not text:
            return False
        lowered = text.lower()
        for pattern in self._name_decline_patterns:
            if re.search(pattern, lowered, re.IGNORECASE):
                return True
        return False

    def _normalize(self, s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip().lower())

    def _is_ambiguity_answer(self, text: str) -> bool:
        """Looser check to detect our previous disambiguation prompt regardless of tiny variations/whitespace."""
        norm = self._normalize(text)
        return "which school are you asking about" in norm

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

                # Download latest index files from S3 into the local temp folder
                try:
                    logger.info(f"Downloading index for {school} from s3://{self.bucket}/{s3_faiss_key} ...")
                    self.s3_client.download_file(self.bucket, s3_faiss_key, local_faiss_path)
                    self.s3_client.download_file(self.bucket, s3_pkl_key, local_pkl_path)
                    logger.info(f"Downloaded FAISS index for {school} to {local_dir}")
                except Exception as dl_err:
                    # If download fails, skip this store and continue with others
                    raise RuntimeError(f"Failed to download FAISS files for {school}: {dl_err}")

                stores[school] = FAISS.load_local(
                    folder_path=local_dir,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True,
                    index_name="vector_store"
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
    def get_rag_response(self, query: str, chat_history: list, user_name: str = None) -> dict:
        """
        The main RAG pipeline: Classify -> Retrieve -> Generate
        Includes conversational handling for greetings, appreciation, and farewells.
        
        *** UPDATED to return flags for dashboard logging. ***
        """
        try:
            # 0. Handle conversational patterns first
            
            # Check for greetings
            if self._is_greeting(query) and len(chat_history) == 0:
                # First greeting - ask for name
                return {
                    "answer": "Hello! Welcome to SVKM's NMIMS Hyderabad Campus Assistant. I'm here to help you with information about our campus, policies, and programs.\n\nMay I know your name? (You can share your name or type 'skip' if you prefer not to)",
                    "sources": [],
                    "conversation_type": "greeting",
                    "classified_context": "conversational",
                    "was_ambiguous": False,
                    "was_no_docs": False
                }
            elif self._is_greeting(query) and user_name:
                # Greeting with known name
                return {
                    "answer": f"Hello {user_name}! How can I assist you today with information about NMIMS Hyderabad?",
                    "sources": [],
                    "conversation_type": "greeting",
                    "classified_context": "conversational",
                    "was_ambiguous": False,
                    "was_no_docs": False
                }
            elif self._is_greeting(query):
                # Greeting without name (not first time)
                return {
                    "answer": "Hello again! How can I help you with your NMIMS queries?",
                    "sources": [],
                    "conversation_type": "greeting",
                    "classified_context": "conversational",
                    "was_ambiguous": False,
                    "was_no_docs": False
                }
            
            # Check if user is providing their name (after initial greeting)
            if len(chat_history) > 0 and chat_history[-1].get('answer', '').startswith("Hello! Welcome to SVKM's NMIMS"):
                extracted_name = self._extract_name(query)
                if extracted_name:
                    return {
                        "answer": f"Nice to meet you, {extracted_name}! I'm here to help you with any questions about NMIMS Hyderabad Campus. Feel free to ask me about academic calendars, policies, programs, facilities, or anything else related to our campus.",
                        "sources": [],
                        "conversation_type": "name_captured",
                        "user_name": extracted_name,
                        "classified_context": "conversational",
                        "was_ambiguous": False,
                        "was_no_docs": False
                    }
                elif self._declines_name(query):
                    return {
                        "answer": "No problem! I'm here to help you with any questions about NMIMS Hyderabad Campus. What would you like to know?",
                        "sources": [],
                        "conversation_type": "name_declined",
                        "classified_context": "conversational",
                        "was_ambiguous": False,
                        "was_no_docs": False
                    }
            
            # Check for appreciation
            if self._is_appreciation(query):
                appreciation_responses = [
                    f"You're very welcome{', ' + user_name if user_name else ''}! I'm glad I could help. Feel free to ask if you have any other questions about NMIMS Hyderabad.",
                    f"Happy to help{', ' + user_name if user_name else ''}! Don't hesitate to reach out if you need more information.",
                    f"My pleasure{', ' + user_name if user_name else ''}! I'm here whenever you need assistance with NMIMS-related queries.",
                ]
                # Use hash of query to deterministically pick a response
                response_idx = hash(query) % len(appreciation_responses)
                return {
                    "answer": appreciation_responses[response_idx],
                    "sources": [],
                    "conversation_type": "appreciation",
                    "classified_context": "conversational",
                    "was_ambiguous": False,
                    "was_no_docs": False
                }
            
            # Check for farewells
            if self._is_farewell(query):
                farewell_responses = [
                    f"Goodbye{', ' + user_name if user_name else ''}! Best wishes with your studies and endeavors at NMIMS Hyderabad. Feel free to return anytime you need assistance!",
                    f"Take care{', ' + user_name if user_name else ''}! Wishing you all the best at NMIMS. I'm here whenever you need help!",
                    f"See you later{', ' + user_name if user_name else ''}! Good luck with everything at NMIMS. Come back anytime!",
                ]
                response_idx = hash(query) % len(farewell_responses)
                return {
                    "answer": farewell_responses[response_idx],
                    "sources": [],
                    "conversation_type": "farewell",
                    "classified_context": "conversational",
                    "was_ambiguous": False,
                    "was_no_docs": False
                }
            
            # 1. Check if query mentions other institutions - reject immediately
            if self._mentions_other_institution(query):
                logger.info(f"Query mentions other institution (not NMIMS): {query}")
                return {
                    "answer": "I'm sorry, I can only answer questions about SVKM's NMIMS Hyderabad Campus. I don't have information about other institutions.",
                    "sources": [],
                    "classified_context": "OOS", # Out of Scope
                    "was_ambiguous": False,
                    "was_no_docs": True # Treat as a failure
                }
            
            # 2. Try deterministic school extraction first (no LLM)
            school_context = self._extract_school_from_text(query) or ""
            query_to_search = query
            search_scope = None  # Will hold "UG" or "PG" if detected

            # 1.5. Check for general keywords first - these should use "general" store
            if not school_context and self._has_general_keywords(query):
                logger.info(f"Query has general campus-wide keywords: {query}")
                school_context = "general"

            # 1.6. Check for UG/PG program keywords
            if not school_context:
                if self._has_ug_keywords(query):
                    logger.info(f"Query mentions UG programs. Will search UG schools: {self._ug_schools}")
                    search_scope = "UG"
                elif self._has_pg_keywords(query):
                    logger.info(f"Query mentions PG programs. Will search PG schools: {self._pg_schools}")
                    search_scope = "PG"

            # 1.7. Force ambiguity if query has school-specific keywords but no school mention
            if not school_context and not search_scope and self._has_ambiguous_keywords(query):
                logger.info(f"Query has ambiguous keywords without school mention: {query}")
                school_context = "AMBIGUOUS"

            # 2. Handle Follow-up Questions and short school replies using history
            if chat_history:
                last_turn = chat_history[-1]
                last_bot_answer = last_turn.get('answer', '')
                last_user_query = last_turn.get('query', '')

                # If we previously asked for school, and user just provided it, reuse the prior question
                if self._is_ambiguity_answer(last_bot_answer):
                    explicit = self._extract_school_from_text(query)
                    if explicit:
                        school_context = explicit
                        query_to_search = last_user_query
                        logger.info(f"Follow-up disambiguation: school='{school_context}', original_query='{query_to_search}'")

            # If user typed only a school token and we have history, try to resolve to previous question
            if not school_context:
                # fall back to LLM classifier only if heuristic not found
                school_context = self._classify_query(query)

                # If still ambiguous, try to mine recent history for an explicit school mention
                if school_context == "AMBIGUOUS" and chat_history:
                    for turn in reversed(chat_history[-3:]):  # look back up to 3 turns
                        mention = self._extract_school_from_text(turn.get('query', ''))
                        if mention:
                            school_context = mention
                            logger.info(f"Adopted school from history: {school_context}")
                            break

            # If the user reply is just a school name (very short), prefer the previous meaningful question
            if len(query.strip()) <= 8 and self._extract_school_from_text(query) and chat_history:
                query_to_search = chat_history[-1].get('query', query)

            # 3. Check for ambiguity
            if school_context == "AMBIGUOUS":
                logger.info(f"Ambiguous query detected: {query_to_search}")
                return {
                    "answer": AMBIGUITY_RESPONSE,
                    "sources": [],
                    "classified_context": "AMBIGUOUS",
                    "was_ambiguous": True,
                    "was_no_docs": False
                }

            # 4. --- FEDERATED SEARCH STRATEGY ---
            # For "general" queries: search ALL school SRBs + general index (common policies documented in any SRB)
            # For UG queries: search only UG schools (SOL, SOC, STME, SPTM)
            # For PG queries: search only PG schools (SBM)
            # For specific school: search that school + general index
            all_docs = []
            stores_to_search = {}
            
            if search_scope == "UG":
                # Search only UG schools + general
                logger.info(f"UG query detected. Searching UG school indexes for: '{query_to_search}'")
                for school in self._ug_schools:
                    if school in self.vector_stores:
                        stores_to_search[school] = self.vector_stores[school]
                if "general" in self.vector_stores:
                    stores_to_search["general"] = self.vector_stores["general"]
            
            elif search_scope == "PG":
                # Search only PG schools + general
                logger.info(f"PG query detected. Searching PG school indexes for: '{query_to_search}'")
                for school in self._pg_schools:
                    if school in self.vector_stores:
                        stores_to_search[school] = self.vector_stores[school]
                if "general" in self.vector_stores:
                    stores_to_search["general"] = self.vector_stores["general"]
            
            elif school_context == "general":
                # Search across ALL stores to find common policies documented in any SRB
                logger.info(f"General query detected. Searching ALL school indexes + general for: '{query_to_search}'")
                stores_to_search = self.vector_stores.copy()  # Search all available stores
            
            else:
                # Specific school query: search that school + general
                specific_store = self.vector_stores.get(school_context)
                general_store = self.vector_stores.get("general")

                if specific_store:
                    stores_to_search[school_context] = specific_store
                
                if general_store:
                    stores_to_search["general"] = general_store
            
            if not stores_to_search:
                 logger.warning(f"No vector stores found for context '{school_context}'.")
                 return {
                     "answer": "I'm sorry, I don't have that specific information in my knowledge base.", 
                     "sources": [],
                     "classified_context": school_context,
                     "was_ambiguous": False,
                     "was_no_docs": True
                 }

            for store_name, store in stores_to_search.items():
                logger.info(f"Searching index '{store_name}' for: '{query_to_search}'")
                # Use k=1 for multi-school searches (cleaner UI), k=2 for specific school queries
                k_value = 1 if (school_context == "general" or search_scope) else 2
                all_docs.extend(store.similarity_search(query_to_search, k=k_value))
            # --- END FEDERATED SEARCH ---


            if not all_docs:
                logger.warning(f"No documents found in '{school_context}' or 'general' for query: '{query_to_search}'")
                return {
                    "answer": "I'm sorry, I don't have that specific information in my knowledge base.", 
                    "sources": [],
                    "classified_context": school_context,
                    "was_ambiguous": False,
                    "was_no_docs": True
                }

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
            formatted_prompt = self.rag_prompt.format(history=formatted_history, context=context, question=query_to_search)
            answer_text = self.llm.invoke(formatted_prompt)
            # Sanitize incorrect expansion of SOL
            if school_context == 'SOL' and 'liberal arts' in answer_text.lower():
                answer_text = re.sub(r'(?i)school\s+of\s+liberal\s+arts', 'School of Law', answer_text)
            
            answer_text = re.sub(r'^\s*Assistant:\s*', '', answer_text).strip()
            
            return {
                "answer": answer_text, 
                "sources": sources,
                "classified_context": school_context,
                "was_ambiguous": False,
                "was_no_docs": False
            }

        except Exception as e:
            logger.error(f"RAG Error: {e}", exc_info=True)
            return {
                "answer": "I encountered an error while processing your request. Please try again.", 
                "sources": [],
                "classified_context": "error",
                "was_ambiguous": False,
                "was_no_docs": True # Treat errors as a failure
            }

    def get_source_snippet(self, source_file: str, page_num: str) -> str:
        """
        Retrieves a concise snippet (max 500 chars) for a specific source and page/row.
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
                                content = doc.page_content.strip()
                                # Limit snippet to 500 characters for better readability
                                if len(content) > 500:
                                    content = content[:500] + "..."
                                return content
                else:
                    logger.warning(f"Store {school} has no accessible .docstore._dict. Cannot retrieve snippet.")

            except Exception as e:
                logger.error(f"Error searching docstore for {school}: {e}", exc_info=True)

        logger.warning(f"Snippet not found for {source_file}, page {page_num}")
        return f"Sorry, the snippet for {os.path.basename(source_file)} (Page {page_num}) could not be retrieved."