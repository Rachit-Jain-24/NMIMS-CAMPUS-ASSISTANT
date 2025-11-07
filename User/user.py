import os
import boto3
from botocore.exceptions import ClientError
import logging
import tempfile
import streamlit as st
import re
from dotenv import load_dotenv
from PIL import Image
import base64

# LangChain components
from langchain_core.prompts import PromptTemplate
from langchain_aws import BedrockEmbeddings, BedrockLLM
from langchain_community.vectorstores import FAISS

# --- NEW IMPORTS FOR MULTILINGUAL + VOICE ---
import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from deep_translator import GoogleTranslator


# Load environment
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config aligned with Admin app (overridable via environment)
EMBEDDING_MODEL_ID = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
LLM_MODEL_ID = os.getenv("BEDROCK_LLM_MODEL_ID", "mistral.mixtral-8x7b-instruct-v0:1")

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

FAISS_S3_KEY = "nmims_rag/vector_store.faiss"
PKL_S3_KEY = "nmims_rag/vector_store.pkl"

LOCAL_INDEX_DIR = tempfile.gettempdir()
LOCAL_INDEX_NAME = "nmims_rag_index"

if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, BUCKET_NAME]):
    st.error("Missing AWS config. Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, BUCKET_NAME in environment/.env")
    st.stop()

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
        model_id=EMBEDDING_MODEL_ID,
        client=bedrock_client,
        region_name=AWS_REGION
    )
except Exception as e:
    st.error(f"AWS initialization failed: {e}")
    st.stop()

@st.cache_resource(show_spinner="Loading knowledge base...")
def load_vector_store():
    try:
        # Check if files exist in S3 before attempting download
        try:
            s3_client.head_object(Bucket=BUCKET_NAME, Key=FAISS_S3_KEY)
            s3_client.head_object(Bucket=BUCKET_NAME, Key=PKL_S3_KEY)
            logger.info(f"Vector store files found in S3: {FAISS_S3_KEY}, {PKL_S3_KEY}")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == '404' or error_code == 'NoSuchKey':
                error_msg = f"Vector store files not found in S3 bucket '{BUCKET_NAME}'. Please upload a document using the Admin portal first."
                logger.error(error_msg)
                st.error(error_msg)
                st.info(f"📝 **Next Steps:**\n1. Go to the Admin portal\n2. Upload a PDF document\n3. Wait for processing to complete\n4. Click 'Reload Knowledge Base' button")
                return None
            elif error_code == '403':
                error_msg = f"Permission denied accessing S3 bucket '{BUCKET_NAME}'. Please check your AWS credentials."
                logger.error(error_msg)
                st.error(error_msg)
                return None
            else:
                error_msg = f"Error checking S3 files: {error_code} - {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
                return None
        
        local_faiss_path = os.path.join(LOCAL_INDEX_DIR, f"{LOCAL_INDEX_NAME}.faiss")
        local_pkl_path = os.path.join(LOCAL_INDEX_DIR, f"{LOCAL_INDEX_NAME}.pkl")

        logger.info(f"Downloading vector store files from S3...")
        s3_client.download_file(BUCKET_NAME, FAISS_S3_KEY, local_faiss_path)
        s3_client.download_file(BUCKET_NAME, PKL_S3_KEY, local_pkl_path)
        logger.info(f"Files downloaded successfully to: {LOCAL_INDEX_DIR}")

        # Verify downloaded files exist
        if not os.path.exists(local_faiss_path) or not os.path.exists(local_pkl_path):
            error_msg = "Downloaded files not found locally. Please try again."
            logger.error(error_msg)
            st.error(error_msg)
            return None

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
        logger.exception(f"Failed to load vector store")
        error_msg = f"Could not load the knowledge base: {str(e)}"
        st.error(error_msg)
        st.info("💡 **Troubleshooting:**\n1. Check if vector store files exist in S3\n2. Verify AWS credentials are correct\n3. Check network connectivity\n4. Try clearing the cache and reloading")
        return None

def get_llm():
    try:
        logger.info(f"Initializing BedrockLLM with model_id='{LLM_MODEL_ID}' and region='{AWS_REGION}'")
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
        st.error(f"LLM initialization failed: {e}")
        return None

def get_rag_response(llm, vectorstore, question: str, k: int = 3) -> tuple[str, list]:
    guardrail_no_info = "I don't have that information in the NMIMS knowledge base."
    prompt_template = """
    Human: You are an assistant for NMIMS students and staff.
    Your job is to answer the user's question using ONLY the provided context.
    Rules:
    - If the answer is not present in the context, respond exactly with: "I don't have that information in the NMIMS knowledge base."
    - Be concise. Start with a 1–2 sentence direct answer, then add 2–5 short bullet points if helpful.
    - Quote specific policy or rule text when appropriate.
    - When you reference content, add page markers like [Page X] where X is the page number shown in the context.
    - Never invent information beyond the context.

    Context (each chunk may include a page marker):
    {context}

    Question: {question}

    Assistant:"
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    try:
        docs = vectorstore.similarity_search(question, k=k)
        if not docs:
            return guardrail_no_info, []
        # Build context with page markers to improve grounded citations
        context_parts = []
        for d in docs:
            page = d.metadata.get("page", "?")
            snippet = (d.page_content or "").strip()
            context_parts.append(f"[Page {page}]\n{snippet}")
        context = "\n\n".join(context_parts)
        if not context.strip():
            return guardrail_no_info, docs
        formatted_prompt = PROMPT.format(context=context, question=question)
        logger.info(f"Invoking LLM (model_id='{LLM_MODEL_ID}') with prompt length={len(formatted_prompt)}")
        try:
            response = llm.invoke(formatted_prompt)
        except Exception as e:
            # Log full exception with traceback to help debug validation errors from Bedrock
            logger.exception("Error while invoking LLM")
            raise
        text = response if isinstance(response, str) else str(response)
        if not text.strip():
            return guardrail_no_info, docs
        return text, docs
    except Exception as e:
        return f"I encountered an error while processing your request: {e}", []
def _format_answer_for_lists(text: str) -> str:
    """Ensure bullets/numbers render with proper line breaks in markdown."""
    try:
        # Insert newline before hyphen bullets if missing
        text = re.sub(r"(?<!\n)\s*-\s+", lambda m: "\n" + m.group(0).lstrip(), text)
        # Insert newline before asterisk bullets if missing
        text = re.sub(r"(?<!\n)\s*\*\s+", lambda m: "\n" + m.group(0).lstrip(), text)
        # Insert newline before numbered bullets like '1.' or '1)'
        text = re.sub(r"(?<!\n)(\s*)(\d{1,2}[\.)]\s+)", lambda m: "\n" + m.group(2), text)
        return text
    except Exception:
        return text

def get_logo_base64():
    """Convert logo to base64 for embedding in HTML."""
    try:
        if os.path.exists("logo.jpg"):
            with open("logo.jpg", "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        # Fallback SVG logo if file not found
        return "PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgdmlld0JveD0iMCAwIDEwMCAxMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIxMDAiIGhlaWdodD0iMTAwIiBmaWxsPSIjMDAwMDAwIi8+CjxyZWN0IHg9IjMzLjMiIHdpZHRoPSIzMy4zIiBoZWlnaHQ9IjEwMCIgZmlsbD0iI2ZmZmZmZiIvPgo8cmVjdCB4PSI2Ni42IiB3aWR0aD0iMzMuNCIgaGVpZ2h0PSIxMDAiIGZpbGw9IiM2NjY2NjYiLz4KPC9zdmc+Cg=="
    except Exception as e:
        st.error(f"Logo loading error: {e}")
        return "PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgdmlld0JveD0iMCAwIDEwMCAxMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIxMDAiIGhlaWdodD0iMTAwIiBmaWxsPSIjMDAwMDAwIi8+CjxyZWN0IHg9IjMzLjMiIHdpZHRoPSIzMy4zIiBoZWlnaHQ9IjEwMCIgZmlsbD0iI2ZmZmZmZiIvPgo8cmVjdCB4PSI2Ni42IiB3aWR0aD0iMzMuNCIgaGVpZ2h0PSIxMDAiIGZpbGw9IiM2NjY2NjYiLz4KPC9zdmc+Cg=="

LANG_CODES = {"English": "en", "Hindi": "hi", "Telugu": "te"}

def translate_text(text, target_lang):
    try:
        return GoogleTranslator(source='auto', target=LANG_CODES[target_lang]).translate(text)
    except:
        return text  # fallback if translation service fails


def transcribe_speech():
    st.info("🎤 Listening... Speak now.")

    audio = sd.rec(int(16000 * 4), samplerate=16000, channels=1, dtype='int16')
    sd.wait()

    if st.session_state.lang == "English":
        model_path = r"D:\Data\Rachit\NMIMS\Sem 7\Capstone Project\work\Project - Copy\User\models\english\vosk-model-en-in-0.5"
    elif st.session_state.lang == "Hindi":
        model_path = r"D:\Data\Rachit\NMIMS\Sem 7\Capstone Project\work\Project - Copy\User\models\hindi\vosk-model-small-hi-0.22"
    else:
        model_path = r"D:\Data\Rachit\NMIMS\Sem 7\Capstone Project\work\Project - Copy\User\models\telugu\vosk-model-small-te-0.42"

    model = Model(model_path)
    recognizer = KaldiRecognizer(model, 16000)
    recognizer.AcceptWaveform(audio.tobytes())
    result = json.loads(recognizer.Result())
    return result.get("text", "").strip()

def main():
    st.set_page_config(page_title="NMIMS Campus Assistant", page_icon="🎓", layout="wide")

    # NMIMS Official Style Header
    st.markdown(
        """
        <style>
        .nmims-header {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 50%, #dee2e6 100%);
            border-bottom: 3px solid #d32f2f;
            padding: 20px 25px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1), 0 2px 4px rgba(0,0,0,0.06);
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
        .nmims-chatbot-badge {
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
        </style>
        <div class="nmims-header">
          <div class="nmims-logo-section">
            <div class="nmims-shield"></div>
            <div>
              <div class="nmims-title">SVKM'S NMIMS Deemed to be UNIVERSITY</div>
              <div class="nmims-subtitle">HYDERABAD</div>
            </div>
          </div>
          <div class="nmims-chatbot-badge">🤖 NMIMS Campus Assistant</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar with information and settings
    with st.sidebar:
        # Welcome Information Section
        st.header("📋 Information")
        st.subheader("🌐 Language")
        st.session_state.lang = st.selectbox("Choose Chat Language:", ["English", "Hindi", "Telugu"])
        st.markdown("**🏠 Welcome to NMIMS Hyderabad's AI Assistant!**")
        st.markdown("Ask questions about:")
        st.markdown("• Academic policies")
        st.markdown("• Course information") 
        st.markdown("• Campus resources")
        st.markdown("• Student Facilities")
        st.markdown("• Admission procedures")
        st.markdown("• Fee structure")
        
        
        
        # st.divider()
        
        # # Contact Information
        # st.subheader("📞 Contact")
        # st.markdown("**NMIMS Hyderabad**")
        # st.markdown("📍 Address: Survey No. 102, Shamirpet")
        # st.markdown("📧 Email: info@nmims.edu")
        # st.markdown("📱 Phone: +91-40-2726-0000")
        # st.markdown("🌐 Website: hyderabad.nmims.edu")
        
        st.divider()
        
        # Chat Settings
        st.header("⚙️ Settings")
        top_k = st.slider("Results to retrieve", min_value=1, max_value=8, value=3, step=1)
        show_sources = st.checkbox("Show sources", value=True)
        
        st.divider()
        
        # Technical Info
        st.subheader("🔧 Technical")
        st.caption("Powered by Amazon Bedrock + FAISS on S3")
        st.caption(f"Embeddings: {EMBEDDING_MODEL_ID}")
        st.caption(f"LLM: {LLM_MODEL_ID}")
        
        if st.button("🗑️ Clear chat"):
            st.session_state.pop("messages", None)
            st.rerun()
        
        st.divider()
        
        # Reload Knowledge Base button
        if st.button("🔄 Reload Knowledge Base"):
            # Clear the cache for load_vector_store
            load_vector_store.clear()
            st.success("Cache cleared! Reloading knowledge base...")
            st.rerun()

    # Load the vector store (knowledge base)
    vector_store = load_vector_store()
    if not vector_store:
        # Show a helpful message instead of blank page
        st.warning("⚠️ **Knowledge Base Not Available**")
        st.markdown("""
        The chatbot knowledge base could not be loaded. This might happen if:
        - No documents have been uploaded yet via the Admin portal
        - There's an issue with AWS S3 connectivity
        - AWS credentials need to be configured
        
        **What you can do:**
        1. Verify that vector store files exist in S3 bucket: `{bucket_name}`
        2. Check AWS credentials in your `.env` file or Streamlit secrets
        3. Try clicking the "🔄 Reload Knowledge Base" button in the sidebar
        4. Contact your administrator if the problem persists
        """.format(bucket_name=BUCKET_NAME))
        return

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render history with avatars
    for role, content in st.session_state.messages:
        avatar = "🧑" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(content)

    # Example questions to copy
    if not st.session_state.messages:
        st.markdown("#### Example questions (copy and paste)")
        st.markdown(
            "- What are the key academic calendar deadlines?\n"
            "- What is the NMIMS Ragging policy?\n"
            "- What is the minimum attendance required for a student to appear in the exams?\n"
            "- What are the fees and refund rules?\n"
            "- How do I apply for an official transcript?\n"
            "- What is the grading policy for UG programs?"
        )

    # Chat input
    col1, col2 = st.columns([8,2])
    with col1:
        user_input = st.chat_input("Type your question about NMIMS...")

    with col2:
        if st.button("🎤 Voice"):
            user_input = transcribe_speech()
            if user_input:
                st.session_state.messages.append(("user", f"🎙 {user_input}"))
                st.chat_message("user", avatar="🧑").markdown(f"🎙 {user_input}")

    if user_input:
        st.session_state.messages.append(("user", user_input))
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            # Show typing indicator
            with st.spinner("🤖 Assistant is typing..."):
                llm = get_llm()
                if not llm:
                    st.error("Language model unavailable. Please try again later.")
                else:
                    translated_query = translate_text(user_input, "English")
                    answer, docs = get_rag_response(llm, vector_store, translated_query, k=top_k)
                    answer = translate_text(answer, st.session_state.lang)

            
            # Display the response
            display_answer = _format_answer_for_lists(answer)
            st.markdown(display_answer)
            if show_sources and docs:
                with st.expander("📚 Sources"):
                    for i, d in enumerate(docs, start=1):
                        preview = (d.page_content or "").strip().replace("\n", " ")
                        if len(preview) > 240:
                            preview = preview[:240] + "..."
                        page = d.metadata.get("page", "?")
                        st.markdown(f"**Source {i} • Page {page}:** {preview}")
        st.session_state.messages.append(("assistant", display_answer))



if __name__ == "__main__":
    main()


