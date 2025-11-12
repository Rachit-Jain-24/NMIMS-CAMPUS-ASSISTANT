# NMIMS Campus Assistant (RAG Chatbot)

This project is a sophisticated, dual-application **Retrieval-Augmented Generation (RAG)** chatbot designed to serve as a campus assistant for NMIMS Hyderabad.

The system is architecturally split into two main components:

1.  **Admin Application:** A password-protected Flask web interface that allows an administrator to upload, manage, and build the knowledge base. It also features an analytics dashboard to monitor user queries and bot performance.
2.  **User Application:** A Flask-based API that serves the end-user chatbot. It handles user queries in multiple languages, transcribes audio, and intelligently routes questions to the correct, school-specific knowledge base to generate answers.

A key feature of this architecture is its **federated vector store** model. Instead of a single, monolithic knowledge base, the system maintains separate vector stores for each school (SBM, SOL, STME, etc.) and for "general" campus information. This allows for more precise, context-aware, and ambiguity-free answers.

## Features

### User-Facing App (`User/`)

  * **Multi-Language Support:** Automatically translates non-English queries to English for processing and translates the final answer back to the user's original language using `deep_translator`.
  * **Voice-to-Text:** Provides an endpoint (`/api/transcribe`) that uses `openai-whisper` to transcribe audio inputs into text.
  * **Intelligent Query Classification:** A multi-step process to understand user intent:
    1.  **Conversational:** First checks for simple greetings, farewells, or appreciation using regex to provide fast, non-LLM answers.
    2.  **Heuristic:** Uses regex rules to quickly identify explicit school names (e.g., "SBM", "STME") or general-intent keywords (e.g., "hostel", "holiday list").
    3.  **LLM-Based:** If heuristics fail, it uses an LLM (Mistral) with a strict prompt to classify the query into a specific school context or mark it as `AMBIGUOUS`.
  * **Federated Search:**
      * Queries for a specific school (e.g., "STME syllabus") search *only* the `STME` and `general` vector stores.
      * General queries (e.g., "hostel rules") search the `general` store and *all* school-specific stores (as the rule might be in a school's SRB).
      * Ambiguous queries (e.g., "When are exams?") are not searched; the bot replies by asking for clarification (e.g., "Which school are you asking about?").
  * **Feedback Mechanism:** A `/api/feedback` endpoint allows users to "up" or "down" vote a response. This feedback is logged in the database and displayed on the admin dashboard.

### Admin Panel (`Admin/`)

  * **Secure Admin Interface:** A Flask-based web app protected by a username and hashed password login (using `Flask-Login` and `Bcrypt`).
  * **Knowledge Base Management:**
      * **Incremental Upload ("Fast Path"):** An admin can upload a new file (PDF, CSV, DOCX, etc.) and tag it with a `school_context` and `doc_type`. This downloads *only* the relevant vector index from S3, adds the new document, and re-uploads the updated index.
      * **Full Rebuild ("Slow Path"):** A single-click button (`/rebuild`) to download *all* source documents from S3, re-process everything from scratch, and rebuild *all* federated vector stores. This is used after deleting files to ensure they are removed from the index.
      * **Delete & Clear:** Secure endpoints to delete a single source file (`/delete`) or wipe the *entire* knowledge base, including all S3 files (`/clear`).
  * **Robust Document Processing:**
      * **PDFs:** Uses **AWS Textract** as the primary method for high-accuracy OCR. It includes a fallback to `PyPDF` if Textract fails.
      * **Excel/CSV:** Uses `pandas` to convert each row into a separate document, ideal for structured data like library book lists.
  * **Live Analytics Dashboard (`/dashboard`):**
      * Connects to the shared SQL database to provide a real-time view of chatbot usage.
      * **KPIs:** Displays total queries, failure rate (ambiguous or no-doc answers), and good feedback rate.
      * **Query Tables:** Shows "Top Queries" (most frequently asked) and "Failed Queries" (recent ambiguous/no-doc/disliked) to identify knowledge gaps.
      * **Charts:** Visualizes "Queries by School" and "Query Status" (Successful, Disliked, Ambiguous).
  * **Live Reload Trigger:** After the Admin app uploads a new index to S3, it sends a secure, authorized POST request to the User app's `/api/refresh-knowledge-base` endpoint. This tells the User app to immediately dump its old in-memory index and reload the new one from S3, ensuring the chatbot is updated in real-time without a restart.

## Architecture & Data Flow

1.  **Admin:** An admin logs into the `Admin` app (`http://localhost:5000`) and uploads `STME_Academic_Calendar.pdf`, tagging it for the `STME` school.
2.  **`Admin/backend_processor.py`:** The backend processes the PDF using AWS Textract, splits it into chunks, embeds them using Amazon Titan, downloads the existing `STME` FAISS index from S3, adds the new chunks, and uploads the updated index back to S3.
3.  **Live Reload:** The Admin app pings the `User` app's `/api/refresh-knowledge-base` endpoint.
4.  **`User/rag_backend.py`:** The User app receives the signal and re-downloads the `STME` index from S3 into its in-memory FAISS store.
5.  **User:** A user sends a query (e.g., "When are the STME exams?") to the `User` app's `/api/chat` endpoint (`http://localhost:8086`).
6.  **`User/rag_backend.py`:**
      * The query is classified (heuristically or via LLM) as belonging to `STME`.
      * A similarity search is performed *only* on the `STME` and `general` vector stores.
      * The relevant chunks from `STME_Academic_Calendar.pdf` are retrieved.
      * The chunks, query, and chat history are passed to the Mistral LLM to generate a natural language answer.
7.  **`User/app.py`:**
      * The English query, the English answer, and the classified context (`STME`) are logged to the SQL database.
      * The final answer is sent back to the user.
8.  **Admin:** The admin can now visit the `/dashboard` on their app and see the new "STME" query reflected in the analytics.

## Technology Stack

  * **Backend:** Flask, Flask-Login, Flask-SQLAlchemy, Flask-Bcrypt
  * **AI & LLM Orchestration:** LangChain
  * **LLM:** AWS Bedrock (e.g., `mistral.mixtral-8x7b-instruct-v0:1`)
  * **Embeddings:** AWS Bedrock (e.g., `amazon.titan-embed-text-v2:0`)
  * **Vector Store:** FAISS (CPU)
  * **Cloud & Storage:** AWS S3 (for source documents and FAISS indexes)
  * **Document Processing:** AWS Textract (primary PDF OCR), `pypdf` (fallback), `pandas` (Excel/CSV), `docx2txt` (Word), `unstructured` (PowerPoint)
  * **Database:** PostgreSQL (inferred from `psycopg2-binary`)
  * **User Features:** `openai-whisper` (Audio Transcription), `deep_translator` (Multi-language)

## Setup and Installation

### Prerequisites

  * Python (3.9+ recommended)
  * An AWS Account with permissions for:
      * **S3** (to create/read/write to a bucket)
      * **Bedrock** (to access Titan embedding and Mistral LLM models)
      * **Textract** (for PDF processing)
  * A PostgreSQL Database

### 1\. Clone the Repository

```bash
git clone <repository-url>
cd NMIMS-CAMPUS-ASSISTANT
```

### 2\. Set Up Virtual Environments

It is highly recommended to use separate virtual environments for the two apps, as they have different dependencies.

```bash
# Set up Admin venv
python -m venv venv_admin
source venv_admin/bin/activate
pip install -r Admin/requirements.txt

# Set up User venv in a separate terminal
python -m venv venv_user
source venv_user/bin/activate
pip install -r User/requirements.txt
```

### 3\. Configure Environment Variables

Create a `.env` file in the *root* directory. Both applications will load this file.

```ini
# --- Database ---
# (Must be accessible by both apps)
DATABASE_URL="postgresql://YOUR_DB_USER:YOUR_DB_PASSWORD@YOUR_DB_HOST:5432/YOUR_DB_NAME"

# --- Flask ---
# (Must be THE SAME for both apps for the refresh-key to work)
FLASK_SECRET_KEY="your_very_strong_random_secret_key"

# --- AWS Credentials ---
AWS_ACCESS_KEY_ID="your_aws_access_key"
AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
AWS_DEFAULT_REGION="your_aws_region" # e.g., ap-south-1

# --- S3 Bucket ---
BUCKET_NAME="your-s3-bucket-name-for-docs-and-indexes"

# --- Bedrock Models ---
# (These are the defaults, change if needed)
BEDROCK_EMBEDDING_MODEL_ID="amazon.titan-embed-text-v2:0"
BEDROCK_LLM_MODEL_ID="mistral.mixtral-8x7b-instruct-v0:1"

# --- Admin Login ---
ADMIN_USERNAME="admin"
# Generate a bcrypt hash of your desired password and paste it here
ADMIN_HASHED_PASSWORD="your_bcrypt_hashed_password"
```

### 4\. Initialize the Database

Before running the apps, you need to create the database tables defined in `models.py`. You can do this with a simple Python script.

*(In one of your activated virtual environments, e.g., `venv_admin`):*

```python
# run_create_db.py
from dotenv import load_dotenv
load_dotenv() # Load the .env file

from Admin.app import app, db
# Or from User.app import app, db
# They share the same models

with app.app_context():
    print("Creating database tables...")
    db.create_all()
    print("Done.")
```

Now run this script:

```bash
python run_create_db.py
```

## Running the Application

You must run both applications simultaneously in separate terminals.

### Terminal 1: Run the User App

```bash
# For Windows PowerShell
./startuser.ps1

# For Linux/MacOS Terminal
./startuser.sh
```

*(The User App will be live at `http://localhost:8086`)*

### Terminal 2: Run the Admin App

```bash
# For Windows PowerShell
./startadmin.ps1

# For Linux/MacOS Terminal
./startadmin.sh
```

*(The Admin App will be live at `http://localhost:5000`)*