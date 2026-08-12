# NMIMS Campus Assistant — RAG Chatbot

An AI-powered **Retrieval-Augmented Generation (RAG)** campus assistant built for NMIMS Hyderabad.

The system uses a **federated vector store architecture**, maintaining separate knowledge bases for different schools such as SBM, SOL, STME, along with a general campus knowledge base. This helps provide more accurate and context-aware responses.

---

## Overview

The project consists of two applications:

* **Admin Application** — Manage documents, build knowledge bases, and monitor chatbot usage.
* **User Application** — Chat with the AI assistant using text or voice, with support for multiple languages.

### System Architecture

```text
                         NMIMS Campus Assistant
                                  |
                    +-------------+-------------+
                    |                           |
              Admin Application          User Application
                    |                           |
            Document Management          User Queries
                    |                           |
              AWS S3 + Textract          Query Classification
                    |                           |
              Document Processing        Federated Retrieval
                    |                           |
              FAISS Vector Stores       School + General
                    |                           |
                    +-------------+-------------+
                                  |
                            AWS Bedrock
                         Mistral + Titan
                                  |
                              Response
```

---

# Screenshots

## User Application

### Chat Interface

<p align="center">
  <img src="screenshots/user_chat_interface.png" width="90%">
</p>

The chatbot provides a conversational interface for asking questions about academics, hostels, policies, events, placements, and other campus information.

---

### Conversational Flow

<p align="center">
  <img src="screenshots/user_conversational_flow.png" width="90%">
</p>

The system handles different types of user interactions and determines the appropriate processing path before generating a response.

---

### Query Classification

<p align="center">
  <img src="screenshots/query_classification.png" width="90%">
</p>

The chatbot classifies queries before retrieval to determine the appropriate school-specific knowledge base.

For example:

```text
"STME syllabus"
       ↓
Query Classification
       ↓
STME Context
       ↓
STME + General Vector Stores
       ↓
Relevant Documents
       ↓
LLM Response
```

---

## Admin Application

### Admin Dashboard

<p align="center">
  <img src="screenshots/adminportal_enhanced.png" width="90%">
</p>

The admin application provides a secure interface for managing the knowledge base and monitoring chatbot activity.

---

### Knowledge Base Management

<p align="center">
  <img src="screenshots/knowledgebase_interface.png" width="90%">
</p>

Administrators can upload and manage documents that are used by the RAG pipeline.

Documents can be associated with:

* Specific schools
* General campus information
* Different document types

---

# Key Features

## 1. Federated RAG Architecture

Instead of using a single large vector database, the system maintains separate FAISS indexes for different school contexts.

```text
S3
│
├── general/
│   └── general.faiss
│
├── SBM/
│   └── sbm.faiss
│
├── SOL/
│   └── sol.faiss
│
└── STME/
    └── stme.faiss
```

This allows the system to retrieve information from the most relevant knowledge bases.

### Example

For:

```text
"STME syllabus"
```

the system searches:

```text
STME + General
```

rather than searching every available document.

---

## 2. Intelligent Query Classification

The system uses a multi-step classification process.

### Conversational Detection

Simple interactions such as:

```text
Hello
Thank you
Goodbye
```

are handled directly without unnecessary LLM calls.

### Heuristic Classification

Explicit school names and common keywords are detected using rules.

Example:

```text
"STME syllabus"
        ↓
STME
```

### LLM Classification

If the query cannot be confidently classified using heuristics, an LLM is used to determine the appropriate context.

For ambiguous queries such as:

```text
"When are the exams?"
```

the system can request clarification rather than retrieving information from the wrong school.

---

# 3. Federated Search

The retrieval strategy depends on the classified context.

| Query Type      | Vector Stores             |
| --------------- | ------------------------- |
| School-specific | Relevant School + General |
| General         | General + School Stores   |
| Ambiguous       | No Retrieval              |

This helps reduce irrelevant context and improve retrieval precision.

---

# 4. Document Processing Pipeline

```text
Document Upload
      ↓
File Type Detection
      ↓
Document Extraction
      ↓
Text Chunking
      ↓
Amazon Titan Embeddings
      ↓
FAISS Index
      ↓
AWS S3
```

### Supported Processing

* **PDF** → AWS Textract with PyPDF fallback
* **CSV / Excel** → Pandas
* **DOCX** → docx2txt
* **PowerPoint** → Unstructured

---

# 5. Live Knowledge Base Reload

The chatbot can update its knowledge base without requiring a restart.

```text
Admin uploads document
        ↓
Document processed
        ↓
FAISS index updated
        ↓
Index uploaded to S3
        ↓
Refresh API triggered
        ↓
User application reloads index
        ↓
New knowledge becomes available
```

This allows administrators to update campus information while the chatbot remains running.

---

# 6. Feedback System

Users can provide feedback on generated responses.

```text
User Response
     ↓
👍 / 👎
     ↓
PostgreSQL
     ↓
Admin Dashboard
```

Feedback can be used to identify poor responses and knowledge gaps.

---

# Architecture & Data Flow

```text
                         USER
                           |
                           ↓
                    /api/chat
                           |
                           ↓
                 Query Classification
                           |
             +-------------+-------------+
             |             |             |
       Conversational   Heuristic       LLM
             |             |             |
             +-------------+-------------+
                           |
                           ↓
                  School Context
                           |
                           ↓
                 Federated Retrieval
                           |
             +-------------+-------------+
             |             |             |
           SBM           STME          General
             |             |             |
             +-------------+-------------+
                           |
                           ↓
                    FAISS Search
                           |
                           ↓
                   Relevant Chunks
                           |
                           ↓
                     AWS Bedrock
                        Mistral
                           |
                           ↓
                     Final Answer
                           |
                           ↓
                     PostgreSQL
                           |
                           ↓
                   Admin Analytics
```

---

# Admin Data Pipeline

```text
Admin Upload
     ↓
AWS S3
     ↓
Document Processing
     ↓
AWS Textract / Pandas / PyPDF
     ↓
Text Chunking
     ↓
Amazon Titan Embeddings
     ↓
FAISS
     ↓
AWS S3
     ↓
Live Reload
     ↓
User Application
```

---

# Technology Stack

### Backend

* Python
* Flask
* Flask-SQLAlchemy
* Flask-Login
* Flask-Bcrypt

### AI & RAG

* LangChain
* AWS Bedrock
* Mistral / Mixtral
* Amazon Titan Embeddings
* FAISS

### Cloud

* AWS S3
* AWS Textract
* AWS Bedrock

### Database

* PostgreSQL

### Document Processing

* PyPDF
* Pandas
* docx2txt
* Unstructured

### User Features

* OpenAI Whisper
* Deep Translator

---

# Project Structure

```text
NMIMS-CAMPUS-ASSISTANT/
│
├── Admin/
│   ├── app.py
│   ├── backend_processor.py
│   ├── models.py
│   └── ...
│
├── User/
│   ├── app.py
│   ├── rag_backend.py
│   └── ...
│
├── screenshots/
│   ├── adminportal_enhanced.png
│   ├── knowledgebase_interface.png
│   ├── query_classification.png
│   ├── user_chat_interface.png
│   └── user_conversational_flow.png
│
├── .gitignore
├── README.md
├── requirements.txt
├── run_create_db.py
├── startadmin.ps1
├── startadmin.sh
├── startuser.ps1
└── startuser.sh
```

---

# Setup & Installation

## Prerequisites

* Python 3.9+
* AWS Account
* Amazon S3 access
* Amazon Bedrock access
* AWS Textract access
* PostgreSQL database

---

## 1. Clone the Repository

```bash
git clone <repository-url>
cd NMIMS-CAMPUS-ASSISTANT
```

---

## 2. Create Virtual Environments

The Admin and User applications use separate environments.

### Admin

```bash
python -m venv venv_admin
```

Windows:

```bash
venv_admin\Scripts\activate
```

Linux / macOS:

```bash
source venv_admin/bin/activate
```

```bash
pip install -r Admin/requirements.txt
```

### User

```bash
python -m venv venv_user
```

Windows:

```bash
venv_user\Scripts\activate
```

Linux / macOS:

```bash
source venv_user/bin/activate
```

```bash
pip install -r User/requirements.txt
```

---

# 3. Environment Variables

Create a `.env` file in the project root.

```env
DATABASE_URL="postgresql://YOUR_DB_USER:YOUR_DB_PASSWORD@YOUR_DB_HOST:5432/YOUR_DB_NAME"

FLASK_SECRET_KEY="your_very_strong_random_secret_key"

AWS_ACCESS_KEY_ID="your_aws_access_key"
AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
AWS_DEFAULT_REGION="your_aws_region"

BUCKET_NAME="your-s3-bucket-name"

BEDROCK_EMBEDDING_MODEL_ID="amazon.titan-embed-text-v2:0"
BEDROCK_LLM_MODEL_ID="mistral.mixtral-8x7b-instruct-v0:1"

ADMIN_USERNAME="admin"
ADMIN_HASHED_PASSWORD="your_bcrypt_hashed_password"
```

**Do not commit real AWS credentials, passwords, API keys, or secrets to GitHub.**

Add the following to `.gitignore`:

```gitignore
.env
venv_admin/
venv_user/
__pycache__/
*.pyc
```

---

# 4. Initialize Database

Create the required PostgreSQL tables:

```python
from dotenv import load_dotenv

load_dotenv()

from Admin.app import app, db

with app.app_context():
    db.create_all()
    print("Database initialized.")
```

Run:

```bash
python run_create_db.py
```

---

# Running the Application

Both applications need to run simultaneously.

### User Application

Windows:

```powershell
./startuser.ps1
```

Linux / macOS:

```bash
./startuser.sh
```

User application:

```text
http://localhost:8086
```

### Admin Application

Windows:

```powershell
./startadmin.ps1
```

Linux / macOS:

```bash
./startadmin.sh
```

Admin application:

```text
http://localhost:5000
```

---

# Research

This project explores **Federated Retrieval-Augmented Generation for university knowledge systems**.

The architecture focuses on improving retrieval precision by routing queries to domain-specific knowledge bases instead of searching a single monolithic vector store.

---

# Future Improvements

* Improved RAG evaluation and benchmarking
* Advanced retrieval and reranking
* Conversation-aware retrieval
* Automated document ingestion
* Containerized deployment
* Improved monitoring and observability
* Scalable cloud deployment

---

# Author

**Rachit Jain**

B.Tech Computer Science & Engineering — Data Science

Interested in **AI/ML Engineering, Generative AI, RAG, Agentic AI, and Data Science**.
