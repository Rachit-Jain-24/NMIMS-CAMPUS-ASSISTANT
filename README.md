# 🎓 NMIMS Campus Assistant - RAG-Powered Chatbot

A comprehensive AI-powered campus assistant system for SVKM'S NMIMS Deemed to be UNIVERSITY, Hyderabad Campus. This system provides intelligent responses to student queries about academic policies, course information, campus resources, and more.

## 🏗️ System Architecture

### Dual-Portal RAG System
- **Admin Portal**: Document management and knowledge base maintenance
- **User Portal**: Interactive chatbot interface for student queries

### Technology Stack
- **Frontend**: Streamlit (Python)
- **AI/ML**: Amazon Bedrock (Titan Embeddings + Claude/Titan/Mistral LLMs)
- **Storage**: Amazon S3
- **Vector Search**: FAISS
- **Document Processing**: LangChain, pypdf

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- AWS Account with Bedrock access
- Git

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/campus-assistant-capstone.git
   cd nmims-rag-chatbot
   ```

2. **Set up environment variables**
   ```bash
   # Copy template files
   cp env_template.txt Admin/.env
   cp env_template.txt User/.env
   
   # Edit both .env files with your AWS credentials
   ```

3. **Install dependencies**
   ```bash
   # Admin Portal
   cd Admin
   pip install -r requirements.txt
   
   # User Portal
   cd User
   pip install -r requirements.txt
   ```

4. **Run applications**
   ```bash
   # Admin Portal (Terminal 1)
   cd Admin
   streamlit run Admin.py --server.port 8085
   
   # User Portal (Terminal 2)
   cd User
   streamlit run user.py --server.port 8086
   ```

5. **Access applications**
   - Admin Portal: http://localhost:8085
   - User Portal: http://localhost:8086

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)
```bash
# Build and run both services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Individual Docker Builds
```bash
# Admin Portal
cd Admin
docker build -t nmims-admin .
docker run -p 8085:8085 --env-file .env nmims-admin

# User Portal
cd User
docker build -t nmims-user .
docker run -p 8086:8085 --env-file .env nmims-user
```

## ☁️ Streamlit Cloud Deployment

### Step 1: Prepare Repository
1. Push your code to GitHub
2. Ensure all files are committed

### Step 2: Deploy Admin Portal
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Configure:
   - Repository: `your-username/nmims-rag-chatbot`
   - Branch: `main`
   - Main file path: `Admin/Admin.py`
   - App URL: `nmims-admin-portal`

### Step 3: Deploy User Portal
1. Click "New app" again
2. Configure:
   - Repository: `your-username/nmims-rag-chatbot`
   - Branch: `main`
   - Main file path: `User/user.py`
   - App URL: `nmims-user-portal`

### Step 4: Configure Secrets
Add these secrets in Streamlit Cloud dashboard:
```toml
[secrets]
AWS_ACCESS_KEY_ID = "your_aws_access_key"
AWS_SECRET_ACCESS_KEY = "your_aws_secret_key"
AWS_DEFAULT_REGION = "ap-south-1"
BUCKET_NAME = "your-s3-bucket-name"
```

## 📋 Features

### Admin Portal
- ✅ PDF document upload and processing
- ✅ Automated text extraction and chunking
- ✅ Vector embedding generation
- ✅ FAISS vector store creation
- ✅ S3 upload and management
- ✅ Professional NMIMS-branded UI
- ✅ Real-time processing status

### User Portal
- ✅ Interactive chat interface
- ✅ Intelligent query processing
- ✅ Context-aware responses
- ✅ Source citations and references
- ✅ Chat history management
- ✅ Professional NMIMS branding
- ✅ Mobile-responsive design

## 🔧 Configuration

### Environment Variables
```bash
# Required
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=ap-south-1
BUCKET_NAME=your-s3-bucket-name

# Optional
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
BEDROCK_LLM_MODEL_ID=anthropic.claude-v2:1
```

### AWS Services Used
- **Amazon Bedrock**
  - `amazon.titan-embed-text-v2:0` (Text Embeddings)
  - `anthropic.claude-v2:1` (Primary LLM)
  - `amazon.titan-text-lite-v1` (Secondary LLM)
  - `mistral.mistral-7b-instruct-v0:2` (Secondary LLM)
- **Amazon S3** (Vector store and document storage)
- **AWS IAM** (Access management)

## 📊 Data Flow

### Knowledge Base Creation (Admin)
1. PDF Upload → Text Extraction → Chunking → Embeddings → FAISS → S3

### Query Processing (User)
1. Question → Embeddings → Similarity Search → Context → LLM → Response

## 🛡️ Security

- Environment-based credential management
- IAM role-based AWS access
- S3 bucket security policies
- Educational data compliance
- Non-root Docker user execution

## 📈 Performance

- Streamlit resource caching
- Efficient FAISS vector search
- Batch document processing
- Real-time query responses
- Health check monitoring

## 🔍 Troubleshooting

### Common Issues
1. **AWS Credentials**: Verify environment variables
2. **Import Errors**: Check requirements.txt versions
3. **Memory Issues**: Optimize caching strategies
4. **Port Conflicts**: Change ports in configuration

### Debug Commands
```bash
# Check container logs
docker logs nmims-admin
docker logs nmims-user

# Test AWS connectivity
aws s3 ls s3://your-bucket-name

# Verify Bedrock access
aws bedrock list-foundation-models --region ap-south-1
```

## 📚 Documentation

- [AWS Architecture Guide](aws_architecture_prompt.md)
- [Docker Deployment Guide](DOCKER_DEPLOYMENT_GUIDE.md)
- [Streamlit Cloud Deployment](STREAMLIT_CLOUD_DEPLOYMENT.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For support and questions:
- **Email**: admin@nmims.edu
- **Website**: hyderabad.nmims.edu
- **Address**: Survey No. 102, Shamirpet, Hyderabad

## 🎯 Roadmap

- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] Mobile app integration
- [ ] Voice input/output
- [ ] Integration with university systems

---

**Built with ❤️ for NMIMS Hyderabad Campus**
