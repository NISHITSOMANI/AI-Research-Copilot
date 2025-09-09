<<<<<<< HEAD
# AI-Research-Copilot
=======
# AI Research Copilot

A Retrieval-Augmented Generation (RAG) system for research papers. This application allows you to ingest, search, and ask questions about research papers using natural language.

## Features

- **Document Ingestion**: Process PDF and TXT documents
- **Semantic Search**: Find relevant document chunks using FAISS
- **Question Answering**: Get direct answers to your questions
- **RAG Pipeline**: Combined retrieval and generation for more accurate answers
- **REST API**: Easy integration with other applications

## Prerequisites

- Python 3.8+
- pip

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ai-research-copilot
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare your documents**:
   - Place your research papers (PDF or TXT) in the `data/` directory

2. **Ingest documents**:
   ```bash
   python ingest.py
   ```
   This will process all documents in the `data/` directory and create a FAISS index.

3. **Start the API server**:
   ```bash
   python app.py
   ```
   The server will start on `http://localhost:5000`

## API Endpoints

### 1. Check Status
- **GET** `/status`
  - Check if the service is running and get index information

### 2. Ingest Documents
- **GET/POST** `/ingest`
  - Upload new documents to be processed and indexed
  - Supports file upload via form-data

### 3. Search Documents
- **POST** `/search`
  - Find relevant document chunks
  - Request body: `{"query": "your search query", "top_k": 3}`

### 4. Ask a Question (Direct)
- **POST** `/ask`
  - Get a direct answer without retrieval
  - Request body: `{"query": "your question"}`

### 5. RAG (Retrieve and Generate)
- **POST** `/rag`
  - Get an answer using retrieved context
  - Request body: `{"query": "your question", "top_k": 3}`

## Example Usage with cURL

```bash
# Check status
curl http://localhost:5000/status

# Search for documents
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is FAISS?"}'

# Ask a question
curl -X POST http://localhost:5000/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main applications of neural networks?"}'

# Upload a document
curl -X POST -F 'file=@/path/to/your/document.pdf' http://localhost:5000/ingest
```

## Project Structure

```
ai-research-copilot/
├── app.py          # Flask API server
├── ingest.py       # Document ingestion and FAISS indexing
├── requirements.txt
├── data/           # Store research papers here (PDF/TXT)
└── faiss_index/    # FAISS index and metadata
```

## License

MIT License - Feel free to use this project for any purpose.
>>>>>>> 94b73b4 (AI-Research-Copilot Initialized)
