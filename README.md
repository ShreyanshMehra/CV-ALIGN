# CV-ALIGN: AI-Powered Resume-Job Description Alignment Tool

CV-ALIGN is an advanced tool that uses state-of-the-art Natural Language Processing (NLP) techniques to analyze and score the alignment between resumes/CVs and job descriptions. It provides detailed insights and matching scores to help job seekers optimize their applications and helps recruiters efficiently screen candidates.

## 🚀 Features

- **Semantic Analysis**: Uses advanced NLP models to understand context beyond simple keyword matching
- **Comprehensive Scoring**: Provides detailed scoring metrics including:
  - Overall alignment score
  - Maximum similarity score
  - Average similarity score
  - Coverage score
- **Keyword Analysis**: Extracts and weights important keywords from job descriptions
- **Real-time Processing**: Fast and efficient processing of documents
- **User-friendly Interface**: Modern, responsive web interface for easy document upload and analysis

## 🛠️ Tech Stack

### Backend
- **Python 3.x**
- **FastAPI**: High-performance web framework for building APIs
- **Sentence-Transformers**: For semantic text embeddings using "all-MiniLM-L6-v2" model
- **FAISS**: Facebook AI Similarity Search for efficient similarity computations
- **KeyBERT**: For keyword extraction and analysis
- **NumPy**: For numerical computations

### Frontend
- **HTML5**: Structure and content
- **CSS3**: Modern styling and responsive design
- **JavaScript**: Client-side logic and API integration
- **Drag & Drop API**: Native file upload functionality

## 🧠 Technical Approach

### RAG (Retrieval-Augmented Generation) Engine
The core of CV-ALIGN is built on a custom RAG engine that:
1. **Text Preprocessing**: Cleanses and normalizes input text
2. **Chunking**: Intelligently splits text into meaningful segments
3. **Embedding Generation**: Creates dense vector representations using Sentence-Transformers
4. **Similarity Computation**: Uses FAISS for efficient similarity search
5. **Score Calculation**: Implements a weighted scoring system combining:
   - Semantic similarity
   - Keyword matching
   - Coverage analysis

### Scoring Algorithm
The final score is calculated using a weighted combination of:
- 40% Maximum similarity
- 40% Average similarity
- 20% Coverage score

## 📁 Project Structure

```
CV-ALIGN/
├── Backend/
│   ├── rag_engine.py    # Core RAG implementation
│   ├── main.py          # FastAPI application
│   └── server.py        # Server configuration
├── Frontend/
│   ├── index.html       # Main HTML structure
│   ├── styles.css       # CSS styling
│   └── script.js        # Frontend logic
└── requirements.txt     # Python dependencies
```

## 🚀 Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the server:
   ```bash
   python server.py
   ```
4. Open your browser and navigate to:
   ```
   http://127.0.0.1:8001
   ```

## 📋 Files to Upload to GitHub

Here are the essential files that should be uploaded to GitHub:

1. Backend Files:
   - `rag_engine.py`
   - `main.py`
   - `server.py`
   - `requirements.txt`

2. Frontend Files:
   - `index.html`
   - `styles.css`
   - `script.js`

3. Configuration and Documentation:
   - `.gitignore`
   - `README.md`

Make sure to exclude:
- `__pycache__` directories
- Environment files
- IDE configuration files
