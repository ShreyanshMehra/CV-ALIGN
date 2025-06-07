# CV-ALIGN: AI-Powered Resume-Job Description Alignment Tool

CV-ALIGN is an advanced tool that uses state-of-the-art Natural Language Processing (NLP) techniques to analyze and score the alignment between resumes/CVs and job descriptions. It provides detailed insights and matching scores to help job seekers optimize their applications and helps recruiters efficiently screen candidates.

## 🚀 Features

- **Semantic Analysis**: Uses advanced NLP models to understand context beyond simple keyword matching
- **Comprehensive Scoring**: Provides detailed scoring metrics including:
  - Overall alignment score (weighted combination of all components)
  - Section-wise scores (Projects, Experience, Technical Skills)
  - CGPA evaluation
  - Keyword matching scores
- **Intelligent Processing**:
  - Smart text chunking and preprocessing
  - Section-specific analysis
  - PDF and DOCX file support
- **User-friendly Interface**: Modern, responsive web interface with drag & drop functionality

## 🛠️ Tech Stack

### Backend
- **Python 3.x**
- **FastAPI**: High-performance web framework for building APIs
- **Sentence-Transformers**: Using "all-MiniLM-L6-v2" model for semantic embeddings
- **FAISS**: Facebook AI Similarity Search for efficient similarity computations
- **KeyBERT**: For keyword extraction and analysis
- **pdfplumber & python-docx**: For document parsing
- **NumPy**: For numerical computations

### Frontend
- **HTML5**: Modern semantic markup
- **CSS3**: Responsive design with animations
- **JavaScript**: Dynamic UI and API integration
- **Drag & Drop API**: Native file upload functionality

## 🧠 Technical Approach

### RAG (Retrieval-Augmented Generation) Engine
The core of CV-ALIGN uses a sophisticated RAG engine that:
1. **Text Preprocessing**: Cleanses and normalizes input text
2. **Smart Chunking**: Splits text into meaningful segments with configurable overlap
3. **Semantic Analysis**: Generates dense vector embeddings using Sentence Transformers
4. **Similarity Computation**: Uses FAISS for efficient similarity search
5. **Comprehensive Scoring**: Implements a weighted scoring system:
   - CV Overall (40%)
   - Projects (20%)
   - Experience (20%)
   - CGPA (10%)
   - Technical Skills (10%)



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
   python Backend/server.py
   ```
4. Open your browser and navigate to:
   ```
   http://127.0.0.1:8001
   ```

## 💡 Key Features in Detail

### Document Processing
- Supports PDF and DOCX formats
- Intelligent section extraction (Education, Experience, Projects, etc.)
- Maintains document structure and formatting
- Handles multi-page documents

### Scoring System
- **Semantic Scoring**: Uses advanced NLP for context understanding
- **Keyword Analysis**: Extracts and weights important terms from job descriptions
- **Section Analysis**: Individual scoring for different CV sections
- **CGPA Evaluation**: Intelligent CGPA extraction and normalization
- **Relative Ranking**: Compares candidates within a batch

### User Interface
- Drag & drop file uploads
- Real-time processing feedback
- Clear result visualization
- Error handling and status updates

## 🔒 Best Practices

- Implements singleton pattern for resource management
- Uses caching for performance optimization
- Follows modular architecture
- Includes comprehensive error handling
- Provides detailed logging
