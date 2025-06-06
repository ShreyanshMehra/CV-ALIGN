# CV-ALIGN

A web application for aligning and comparing CVs (resumes) using a user-friendly frontend and a backend powered by Python and Firebase.

## ✨ Features

- **User-friendly interface** for uploading and viewing CVs.
- **Firebase integration** for secure data storage and authentication.
- **Python backend** for processing and aligning CV content.
- **RAG (Retrieval-Augmented Generation) engine** for advanced content matching.
- **Responsive design** with CSS and JavaScript.

## 🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

- **Python 3.x**
- **Node.js** (for frontend dependencies, if needed)
- **Firebase account** (for authentication and storage)

### Installation

1. **Clone the repository**
    git clone https://github.com/VaibhavNavneet/CV-ALIGN.git
    cd CV-ALIGN

2. **Install Python dependencies**
    pip install -r requirements.txt


3. **Set up Firebase**
- Add your Firebase config to `firebase-config.js`.
- Enable authentication and storage in your Firebase console.

4. **Start the backend server**
    python server.py

5. **Open the frontend**
- Open `index.html` in your browser.

## 📂 Project Structure

    CV-ALIGN/
    ├── cv-align-frontend/ # Frontend files (if using a separate folder)
    ├── .env # Environment variables
    ├── firebase-config.js # Firebase configuration
    ├── index.html # Main frontend file
    ├── main.py # Main backend logic
    ├── rag_engine.py # RAG engine for CV alignment
    ├── requirements.txt # Python dependencies
    ├── server.py # Backend server
    ├── script.js # Frontend JavaScript
    └── styles.css # Frontend styling


## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository.
2. **Create a new branch** for your feature or bugfix.
3. **Commit** your changes.
4. **Push** to the branch.
5. **Open a pull request**.


