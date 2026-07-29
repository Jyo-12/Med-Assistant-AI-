# 🩺 MedAssist AI

MedAssist AI is an AI-powered medical assistant that combines **Retrieval-Augmented Generation (RAG)**, **Google Gemini**, **FAISS**, and **Streamlit** to answer medical questions from trusted medical documents.

> **Disclaimer:** This project is for educational purposes only. It is not intended to diagnose, treat, cure, or prevent any disease.

---

# Features

- 🤖 Gemini 2.5 Flash LLM
- 📚 Retrieval-Augmented Generation (RAG)
- 🔍 Hybrid Search (FAISS + BM25)
- 📄 PDF Medical Document Search
- 🧠 Sentence Transformer Embeddings
- 🎤 Voice Input
- 🖼 Medical Image Analysis
- 💬 Chat Interface
- 📖 Source References
- ⚡ Streamlit Dashboard

---

# Project Structure

```
MedAssist_AI/
│
├── app.py
├── chat_engine.py
├── build_index.py
├── requirements.txt
├── README.md
├── .env
│
├── documents/
├── uploads/
├── images/
├── vector_store/
│
├── modules/
│   ├── chunker.py
│   ├── embeddings.py
│   ├── llm.py
│   ├── pdf_loader.py
│   ├── retriever.py
│   ├── speech.py
│   ├── utils.py
│   ├── vector_store.py
│   └── vision.py
│
└── dataset_builder/
```

---

# Installation

Clone the repository

```bash
git clone <repository-url>
```

Move into the project

```bash
cd MedAssist_AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Configure API Key

Create a `.env` file.

```env
GEMINI_API_KEY=YOUR_API_KEY
```

---

# Add Medical PDFs

Copy your medical PDFs into

```
documents/
```

Examples

- WHO Guidelines
- PubMed PDFs
- Medical textbooks

---

# Build the Vector Database

```bash
python build_index.py
```

This creates

```
vector_store/
    medical.index
    metadata.pkl
    texts.pkl
```

---

# Run the Application

```bash
streamlit run app.py
```

---

# Technologies Used

- Python
- Streamlit
- Google Gemini
- FAISS
- Sentence Transformers
- BM25
- PyMuPDF
- SpeechRecognition
- Pillow
- NumPy

---

# Workflow

```
Medical PDFs
      │
      ▼
PDF Loader
      │
      ▼
Chunking
      │
      ▼
Embeddings
      │
      ▼
FAISS Index
      │
      ▼
Hybrid Retrieval
      │
      ▼
Gemini LLM
      │
      ▼
Medical Answer
```

---

# Future Improvements

- OCR Support
- Authentication
- User Accounts
- Chat History Database
- Multi-language Support
- Medical Report Summarization
- Drug Interaction Checker
- Docker Deployment
- FastAPI Backend
- AWS Deployment

---

# Disclaimer

This software is intended for educational and research purposes only.

Always consult a qualified healthcare professional for diagnosis and treatment.

---

# Author

Developed as an AI-powered Medical Assistant using Google Gemini, FAISS, and Retrieval-Augmented Generation.