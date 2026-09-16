# 🩺 Clinica AI

**Clinica AI** is an AI-powered medical report assistant that allows users to upload medical reports and ask questions about them through a simple conversational interface.

The application uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from uploaded reports and generate concise answers using **Groq**.

live demo : [https://clinica-ai-ygxg8k9kwt9ptncpee2uez.streamlit.app/]

## ✨ Features

- 📄 Upload medical PDF reports
- 💬 Ask natural-language questions
- 🔎 Retrieve relevant information from reports
- 🧠 Semantic search using embeddings
- 🤖 AI-powered answers using Groq
- 💭 Conversational follow-up questions
- 🩺 Medical report explanations
- 🔐 Secure API key management using `.env`
- ⚡ Streamlit-based interface

## 🏗️ Architecture

```text
User
  ↓
Upload Medical Report
  ↓
Extract & Split Text
  ↓
BGE Embeddings
  ↓
Pinecone
  ↓
User Question
  ↓
Retrieve Relevant Context
  ↓
Groq LLM
  ↓
Answer
```

## 🛠️ Tech Stack

- Python
- Streamlit
- Groq API
- LangChain
- Pinecone
- Hugging Face Embeddings
- BAAI/bge-large-en-v1.5
- PyPDF
- RAG

## 📁 Project Structure

```text
clinica-ai/
│
├── app.py
├── requirements.txt
├── .gitignore
├── .env
└── README.md
```

> **Note:** Never commit your `.env` file to GitHub.

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/clinica-ai.git
cd clinica-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

### 5. Run the application

```bash
streamlit run app.py
```

## 💬 Example Questions

```text
Is everything okay with my report?

What is my hemoglobin?

Is that normal?

Why is it low?

What about my PCV?

Could this explain why I'm feeling tired?

Should I get anything checked again?

Can you explain this report in simple words?
```

## 🔒 Medical Disclaimer

Clinica AI is intended for **educational and informational purposes only**. It does not replace professional medical advice, diagnosis, or treatment from a qualified healthcare professional.

---

**Built with Python, Streamlit, Groq, Pinecone & RAG.** 🩺
