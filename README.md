# RAG Document Q&A Assistant 🤖📄

Welcome to the **RAG Document Q&A Assistant**! This project is an intelligent, context-aware chatbot designed to let you "talk" to your documents. 

Using **Retrieval-Augmented Generation (RAG)**, the assistant reads your PDFs, Text files, and Markdown files, understands their content, and provides accurate answers based *strictly* on the provided documents.

---

## 🌟 Key Features

* **Multi-Format Document Support**: Upload PDFs, TXT, and MD files effortlessly.
* **Intelligent RAG Pipeline**: Uses advanced chunking (Semantic/Recursive) and vector embeddings (`SentenceTransformers`) to find the most relevant information.
* **Fast Vector Search**: Powered by `FAISS` (Facebook AI Similarity Search) for blazing-fast document retrieval.
* **Flexible LLM Integration**: 
  * ☁️ **Cloud APIs**: Support for OpenRouter (e.g., Llama 3, Claude, GPT-4)
  * 🖥️ **Local Models**: Support for Ollama to run models completely offline and privately.
* **Separation of Concerns**: A high-performance **FastAPI Backend** paired with a beautiful **Streamlit UI**.
* **Performance Tracing**: Real-time latency metrics and tracing for every stage of the pipeline (Embedding, Retrieval, Context Building, LLM Generation).
* **Source Tracking**: Answers include traces back to the exact chunks of text retrieved from your documents.

---

## 🏗️ Architecture overview

The application is split into two main components:
1. **Backend (`backend.main.py`)**: A FastAPI server that handles document ingestion, vector database management, and RAG logic.
2. **Frontend (`app.py`)**: A Streamlit application that provides a clean, modern, ChatGPT-like interface.

---

## 🚀 Getting Started

### Prerequisites

* **Python 3.9+**
* (Optional) **Ollama** installed if you want to run models locally.
* (Optional) **OpenRouter API Key** if you want to use cloud models.

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Aditya-err/RAG-Document-Q-A-Assistant.git
cd "RAG-Document-Q-A-Assistant/new ui/rag_app"

# Install all required Python packages
pip install -r requirements.txt
```

*(Optional)* Create a `.env` file in the `rag_app` directory to store your keys:
```env
OPENROUTER_API_KEY=your_api_key_here
```

### 2. Start the Backend Server

The backend handles all the heavy lifting (Embeddings, FAISS, LLM orchestration). You must start it first.

Open a terminal in the `new ui/rag_app` directory:
```bash
python -m uvicorn backend.main:app --port 8000
```
*Wait for the server to output "Pipeline ready" before starting the frontend.*

### 3. Start the Frontend UI

Open a second terminal in the same `new ui/rag_app` directory:
```bash
streamlit run app.py
```
This will automatically open your web browser to `http://localhost:8501`.

---

## ⚙️ Configuration (UI)

Once the application is running, you can configure it directly from the UI:

1. Click on **Settings** in the left sidebar.
2. Check the box to use **OpenRouter API** and enter your API Key.
3. Choose your preferred model (e.g., `meta-llama/llama-3.1-8b-instruct`).
4. If left unchecked, the system will fall back to using your local **Ollama** setup.

## 🧠 How it Works (The RAG Pipeline)

When you ask a question, the backend performs the following steps:
1. **Query Embedding**: Converts your text question into a vector using a SentenceTransformer model.
2. **Retrieval**: Searches the FAISS vector database for the most mathematically similar text chunks from your uploaded documents.
3. **Context Construction**: Combines those relevant chunks into a single prompt for the LLM.
4. **LLM Generation**: Sends the context and your question to the LLM to generate a natural, grounded answer.

You can view the exact latency of each of these steps beneath every generated answer in the chat UI!
