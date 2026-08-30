# RAG Document Q&A Assistant 🤖📄

Welcome to the **RAG Document Q&A Assistant**! This project is an intelligent, context-aware chatbot designed to let you "talk" to your documents. Using Retrieval-Augmented Generation (RAG), the assistant reads your PDFs, Text files, and Markdown files, understands their content, and provides accurate answers based strictly on the provided documents.

---

## 🌟 Key Features

* **Multi-Format Document Support**: Upload PDFs, TXT, and MD files.
* **Intelligent RAG Pipeline**: Uses advanced chunking and vector embeddings to find the most relevant information.
* **Fast Vector Search**: Powered by FAISS (Facebook AI Similarity Search) for blazing-fast document retrieval.
* **State-of-the-art LLM Integration**: Uses Anthropic Claude models for highly accurate, hallucination-free answer generation.
* **Beautiful Streamlit UI**: A clean, modern, and interactive chat interface.
* **Source Tracking**: Answers include traces back to the exact chunks of text retrieved from your documents.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.9+
* Anthropic API Key (for Claude LLM)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Aditya-err/RAG-Document-Q-A-Assistant.git
   cd RAG-Document-Q-A-Assistant
   ```

2. **Install dependencies:**
   Navigate to the app directory and install required Python packages:
   ```bash
   cd "new ui/rag_app"
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables:**
   Rename `.env.example` to `.env` and add your API keys:
   ```env
   ANTHROPIC_API_KEY=your_api_key_here
   ```

### Running the App

Start the Streamlit interface by running:
```bash
streamlit run "new ui/rag_app/app.py"
```
