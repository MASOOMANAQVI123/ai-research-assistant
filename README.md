# ai-research-assistant
AI Research Assistant using LangGraph, RAG (FAISS), and Web Search.
# 🤖 AI Research Assistant (LangGraph + RAG)

An advanced, agentic AI assistant that combines **Retrieval-Augmented Generation (RAG)** with real-time **Web Search** and autonomous tool-use. Built for high-speed research and document intelligence.

---

## 🚀 Features

- **Document Intelligence (RAG):** Upload multiple PDFs and chat with them using **FAISS** vector search.
- **Agentic Workflow:** Powered by **LangGraph**, the agent autonomously decides whether to search your documents, use the internet, or perform calculations.
- **Web Research:** Integrated with **Tavily Search API** for up-to-the-minute news and global data.
- **Extreme Speed:** Uses **Groq (Llama 3.3 70B)** to provide near-instant reasoning and responses.
- **Memory-Aware:** Maintains session-based chat history for contextual conversations.

---

## 🛠️ Tech Stack

- **LLM Orchestration:** [LangChain](https://www.langchain.com/) & [LangGraph](https://www.langchain.com/langgraph)
- **Vector Database:** [FAISS](https://github.com/facebookresearch/faiss)
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
- **Inference Engine:** [Groq Cloud](https://groq.com/) (Llama-3.3-70B)
- **Search API:** [Tavily AI](https://tavily.com/)
- **UI Framework:** [Streamlit](https://streamlit.io/)

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME
   pip install -r requirements.txt
   Environment Variables:
Create a .env file in the root directory and add your API keys:

Code snippet
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key

Run the App:
Bash
streamlit run combine_agent.py


📄 License
Distributed under the MIT License. See LICENSE for more information.

Developed by [MASOOMA_naqvi]
