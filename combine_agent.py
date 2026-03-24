import streamlit as st
from pypdf import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import shutil
import json
import os

load_dotenv()

# ============================================
# 1. EMBEDDINGS — Sirf 1 baar load
# ============================================
@st.cache_resource(show_spinner="📦 Embeddings load ho rahi hain...")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ============================================
# 2. PDF PROCESSING
# ============================================
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# ============================================
# 3. VECTOR STORE
# ============================================
def get_vectorstore(text, embeddings):
    FAISS_PATH = "faiss_index"

    # Purana FAISS delete karo — naye PDF ke liye
    if os.path.exists(FAISS_PATH):
        shutil.rmtree(FAISS_PATH)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    vectorstore.save_local(FAISS_PATH)
    return vectorstore

# ============================================
# 4. TOOLS
# ============================================
@tool
def calculator(expression: str) -> str:
    """Maths calculate karta hai. Example: 100*278"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

def make_pdf_tool(vectorstore):
    @tool
    def search_pdf(query: str) -> str:
        """
        Uploaded PDF documents se information dhundta hai.
        Use this for questions about the uploaded documents.
        """
        if vectorstore is None:
            return "Koi PDF upload nahi hua. Pehle PDF upload karein."
        docs = vectorstore.similarity_search(query, k=5)
        if not docs:
            return "Document mein yeh information nahi mili."
        return "\n\n".join([doc.page_content for doc in docs])
    return search_pdf

# ============================================
# 5. AGENT
# ============================================
def load_agent(vectorstore=None):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )
    search = TavilySearch(
        max_results=3,
        tavily_api_key=os.getenv("TAVILY_API_KEY")
    )
    pdf_tool = make_pdf_tool(vectorstore)
    tools = [calculator, search, pdf_tool]
    return create_react_agent(model=llm, tools=tools)

# ============================================
# 6. STREAMLIT UI
# ============================================
def main():
    st.set_page_config(
        page_title="AI Research Assistant",
        page_icon="🤖",
        layout="wide"
    )
    st.title("🤖 AI Research Assistant")
    st.caption("PDF Documents + Web Search + Calculator")

    # ---- Session State ----
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "agent" not in st.session_state:
        st.session_state.agent = load_agent()  # PDF ke bina bhi kaam kare

    embeddings = load_embeddings()

    # ---- Sidebar ----
    with st.sidebar:
        st.header("📄 Upload Documents")
        pdf_docs = st.file_uploader(
            "PDF files upload karein",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("📥 Process Documents"):
            if pdf_docs:
                with st.spinner("Processing PDF..."):
                    raw_text = get_pdf_text(pdf_docs)
                    st.session_state.vectorstore = get_vectorstore(
                        raw_text, embeddings
                    )
                    # ✅ PDF ke saath naya agent banao
                    st.session_state.agent = load_agent(
                        st.session_state.vectorstore
                    )
                    st.success(f"✅ {len(pdf_docs)} PDF ready!")
            else:
                st.warning("Pehle PDF upload karein!")

        st.divider()

        # Status dikhao
        if st.session_state.vectorstore:
            st.success("📄 PDF loaded ✅")
        else:
            st.info("📄 No PDF — Web search only")

        st.divider()

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption("Powered by Groq + Tavily + HuggingFace")

    # ---- Chat History ----
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # ---- Input ----
    if query := st.chat_input("PDF ya internet se kuch bhi poochho..."):
        st.chat_message("human").write(query)
        st.session_state.messages.append({
            "role": "human",
            "content": query
        })

        with st.spinner("🔍 Soch raha hoon..."):
            try:
                all_messages = [
                    {
                        "role": "system",
                        "content": """You are a helpful AI research assistant.
You have 3 tools:
1. search_pdf     — for questions about uploaded documents
2. tavily_search  — for current news, weather, prices, internet info
3. calculator     — for math calculations

Rules:
- If user asks about uploaded document → use search_pdf
- If user asks about current events/news → use tavily_search
- If user asks math → use calculator
- Always give clear answers."""
                    }
                ] + [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]

                result = st.session_state.agent.invoke({
                    "messages": all_messages
                })
                response = result["messages"][-1].content

                # Sources nikalo
                sources = []
                for message in result["messages"]:
                    if hasattr(message, "name") and message.name == "tavily_search":
                        try:
                            data = json.loads(message.content)
                            for item in data.get("results", []):
                                if item.get("url"):
                                    sources.append(item["url"])
                        except:
                            pass

            except Exception as e:
                response = f"❌ Error: {str(e)}"
                sources = []

        # ---- Response ----
        st.chat_message("ai").write(response)

        if sources:
            with st.expander("📚 Sources"):
                for src in sources:
                    st.write(f"🔗 {src}")

        st.session_state.messages.append({
            "role": "ai",
            "content": response
        })

if __name__ == "__main__":
    main()
