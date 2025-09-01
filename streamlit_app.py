import os
import streamlit as st
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ================================
# 1. Streamlit UI setup
# ================================
st.set_page_config(page_title="📄 RAG Chatbot", layout="wide")
st.title("📄 Document Q&A with Groq + LangChain")

uploaded_files = st.file_uploader(
    "Upload files (PDF, Word, Excel, TXT)", 
    type=["pdf", "doc", "docx", "xls", "xlsx", "txt"], 
    accept_multiple_files=True
)

# ================================
# 2. Process documents
# ================================
docs_path = "./docs"
os.makedirs(docs_path, exist_ok=True)

valid_extensions = ['.pdf', '.xls', '.xlsx', '.doc', '.docx', '.txt']
file_types = {
    '.pdf': PyPDFLoader,
    '.xls': UnstructuredExcelLoader,
    '.xlsx': UnstructuredExcelLoader,
    '.doc': UnstructuredWordDocumentLoader,
    '.docx': UnstructuredWordDocumentLoader,
    '.txt': TextLoader
}

all_docs = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(docs_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        ext = os.path.splitext(file_path)[1].lower()
        if ext in file_types:
            loader = file_types[ext](file_path)
            try:
                docs = loader.load()
                all_docs.extend(docs)
                st.success(f"✅ Processed {uploaded_file.name} ({len(docs)} sections)")
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {e}")
        else:
            st.warning(f"⚠️ Unsupported file type: {uploaded_file.name}")

if not all_docs and not os.path.exists("./chroma_db"):
    st.info("Please upload at least one document to get started.")
    st.stop()

# ================================
# 3. Text splitter
# ================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=500,
    separators=["\n\n", "\n", ". ", " ", ""]
)

splits = text_splitter.split_documents(all_docs) if all_docs else []
st.write(f"📚 Total new chunks: {len(splits)}")

# ================================
# 4. Embeddings + Chroma vectorstore
# ================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)



persist_dir = "./chroma_db"

if os.path.exists(persist_dir):
    st.write("📦 Loading existing Chroma DB...")
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    if splits:
        st.write("➕ Adding new documents to DB...")
        vectorstore.add_documents(splits)
        st.success("✅ New documents added to the vectorstore")
else:
    st.write("⚡ Creating new Chroma DB...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"}
    )

# ================================
# 5. LLM setup
# ================================
api_key = os.getenv("GROQ_API_KEY")  # ✅ read from env / secrets
if not api_key:
    st.error("❌ Missing GROQ_API_KEY. Please set it in Streamlit secrets or environment.")
    st.stop()

llm = ChatGroq(
    api_key=api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=4096,
    model_kwargs={"frequency_penalty": 0.2}
)

prompt = ChatPromptTemplate.from_template("""
You are a professional document analyst. Analyze and synthesize information from multiple documents to answer the question.
Guidelines:
1. Explicitly reference document sources when possible
2. Acknowledge conflicting information between documents
3. State clearly when information is not found
4. Maintain technical accuracy

Context:
<context>
{context}
</context>

Question: {input}

Provide a comprehensive, well-structured response with references:
""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 15, "lambda_mult": 0.6}
)
rag_chain = create_retrieval_chain(retriever, document_chain)

# ================================
# 6. Chat UI
# ================================
if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("💬 Ask a question about your documents:")
if st.button("Ask") and user_input:
    with st.spinner("Thinking..."):
        try:
            response = rag_chain.invoke({"input": user_input})
            answer = response["answer"]

            # Store conversation
            st.session_state.history.append((user_input, answer))
        except Exception as e:
            st.error(f"🚨 Error: {str(e)}")

# Show chat history
for q, a in st.session_state.history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**AI Assistant:** {a}")
    st.markdown("---")
