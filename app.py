import streamlit as st
from rag_core import build_vectorstore, get_response
from pypdf import PdfReader
import docx

st.set_page_config(page_title="RAG AI Assistant", layout="wide")

st.title("🤖 RAG AI Assistant (HuggingFace)")

# -------------------------------
# 📂 SIDEBAR (LEFT SIDE)
# -------------------------------
st.sidebar.header("📂 Upload & Process")

uploaded_file = st.sidebar.file_uploader(
    "Upload your document",
    type=["txt", "pdf", "docx"]
)

def extract_text(file):
    if file.type == "text/plain":
        return str(file.read(), "utf-8")

    elif file.type == "application/pdf":
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])

    return ""

document = ""

if uploaded_file:
    document = extract_text(uploaded_file)
    st.sidebar.success(f"✅ Loaded: {uploaded_file.name}")

# Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Build Knowledge Base
if st.sidebar.button("Build Knowledge Base"):
    if document:
        st.session_state.vectorstore = build_vectorstore(document)
        st.sidebar.success("✅ Knowledge base created!")
    else:
        st.sidebar.warning("Please upload a document first.")

# -------------------------------
# 💬 MAIN AREA (CENTER)
# -------------------------------
st.markdown("### 💬 Ask Questions")

query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if st.session_state.vectorstore and query:
        with st.spinner("Thinking..."):
            response = get_response(query, st.session_state.vectorstore)

        st.markdown("### 📌 Answer")
        st.write(response)
    else:
        st.warning("Please upload document and build knowledge base first.")
