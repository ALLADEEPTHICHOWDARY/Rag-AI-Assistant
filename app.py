import streamlit as st
from rag_core import build_vectorstore, get_response
from pypdf import PdfReader
import docx

from agent.core import Agent
from agent.llm import flan_t5_synthesize
from tools import document_search  # to call set_vectorstore()

st.set_page_config(page_title="Agentic RAG Assistant", layout="wide")
st.title("🤖 Agentic RAG Assistant (HuggingFace)")
st.caption(
    "Plans multi-step tasks, chooses between document search / web search / "
    "calculator, remembers past turns, and logs a full reasoning trace."
)

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
if "agent" not in st.session_state:
    st.session_state.agent = Agent(synthesize_fn=flan_t5_synthesize)

# Build Knowledge Base
if st.sidebar.button("Build Knowledge Base"):
    if document:
        st.session_state.vectorstore = build_vectorstore(document)
        document_search.set_vectorstore(st.session_state.vectorstore)
        st.sidebar.success("✅ Knowledge base created!")
    else:
        st.sidebar.warning("Please upload a document first.")

# -------------------------------
# 💬 MAIN AREA (CENTER)
# -------------------------------
st.markdown("### 💬 Ask Questions")
query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if query:
        with st.spinner("Thinking..."):
            result = st.session_state.agent.run(query)

        st.markdown("### 📌 Answer")
        st.write(result["answer"])

        with st.expander("🔍 Show agent reasoning trace"):
            for step in result["steps"]:
                event = step["event"]
                if event == "plan_created":
                    st.markdown(f"**Plan:** {step['plan']}")
                elif event == "long_term_memory_hit":
                    st.markdown("**Recalled from memory:**")
                    st.code("\n".join(step["results"]))
                elif event == "tool_call":
                    st.markdown(f"**Tool called:** `{step['tool']}` on _\"{step['subtask']}\"_")
                    st.code(step["result"])
                elif event == "final_answer":
                    st.markdown(f"**Finished in {step['elapsed_s']}s**")

        st.caption(f"Trace ID: `{result['trace_id']}` — saved to /traces")
    else:
        st.warning("Please enter a question.")