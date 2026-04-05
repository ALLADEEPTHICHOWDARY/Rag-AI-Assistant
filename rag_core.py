from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# ✅ IMPORTANT: Works with transformers==4.41.2
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

def build_vectorstore(text):
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    docs = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


def get_response(query, vectorstore):
    # ✅ Reduce retrieved docs
    docs = vectorstore.similarity_search(query, k=2)

    # ✅ Join context
    context = "\n\n".join([doc.page_content for doc in docs])

    # ✅ HARD LIMIT (prevents token overflow)
    MAX_CONTEXT_CHARS = 1500
    context = context[:MAX_CONTEXT_CHARS]

    prompt = f"""
You are a helpful AI assistant.

Answer the question based ONLY on the provided context.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{query}

Answer:
"""

    result = generator(prompt, max_length=200, do_sample=False)
    return result[0]['generated_text']
