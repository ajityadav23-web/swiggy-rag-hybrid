from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from rank_bm25 import BM25Okapi
import numpy as np


# -----------------------------
# 1. Load Embedding Model
# -----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# 2. Load FAISS Vector Store
# -----------------------------
vectorstore = FAISS.load_local(
    "vectorstore",
    embedding_model,
    allow_dangerous_deserialization=True
)

# Extract stored documents
docs = list(vectorstore.docstore._dict.values())

# -----------------------------
# 3. Prepare BM25 (Keyword Search)
# -----------------------------
tokenized_corpus = [doc.page_content.split() for doc in docs]
bm25 = BM25Okapi(tokenized_corpus)


# -----------------------------
# 4. Hybrid Search Function
# -----------------------------
def hybrid_search(query, k=4):

    # Vector search
    vector_results = vectorstore.similarity_search(query, k=k)

    # BM25 search
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(bm25_scores)[-k:]
    bm25_results = [docs[i] for i in top_indices]

    # Combine & remove duplicates
    combined = list(
        {doc.page_content: doc for doc in vector_results + bm25_results}.values()
    )

    return combined[:k]


# -----------------------------
# 5. Generate Answer using Ollama
# -----------------------------
def generate_answer(query):

    context_docs = hybrid_search(query)

    context = "\n\n".join([doc.page_content for doc in context_docs])

    # Local LLM (Mistral via Ollama)
    llm = ChatOllama(
        model="mistral",
        temperature=0
    )

    prompt = f"""
You must answer ONLY from the context below.
If the answer is not found, respond:
"The information is not available in the document."

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content, context_docs