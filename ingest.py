from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load PDF
loader = PyPDFLoader("data/Annual-Report-FY-2023-24.pdf")
docs = loader.load()

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Semantic chunking
semantic_splitter = SemanticChunker(embedding_model)
chunks = semantic_splitter.split_documents(docs)

print("Total semantic chunks:", len(chunks))

# Create vector store
vectorstore = FAISS.from_documents(chunks, embedding_model)
vectorstore.save_local("vectorstore")

print("Vectorstore saved successfully.")