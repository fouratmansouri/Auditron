from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# Initialize Qdrant client
qdrant = QdrantClient(path="../qdrant_data")

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="dangvantuan/sentence-camembert-large")

# Create Qdrant vector store
vectorstore = QdrantVectorStore(
    client=qdrant,
    collection_name="Auditron_legal_chunks",
    content_payload_key="text",
    embedding=embeddings
)