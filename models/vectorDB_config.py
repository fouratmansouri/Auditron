from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
import os

# Path configuration
collection_path = "C:\\Users\\friti\\Downloads\\projfinance\\qdrant_data\\collection\\Auditron_legal_chunks"
collection_name = "Auditron_legal_chunks"

# Vérifier si le chemin existe
if not os.path.exists(collection_path):
    print(f"ATTENTION: Le chemin {collection_path} n'existe pas. Création du dossier.")
    os.makedirs(collection_path, exist_ok=True)

# Initialize Qdrant client en utilisant le chemin parent
# Qdrant s'attend à ce que le chemin pointe vers le dossier parent des collections
qdrant_path = "C:\\Users\\friti\\Downloads\\projfinance\\qdrant_data"
qdrant = QdrantClient(path=qdrant_path)

# Initialize HuggingFace embeddings - conservé comme dans votre code
embeddings = HuggingFaceEmbeddings(model_name="dangvantuan/sentence-camembert-large")

try:
    # Vérifier si la collection existe
    collection_info = qdrant.get_collection(collection_name)
    print(f"Collection '{collection_name}' trouvée avec succès")
    
    # Create Qdrant vector store avec la collection existante
    vectorstore = QdrantVectorStore(
        client=qdrant,
        collection_name=collection_name,
        content_payload_key="text",
        embedding=embeddings
    )
    
except Exception as e:
    print(f"Erreur lors de l'accès à la collection: {str(e)}")
    print("Tentative de création ou récupération avec force_recreate=True...")
    
    # Create Qdrant vector store avec force_recreate=True pour gérer les erreurs
    vectorstore = QdrantVectorStore(
        client=qdrant,
        collection_name=collection_name,
        content_payload_key="text",
        embedding=embeddings,
        force_recreate=True  # Forcer la recréation si nécessaire
    )
    print("Collection créée ou recréée avec succès")