# Simple re-export file
from models.vectorDB_config import qdrant, embeddings, vectorstore

# Re-exporter tout ce dont search.py a besoin
__all__ = ['qdrant', 'embeddings', 'vectorstore']