from retrievers import QdrantScoreRetriever, create_bm25_retriever, create_ensemble_retriever
from vectorDB_config import qdrant, embeddings, vectorstore
from typing import List
from langchain_core.documents import Document


# Initialize QdrantScoreRetriever
qdrant_retriever = QdrantScoreRetriever(
    vectorstore=vectorstore,
    embeddings=embeddings,
    k=5,
    score_threshold=0.4
)

# Initialize BM25Retriever
bm25_retriever = create_bm25_retriever(
    qdrant_client=qdrant,
    collection_name="Auditron_legal_chunks",
    k=5
)

# Create EnsembleRetriever
ensemble_retriever = create_ensemble_retriever(
    qdrant_retriever=qdrant_retriever,
    bm25_retriever=bm25_retriever,
    weights=[0.4, 0.6]
)


def retrieve_with_threshold(query: str, threshold: float = 0.4) -> List[Document]:
    """
    Retrieve documents with combined score above a specified threshold.
    """
    docs = ensemble_retriever.invoke(query)
    filtered_docs = [
        doc for doc in docs
        if doc.metadata.get("score", 0) >= threshold
    ]

    for i, doc in enumerate(filtered_docs, 1):
        print(f"Document {i} [Score: {doc.metadata['score']:.3f}]")
        print(doc.page_content + "\n")

    return filtered_docs


if __name__ == "__main__":
    # Example query
    query = (
        "Quel est le nouveau taux unifié de retenue à la source applicable aux loyers, "
        "rémunérations non commerciales, honoraires et commissions en Tunisie depuis l'adoption "
        "de la loi N° 2020-46 du 23 décembre 2020?"
    )
    results = retrieve_with_threshold(query)