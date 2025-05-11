from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from typing import List
from pydantic import Field


class QdrantScoreRetriever(BaseRetriever):
    """
    Custom retriever for Qdrant using explicit embeddings and score thresholding.
    """
    vectorstore: QdrantVectorStore = Field(...)
    embeddings: Embeddings = Field(...)
    k: int = Field(default=5)
    score_threshold: float = Field(default=0.4)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_embedding = self.embeddings.embed_query(query)
        results = self.vectorstore.client.search(
            collection_name=self.vectorstore.collection_name,
            query_vector=query_embedding,
            limit=self.k,
            score_threshold=self.score_threshold,
            with_payload=True
        )

        documents = []
        for result in results:
            content = result.payload.get("text", "")
            metadata = {k: v for k, v in result.payload.items() if k != "text"}
            metadata["score"] = result.score
            documents.append(Document(
                page_content=content,
                metadata=metadata
            ))
        return documents


def create_bm25_retriever(qdrant_client: QdrantClient, collection_name: str, k: int = 5) -> BM25Retriever:
    """
    Creates a BM25Retriever from documents in a Qdrant collection.
    """
    response = qdrant_client.scroll(
        collection_name=collection_name,
        limit=10_000,
        with_payload=True
    )

    all_documents = [
        Document(
            page_content=point.payload.get("text", ""),
            metadata={k: v for k, v in point.payload.items() if k != "text"}
        )
        for point in response[0]
    ]

    return BM25Retriever.from_documents(
        all_documents,
        k=k,
        with_score=True
    )


def create_ensemble_retriever(
    qdrant_retriever: QdrantScoreRetriever,
    bm25_retriever: BM25Retriever,
    weights: List[float] = [0.4, 0.6]
) -> EnsembleRetriever:
    """
    Creates an ensemble retriever with weighted scores.
    """
    return EnsembleRetriever(
        retrievers=[bm25_retriever, qdrant_retriever],
        weights=weights
    )