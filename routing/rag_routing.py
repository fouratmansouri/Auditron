import sqlite3
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM as Ollama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableBranch, RunnablePassthrough, RunnableLambda

"""
This script demonstrates how to build a RAG-based system with routing using Langchain.
The LLMs used in this script are DeepSeek-R1 (8b) for the main LLM and Mistral for routing and general question answering.
"""

def init_llm():
    return Ollama(model="deepseek-r1:8b", temperature=0.1)

def init_routing_llm():
    return Ollama(model="mistral", temperature=0.1, top_p=0.9, top_k=40)

def classify_question(routing_llm, question):
    prompt = """Votre mission est de classifier la question suivante en fonction de sa complexité. Elle est posée par un expert fiscal tunisien. 
    Vous ne connaissez rien sur la fiscalité tunisienne sauf des généralités (définitions de TVA, retenue à la source, etc.). 
    Si la question demande des détails spécifiques sur la fiscalité tunisienne (montants exacts, mention exacte dans un document fiscal tel que les notes communes, contenu d'un document, etc.), répondez par 'oui'. 
    Si la question demande une définition générale ou un concept que vous connaissez, répondez par 'non'. 
    Répondez SEULEMENT par 'oui' ou 'non'. 

    Définitions générales :
    - **TVA (Taxe sur la Valeur Ajoutée)** : Impôt indirect sur la consommation, appliqué sur la vente de biens et services.
    - **Retenue à la source** : Mécanisme fiscal où un tiers prélève un montant sur un revenu et le reverse au fisc.

    Exemples :
    Question: Qu'est-ce que la TVA ?
    Réponse: non

    Question: Quel est le taux de TVA applicable aux services de conseil en Tunisie en 2024 ?
    Réponse: oui

    Question: Comment fonctionne la retenue à la source ?
    Réponse: non

    Question: Quelle est la référence exacte de la note commune qui traite de la retenue à la source sur les prestations de service ?
    Réponse: oui

    Question: Quelle est la note commune référence à la TVA ?
    Réponse: oui

    Question: De quoi parle la note commune numéro 16 ?
    Réponse: oui

    Question: Quel est le taux de retenue à la source sur les loyers ?
    Réponse: oui

    Voici la question:
    Question: {question}
    Réponse:"""

    response = routing_llm.invoke(prompt.format(question=question))
    return "oui" in response.strip().lower()

def load_documents():
    conn = sqlite3.connect("article52.db")
    cursor = conn.cursor()
    cursor.execute("SELECT title, content, pub_date, category, applied_year FROM laws")
    documents = []
    for row in cursor.fetchall():
        metadata = {"title": row[0], "source": row[2]}
        documents.append({"text": row[1], "metadata": metadata})
    conn.close()
    return documents

def split_documents(documents, chunk_size=2000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for doc in documents:
        texts = text_splitter.split_text(doc["text"])
        for text in texts:
            chunks.append({"text": text, "metadata": doc["metadata"]})
    return chunks

def init_vector_store(chunks=None):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    persist_dir = "./chroma_db"
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    if not chunks:
        raise ValueError("No chunks provided for vector store initialization")
    vector_store = Chroma.from_texts(
        texts=[chunk["text"] for chunk in chunks],
        embedding=embeddings,
        metadatas=[chunk["metadata"] for chunk in chunks],
        persist_directory=persist_dir
    )
    vector_store.persist()
    return vector_store

# Different prompts for different scenarios
retrieval_prompt = PromptTemplate.from_template(
    """<|im_start|>system
    Vous êtes un expert fiscal tunisien. Répondez en français en vous référant AU CONTEXTE DE LA BASE DES CONNAISSANCES:
    {context}
    
    Question: {question}
    <|im_end|>
    <|im_start|>assistant
    """
)

general_prompt = PromptTemplate.from_template(
    """<|im_start|>system
    Vous êtes un assistant tunisien qui répond à des généralités sur la fiscalité tunisienne (définitions TVA, retenue à la source, etc.) sans détails sur les taux en question. 
    Répondez en français de manière concise et claire, sans inclure de raisonnement ou de formatage supplémentaire et sans mentionner aucun pays sauf si c'est dans la question. 
    Assurez-vous de raffiner votre réponse à la question posée et veillez à éviter toute confusion entre les impôts directs et indirects.
    
    Question: {question}
    <|im_end|>
    <|im_start|>assistant
    """
)

def create_rag_chain(vector_store, llm, routing_llm):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    def route(inputs):
        question = inputs["question"]
        # Classify whether retrieval is needed
        needs_retrieval = classify_question(routing_llm, question)
        print("Needs retrieval: ", needs_retrieval)
        return {"question": question, "needs_retrieval": needs_retrieval}

    retrieval_chain = (
        {
            "question": RunnablePassthrough(),
            "context": lambda x: retriever.invoke(x["question"])
        }
        | retrieval_prompt
        | llm
    )

    general_chain = (
        {
            "question": RunnablePassthrough()
        }
        | general_prompt
        | routing_llm
    )  # Use Phi for general answers

    # Use RunnableBranch to conditionally execute the retrieval or general chain
    full_chain = RunnableBranch(
        (lambda x: x["needs_retrieval"], retrieval_chain),
        general_chain
    ).with_types(input_type=dict)

    return route | full_chain

def initialize_system():
    persist_dir = "./chroma_db"
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        documents = load_documents()
        chunks = split_documents(documents)
        vector_store = init_vector_store(chunks)
    else:
        vector_store = init_vector_store()
    
    llm = init_llm()
    routing_llm = init_routing_llm()
    return create_rag_chain(vector_store, llm, routing_llm), vector_store, llm, routing_llm

def main():
    rag_chain, vector_store, llm, routing_llm = initialize_system()
    print("Système prêt. Posez votre question (tapez 'exit' pour quitter):")
    while True:
        question = input("\nQuestion : ")
        if question.lower() == 'exit':
            break
        try:
            print("Traitement en cours...")
            response = rag_chain.invoke({"question": question})
            
            print("\nRéponse :")
            print(response)
            
            # Only show the sources if retrieval was done
            if "needs_retrieval" in response and response["needs_retrieval"]:
                print("\nSources utilisées:")
                docs = vector_store.similarity_search(question, k=3)
                for i, doc in enumerate(docs, 1):
                    print(f"{i}. {doc.metadata['title']} - {doc.metadata['source']}")
        except Exception as e:
            print(f"Erreur: {str(e)}")

if __name__ == "__main__":
    main()