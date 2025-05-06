import sqlite3
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM as Ollama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableBranch, RunnablePassthrough
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory

# Initialize the LLMs
def init_llm():
    return Ollama(model="deepseek-r1:32b", temperature=0.1)

def init_routing_llm():
    return Ollama(model="mistral", temperature=0.1, top_p=0.9, top_k=40)

# Load and prepare documents
def load_documents():
    conn = sqlite3.connect("article52.db")
    cursor = conn.cursor()
    cursor.execute("SELECT title, content, pub_date FROM laws")
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

# Define tools
def retrieve_docs(question, vector_store):
    """Retrieve relevant documents from the database."""
    docs = vector_store.similarity_search(question, k=3)
    sources = "\n".join([f"{doc.metadata['title']} - {doc.metadata['source']}" for doc in docs])
    return sources if sources else "Aucune source trouvée."

def classify_question(routing_llm, question):
    """Classify whether retrieval is needed."""
    prompt = """Votre mission est de classifier la question suivante en fonction de sa complexité...
    Question: {question}
    Réponse:"""
    
    response = routing_llm.invoke(prompt.format(question=question))
    return "oui" in response.strip().lower()

# Initialize the system
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
    
    # Define tools
    retrieval_tool = Tool(
        name="Document Retrieval",
        func=lambda question: retrieve_docs(question, vector_store),
        description="Utilisez cet outil pour rechercher des informations fiscales spécifiques dans la base de connaissances."
    )

    classification_tool = Tool(
        name="Question Classification",
        func=lambda question: "Oui" if classify_question(routing_llm, question) else "Non",
        description="Utilisez cet outil pour déterminer si la question nécessite une recherche documentaire."
    )

    # Memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Initialize Agent
    agent = initialize_agent(
        tools=[retrieval_tool, classification_tool],
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )

    return agent

# Main interaction
def main():
    agent = initialize_system()
    print("Agent prêt. Posez votre question (tapez 'exit' pour quitter):")
    while True:
        question = input("\nQuestion : ")
        if question.lower() == 'exit':
            break
        try:
            print("Traitement en cours...")
            response = agent.invoke(question)
            print("\nRéponse :")
            print(response)
        except Exception as e:
            print(f"Erreur: {str(e)}")

if __name__ == "__main__":
    main()
