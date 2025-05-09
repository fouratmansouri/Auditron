import operator
from typing_extensions import TypedDict
from typing import List, Optional, Dict, Tuple, Any
import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from search import retrieve_with_threshold
from langgraph.graph import StateGraph, END
from IPython.display import Image, display
import logging
from langchain_core.prompts import PromptTemplate
import os
import getpass
from langchain_community.tools.tavily_search import TavilySearchResults

# Initialize the LLM
local_llm = "mistral:7b-instruct"
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")
llm = ChatOllama(model=local_llm, temperature=0)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Initialize the web search tool
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("TAVILY_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# Define allowed domains for Tunisian governmental financial institutions
tunisian_gov_domains = [
    "finances.gov.tn",        # Ministry of Finance
    "douane.gov.tn",          # Tunisian Customs
    "impots.finances.gov.tn", # Tax authority
    "cga.gov.tn",             # General Committee of Insurance
    "cmf.tn",                 # Financial Market Council
    "portail.finances.gov.tn", # Finance Ministry Portal
    "swiver.io"
]

# Create the search tool with domain filtering
web_search_tool = TavilySearchResults(
    k=3,
    include_domains=tunisian_gov_domains
)

import operator
from typing_extensions import TypedDict
from typing import List, Annotated, Optional, Dict

class ConversationState(TypedDict):
    """
    Graph state for the Withholding Tax RAG Pipeline containing information
    propagated through each node in the workflow.
    """
    # User input
    original_query: str  # Original user question about withholding taxes
    
    # Query re-writing step
    rewritten_query: str  # Processed query optimized for retrieval
    
    # Retrieval step
    retrieved_documents: List[str]  # Documents from hybrid search (BM25 + semantic)
    relevancy_scores: Optional[List[float]]  # Relevancy scores for retrieved documents
    
    # LLM Document Filtering step - Binary relevance (relevant/irrelevant)
    filtered_documents: List[str]  # Only documents classified as relevant by LLM
    document_relevance: Dict[str, bool]  # Mapping document to relevance (True=relevant, False=irrelevant)
    filter_reasoning: Optional[Dict[str, str]]  # LLM reasoning for each document's relevance decision
    
    # Response generation step
    generated_response: str  # LLM generated response from filtered documents
    
    # Response validation step
    validation_result: str  # "Yes" or "No" - whether response answers the query
    validation_reason: Optional[str]  # Explanation for the validation decision
    
    # Fallback & retry mechanism
    retry_count: Annotated[int, operator.add]  # Track number of retry attempts
    max_retries: int  # Maximum number of retry attempts (default: 3)
    web_search_results: Optional[List[str]]  # Results from web search fallback
    
    # Final response
    final_response: str  # Final validated response to be returned to the user

### Nodes
def query_rewriting(state):
    """
    Process and rewrite the initial user query for optimal retrieval.

    Args:
        state (dict): The current graph state with original_query

    Returns:
        state (dict): Updated state with rewritten_query
    """
    print("---QUERY REWRITING---")
    original_query = state["original_query"]
    
    # Rewrite query for optimal retrieval
    rewritten_query = llm.invoke([
        SystemMessage(content="Tu es un expert en reformulation de prompts. "
        "Ton travail est de prendre un prompt complexe, de le simplifier, de le rendre clair, direct et facile à comprendre. "
        "Le résultat doit être en français parfait. Ne change pas le sens du prompt, seulement rends-le plus simple et clair."
        "répond juste par la question."),
        HumanMessage(content=original_query)
    ])
    
    return {"rewritten_query": rewritten_query.content}

def filter_documents_by_relevance(query: str, documents: List[Any]) -> Tuple[List[Any], Dict[str, bool], Dict[str, str]]:
    """
    Filter documents based on their relevance to the query using a LLM in JSON mode.
    
    Args:
        query (str): The query to evaluate document relevance against
        documents (List[Any]): List of document objects to evaluate
        llm_json_mode: The language model that supports JSON output mode
        
    Returns:
        Tuple containing:
        - filtered_documents (List[Any]): Only the relevant documents
        - document_relevance (Dict[str, bool]): Mapping of document IDs to relevance decision
        - filter_reasoning (Dict[str, str]): Reasoning for each relevance decision
    """
    filtered_documents = []
    document_relevance = {}
    filter_reasoning = {}
    
    # Doc grader instructions
    doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

    # Grader prompt template
    doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 

This carefully and objectively assess whether the document contains at least some information that is relevant to the question.

Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""
    
    for doc in documents:
        # Get document content and ID
        doc_content = doc.page_content
        doc_id = doc.metadata.get("id", str(hash(doc_content)))
        
        # Format the grader prompt with the document content and query
        doc_grader_prompt_formatted = doc_grader_prompt.format(
            document=doc_content, 
            question=query
        )
        
        # Get LLM response using the system and human message approach
        response = llm_json_mode.invoke(
            [SystemMessage(content=doc_grader_instructions)] + 
            [HumanMessage(content=doc_grader_prompt_formatted)]
        )
        
        # Extract the text content and store the raw response
        response_text = response.content if hasattr(response, 'content') else str(response)
        filter_reasoning[doc_id] = response_text
        
        try:
            # Parse the JSON response to determine relevance
            parsed_response = json.loads(response_text)
            is_relevant = parsed_response.get("binary_score", "").lower() == "yes"
            
            # Store relevance result
            document_relevance[doc_id] = is_relevant
            
            # Add to filtered documents if relevant
            if is_relevant:
                filtered_documents.append(doc)
                
        except json.JSONDecodeError:
            # Handle case where response is not valid JSON
            document_relevance[doc_id] = False
            filter_reasoning[doc_id] += " (Error: Response not in valid JSON format)"
    
    return filtered_documents, document_relevance, filter_reasoning

def retrieval_with_filtering(state):
    """
    Perform hybrid search (BM25 + semantic) using vector DB with relevancy threshold,
    then filter documents using Bloom LLM for binary relevance assessment.

    Args:
        state (dict): The current graph state with rewritten_query

    Returns:
        state (dict): Updated state with retrieved documents, relevancy scores, and filtering results
    """
    print("---RETRIEVAL WITH LLM FILTERING---")
    print(f"Processing query: {state['rewritten_query']}")
    query = state["rewritten_query"]
    
    # Hybrid search using BM25 + semantic
    documents = retrieve_with_threshold(query)
    print(f"Retrieved {len(documents)} documents")
    
    # Calculate relevancy scores
    relevancy_scores = [doc.metadata.get("score", 0.0) for doc in documents]
    
    # Filter documents by relevance using Bloom LLM
    filtered_docs, doc_relevance, reasoning = filter_documents_by_relevance(query, documents)
    print(f"Filtered to {len(filtered_docs)} relevant documents")
    
    # Create doc_id to document mapping for accessing original documents
    doc_id_mapping = {doc.metadata.get("id", str(hash(doc.page_content))): doc for doc in documents}
    
    # Format document relevance map to use actual documents as keys if needed
    document_relevance = {str(doc_id): relevance for doc_id, relevance in doc_relevance.items()}
    
    # Add metadata about filtering
    processing_metadata = {
        "processed_by": "fouratmansouri",
        "processing_timestamp": "2025-05-09 01:05:46",
        "total_documents": len(documents),
        "relevant_documents": len(filtered_docs)
    }
    
    return {
        "retrieved_documents": documents,
        "relevancy_scores": relevancy_scores,
        "filtered_documents": filtered_docs,
        "document_relevance": document_relevance,
        "filter_reasoning": reasoning,
        "processing_metadata": processing_metadata
    }

def response_generation(state):
    """
    LLM processes filtered documents to generate a response.
    No longer uses all retrieved documents as fallback.
    Uses web search as fallback when high-quality documents aren't available.

    Args:
        state (dict): The current graph state with rewritten_query, retrieved_documents,
                     and filtered_documents from LLM relevance filtering

    Returns:
        state (dict): Updated state with generated_response
    """
    print("---RESPONSE GENERATION---")
    current_time = "2025-05-09 12:30:02"
    user_login = "fouratmansouri"
    print(f"Process initiated by {user_login} at {current_time}")
    
    query = state["rewritten_query"]
    
    # First check for filtered documents
    if "filtered_documents" in state and state["filtered_documents"]:
        documents = state["filtered_documents"]
        print(f"Using {len(documents)} filtered documents for response generation")
    
    # Instead of using all retrieved documents, get only those above threshold
    else:
        print("No filtered documents available, extracting high-quality documents")
        # Define threshold for document quality
        threshold_value = 0.7  # Adjust based on your scoring system
        
        # Get retrieved documents
        retrieved_docs = state.get("retrieved_documents", [])
        
        # Filter only high-quality documents
        high_quality_docs = []
        for doc in retrieved_docs:
            # Check if document meets quality threshold
            if doc.metadata.get('relevance_score', 0) >= threshold_value:
                high_quality_docs.append(doc)
        
        documents = high_quality_docs
        print(f"Using {len(documents)} high-quality documents for response generation")
        
        # If no high-quality documents found or documents are insufficient, try web search
        if not documents or len(documents) < 2:  # Adjust minimum document threshold as needed
            print("Insufficient high-quality documents, attempting web search")
            
            # Perform web search to get additional information
            try:
                web_docs = web_search_tool.invoke({"query": query})
                web_results = [Document(page_content=d["content"]) for d in web_docs]
                print(f"Retrieved {len(web_results)} documents from web search")
                
                # Combine existing high-quality documents with web results
                documents = high_quality_docs + web_results
                print(f"Using combined {len(documents)} documents for response generation")
                
                # Still check if we have enough data
                if not documents:
                    raise ValueError("No documents found from web search either")
                    
            except Exception as e:
                print(f"Web search failed: {str(e)}")
                # If web search fails or returns no results, return informative response
                insufficient_data_response = f"Je n'ai pas trouvé d'informations suffisamment pertinentes concernant '{query}', même après recherche sur le web. Pourriez-vous reformuler votre question ou fournir plus de détails?"
                
                response_metadata = {
                    "total_documents": len(retrieved_docs),
                    "relevant_documents": 0,
                    "generated_by": user_login,
                    "generation_timestamp": current_time,
                    "status": "insufficient_data",
                    "web_search_attempted": True,
                    "web_search_successful": False
                }
                
                return {
                    "generated_response": insufficient_data_response,
                    "response_metadata": response_metadata,
                    "insufficient_data": True
                }
    
    # Format documents for context
    docs_txt = format_docs(documents)
    
    rag_prompt = """Vous êtes un assistant pour des tâches de question-réponse.

Voici le contexte à utiliser pour répondre à la question :

{context}

Réfléchissez soigneusement au contexte ci-dessus.

Maintenant, examinez la question de l'utilisateur :

{question}

Fournissez une réponse à cette question en utilisant uniquement le contexte ci-dessus.

Utilisez un maximum de trois phrases et gardez la réponse concise.

Réponse :"""
    # Generate response using RAG
    rag_prompt_formatted = rag_prompt.format(
        context=docs_txt, 
        question=query
    )
    
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    print(generation.content)
    
    # Determine source of documents used
    web_search_used = "web_search_results" in state or any(getattr(doc, 'source', '') == 'web_search' for doc in documents)
    
    # Add metadata about document filtering to the response
    response_metadata = {
        "total_documents": len(state.get("retrieved_documents", [])),
        "relevant_documents": len(documents),
        "generated_by": user_login,
        "generation_timestamp": current_time,
        "status": "success",
        "web_search_used": web_search_used
    }
    
    return {
        "generated_response": generation.content,
        "response_metadata": response_metadata,
        "documents_used": documents,  # Track which documents were actually used
        "web_search_used": web_search_used  # Flag if web search was used
    }

def response_validation(state):
    """
    LLM judge evaluates if response answers the re-written query.
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updated state with validation result and reason
    """
    print("---RESPONSE VALIDATION---")
    rewritten_query = state["rewritten_query"]
    generated_response = state["generated_response"]
    
    # Answer grader instructions
    answer_grader_instructions = """Vous êtes un enseignant en train de corriger un quiz.
Vous recevrez une QUESTION et une RÉPONSE D'ÉLÈVE.
Voici les critères de notation à suivre :
(1) La RÉPONSE DE L'ÉLÈVE aide à répondre à la QUESTION.
Note :
Une note de oui signifie que la réponse de l'élève respecte tous les critères. C'est la note la plus élevée (meilleure).
L'élève peut recevoir une note de oui même si la réponse contient des informations supplémentaires qui ne sont pas explicitement demandées dans la question.
Une note de non signifie que la réponse de l'élève ne respecte pas tous les critères. C'est la note la plus basse que vous pouvez attribuer.
Expliquez votre raisonnement étape par étape afin de garantir la justesse de votre raisonnement et de votre conclusion.
Évitez d'énoncer directement la bonne réponse dès le départ.
"""
    # Grader prompt
    answer_grader_prompt = """QUESTION : \n\n {rewritten_query} \n\n RÉPONSE DE L'ÉLÈVE : {generated_response}.
Retournez un JSON avec deux clés :  
- binary_score : une valeur 'oui' ou 'non' indiquant si la RÉPONSE DE L'ÉLÈVE respecte les critères.  
- explanation : une explication justifiant la note attribuée.
"""
    # Format the prompt correctly using the state variables
    answer_grader_prompt_formatted = answer_grader_prompt.format(
        rewritten_query=rewritten_query, 
        generated_response=generated_response
    )
    
    # Call the LLM
    result = llm_json_mode.invoke(
        [SystemMessage(content=answer_grader_instructions)]
        + [HumanMessage(content=answer_grader_prompt_formatted)]
    )
    
    # Parse the JSON response
    validation_result = json.loads(result.content)
    
    if validation_result["binary_score"] == "oui":
        return {
            "validation_result": validation_result["binary_score"],
            "validation_reason": validation_result["explanation"],
            "final_response": state["generated_response"]  # <-- Add this line
        }
    else:
        return {
            "validation_result": validation_result["binary_score"],
            "validation_reason": validation_result["explanation"]
        }

def fallback_retry(state):
    """
    Fall back to web search for additional content when validation fails.
    Implements retry loop with maximum 3 attempts.
    Combines web search results with only the documents that passed the threshold.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updated state with web_search_results and incremented retry_count
    """
    print("---FALLBACK & RETRY---")
    rewritten_query = state["rewritten_query"]
    retry_count = state.get("retry_count", 0)
    
    # Check if we've reached max retries
    if retry_count >= state["max_retries"]:
        # If max retries reached, prepare final response acknowledging limitations
        final_response = f"Après plusieurs tentatives, je n'arrive pas à trouver une réponse complète concernant '{rewritten_query}'. Veuillez envisager de reformuler votre question ou de consulter un professionnel de la fiscalité pour des conseils spécifiques concernant la retenue à la source."
        return {
            "retry_count": retry_count + 1,
            "final_response": final_response
        }
    
    # Perform web search to get additional information
    web_docs = web_search_tool.invoke({"query": rewritten_query})
    web_results = [Document(page_content=d["content"]) for d in web_docs]
    
    # Get current documents and filter only those above threshold
    current_docs = state.get("retrieved_documents", [])
    above_threshold_docs = state.get("above_threshold_docs", [])
    
    # If above_threshold_docs is not available in state, we need to compute it
    if not above_threshold_docs and current_docs:
        # This assumes you have a way to determine which docs are above threshold
        # If you don't have this in state, you'll need to add logic to filter them here
        # For example, if you have relevance scores:
        # above_threshold_docs = [doc for doc in current_docs if doc.metadata.get('relevance_score', 0) >= threshold_value]
        pass
    
    # Combine web results with documents above threshold
    combined_docs = above_threshold_docs + web_results
    
    return {
        "retrieved_documents": combined_docs,
        "web_search_results": web_results,
        "above_threshold_docs": above_threshold_docs,  # Keep track of above-threshold docs
        "retry_count": retry_count + 1
    }

### Edge Functions
def does_response_answer_query(state):
    """
    Determines whether the response answers the rewritten query based on validation.

    Args:
        state (dict): The current graph state

    Returns:
        str: "oui" if response answers query, "non" if not
    """
    print("---DECISION POINT: DOES RESPONSE ANSWER QUERY?---")
    
    validation_result = state["validation_result"]
    
    if validation_result == "oui":
        print("---DECISION: RESPONSE VALIDATED SUCCESSFULLY---")
        # Set final response to return to user
        state["final_response"] = state["generated_response"]
        return "oui"
    else:
        print(f"---DECISION: VALIDATION FAILED - {state['validation_reason']}---")
        return "non"
    
from langgraph.graph import StateGraph, END
from IPython.display import Image, display
import logging

def configure_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    # Add file handler to save logs to file
    file_handler = logging.FileHandler("conversation_system.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    return logger

def build_graph():
    logger = configure_logging()
    logger.info("Building Withholding Tax RAG Pipeline")
    
    workflow = StateGraph(ConversationState)
    
    # Define the nodes according to the schema
    workflow.add_node("query_rewriting", query_rewriting)                    # 2. Query Re-writing
    workflow.add_node("retrieval_with_filtering", retrieval_with_filtering)  # 3. Combined Retrieval & Filtering 
    workflow.add_node("response_generation", response_generation)            # 4. Response Generation
    workflow.add_node("response_validation", response_validation)            # 5. Response Validation
    workflow.add_node("fallback_retry", fallback_retry)                      # 6. Fallback & Retry
    
    # Build graph
    # Start with user query (1) then goes to query rewriting (2)
    workflow.set_entry_point("query_rewriting")
    
    # Linear flow from query rewriting to retrieval with filtering
    workflow.add_edge("query_rewriting", "retrieval_with_filtering")
    
    # Retrieval with filtering to response generation
    workflow.add_edge("retrieval_with_filtering", "response_generation")
    
    # Response generation to validation
    workflow.add_edge("response_generation", "response_validation")
    
    # Conditional edge from validation
    workflow.add_conditional_edges(
        "response_validation",
        does_response_answer_query,
        {
            "oui": END,  # Return validated response to user
            "non": "fallback_retry"  # Go to fallback mechanism
        },
    )
    
    # Fallback can retry the retrieval+filtering step
    workflow.add_edge("fallback_retry", "retrieval_with_filtering")
    
    # Compile
    logger.info("Compiling Withholding Tax RAG Pipeline with LLM Document Filtering")
    logger.info(f"Pipeline built by {{'user': 'fouratmansouri', 'timestamp': '2025-05-09 01:02:53'}}")
    graph = workflow.compile()

    return graph

def main(query):
    graph = build_graph()
    # Initialize the state with a user query
    inputs = {"original_query": "c'est quoi la plateforme TEJ ?", "max_retries": 3}
    for event in graph.stream(inputs, stream_mode="values"):
        if isinstance(event, dict) and "final_response" in event:
            print("\n\nFINAL RESPONSE:")
            print(event["final_response"])
            return event["final_response"]
        else:
            print(".", end="", flush=True)
            return 'Pas de réponse trouvée'



