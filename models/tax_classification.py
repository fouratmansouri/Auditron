import json
import os
import re
import datetime
import traceback
import xlsxwriter
import logging
import time
import random
from typing import Dict, List, Optional, Any, Union, Tuple
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

# Replace FAISS imports with Qdrant
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings

# Add required imports for Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range

# Import constants from utils
from .utils import EMBED_MODEL, JSON_RULES, MARCH_FRANCH

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Constants
_QDRANT_CLIENT = None
_QDRANT_COLLECTION_NAME = "tax_rules"
# Create unique path for each process/instance with timestamp and random suffix
_timestamp = int(time.time())
_random_suffix = random.randint(1000, 9999)
_QDRANT_LOCAL_PATH = f"./qdrant_data_{_timestamp}_{_random_suffix}"  # Local storage path for embedded Qdrant

# =====================================================
# 1. RULES LOADING
# =====================================================

def load_tax_rules() -> List[Dict]:
    """
    Charge les règles fiscales depuis un fichier JSON.
    
    Returns:
        Liste des règles fiscales ou liste vide en cas d'échec
    """
    # Use the JSON_RULES constant from utils
    json_path = JSON_RULES
    
    try:
        logger.info(f"Tentative de chargement depuis: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            rules = json.load(f)
            logger.info(f"✅ {len(rules)} règles fiscales chargées depuis {json_path}")
            return rules
    except FileNotFoundError:
        logger.warning(f"⚠️ Fichier non trouvé: {json_path}")
    except json.JSONDecodeError as e:
        logger.error(f"❌ Erreur de décodage JSON dans {json_path}: {e}")
        # Erreur critique - fichier JSON mal formé
        raise
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement depuis {json_path}: {e}")
    
    # Fallback paths if the constant didn't work
    possible_paths = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "retenues_final_enrichi.json"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "retenues_final_enrichi.json"),
        "C:\\Users\\friti\\Downloads\\projfinance\\retenues_final_enrichi.json"
    ]
    
    for path in possible_paths:
        try:
            logger.info(f"Tentative de chargement depuis: {path}")
            with open(path, "r", encoding="utf-8") as f:
                rules = json.load(f)
                logger.info(f"✅ {len(rules)} règles fiscales chargées depuis {path}")
                return rules
        except FileNotFoundError:
            logger.warning(f"⚠️ Fichier non trouvé: {path}")
        except json.JSONDecodeError as e:
            logger.error(f"❌ Erreur de décodage JSON dans {path}: {e}")
            # Erreur critique - fichier JSON mal formé
            raise
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement depuis {path}: {e}")
    
    # Si aucun chemin n'a fonctionné
    logger.error("❌ Impossible de charger les règles fiscales")
    return []

# =====================================================
# 2. QDRANT INDEX MANAGEMENT 
# =====================================================

def get_qdrant_client() -> QdrantClient:
    """
    Obtient ou crée un client Qdrant en mode embarqué (pas besoin de Docker).
    
    Returns:
        QdrantClient: Instance du client Qdrant
    """
    global _QDRANT_CLIENT
    
    if _QDRANT_CLIENT is not None:
        return _QDRANT_CLIENT
        
    try:
        # Create directory for Qdrant data if it doesn't exist
        os.makedirs(_QDRANT_LOCAL_PATH, exist_ok=True)
        
        # Use embedded/local mode with persistent storage
        _QDRANT_CLIENT = QdrantClient(path=_QDRANT_LOCAL_PATH)
        logger.info(f"✅ Connected to embedded Qdrant using local storage at: {_QDRANT_LOCAL_PATH}")
        
        # Test if we can list collections
        try:
            _QDRANT_CLIENT.get_collections()
        except Exception as e:
            logger.warning(f"⚠️ Error accessing collections: {e}, will reinitialize client")
            _QDRANT_CLIENT = QdrantClient(path=_QDRANT_LOCAL_PATH)
        
        return _QDRANT_CLIENT
        
    except Exception as e:
        logger.error(f"❌ Error initializing Qdrant client: {e}")
        traceback.print_exc()
        logger.warning("⚠️ Fallling back to in-memory Qdrant (data will not persist)")
        _QDRANT_CLIENT = QdrantClient(":memory:")
        return _QDRANT_CLIENT

def create_qdrant_index(rules_json: List[Dict] = None, force_recreate: bool = True) -> Optional[Qdrant]:
    """
    Crée un index Qdrant à partir des règles fiscales.
    
    Args:
        rules_json: Liste des règles fiscales
        force_recreate: Force la recréation de l'index même s'il existe déjà
        
    Returns:
        Qdrant: Instance du vectorstore Qdrant ou None en cas d'échec
    """
    try:
        if not rules_json:
            rules_json = load_tax_rules()
            
        if not rules_json:
            logger.error("❌ Aucune règle fiscale trouvée")
            return None
            
        logger.info(f"📊 Création d'un index Qdrant à partir de {len(rules_json)} règles")
        
        # Initialize embeddings using constant from utils
        embeddings_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        
        # Prepare documents for Qdrant
        documents = []
        
        for i, rule in enumerate(rules_json):
            # Build document content with all relevant fields
            content = f"""
            Type: {rule.get('type_de_revenu_ou_service', '')}
            Taux: {rule.get('taux', '')}
            Référence: {rule.get('référence_légale', '')}
            Année: {rule.get('année', '')}
            Bénéficiaire: {rule.get('bénéficiaire', '')}
            Description: {rule.get('description', '')}
            """
            
            # Convertir le taux en nombre si possible
            taux_num = None
            taux = rule.get('taux', '')
            if isinstance(taux, (int, float)):
                taux_num = float(taux)
            elif isinstance(taux, str):
                match = re.search(r'(\d+(?:[.,]\d+)?)\s*%', taux)
                if match:
                    try:
                        taux_num = float(match.group(1).replace(',', '.'))
                    except ValueError:
                        pass
            
            # Safely handle the seuil value
            seuil_value = 0.0
            seuil = rule.get('seuil')
            if seuil is not None:
                try:
                    seuil_value = float(seuil)
                except (ValueError, TypeError):
                    # If conversion fails, use default
                    seuil_value = 0.0
            
            # Metadata with all fields for filtering and retrieval
            metadata = {
                "rule_id": i,
                "type": rule.get('type_de_revenu_ou_service', ''),
                "categorie": rule.get('type_de_revenu_ou_service', ''),
                "taux": rule.get('taux', ''),
                "taux_num": taux_num if taux_num is not None else 0.0, 
                "référence": rule.get('référence_légale', ''),
                "année": str(rule.get('année', '')),  # Ensure it's a string for matching
                "date_application": str(rule.get('année', '')),
                "bénéficiaire": rule.get('bénéficiaire', ''),
                "description": rule.get('description', ''),
                "paragraphe": rule.get('description', ''),
                "source_file": rule.get('source_file', ''),
                "lien_source": rule.get('lien_source', ''),
                "seuil": seuil_value,
                "original_rule": json.dumps(rule)
            }
            
            documents.append(Document(page_content=content, metadata=metadata))
        
        # Get Qdrant client
        client = get_qdrant_client()
        
        try:
            # Check if collection exists and delete it if force_recreate
            collections = client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if _QDRANT_COLLECTION_NAME in collection_names:
                if force_recreate:
                    logger.info(f"🗑️ Suppression de la collection existante: '{_QDRANT_COLLECTION_NAME}'")
                    client.delete_collection(_QDRANT_COLLECTION_NAME)
                else:
                    logger.info(f"📂 Collection '{_QDRANT_COLLECTION_NAME}' existe déjà, pas de recréation")
                    return Qdrant(
                        client=client,
                        collection_name=_QDRANT_COLLECTION_NAME,
                        embeddings=embeddings_model
                    )
            
            # Get vector size from embedding model
            vector_size = len(embeddings_model.embed_query("test"))
            
            # Create collection
            client.create_collection(
                collection_name=_QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            
            # Create Qdrant store
            qdrant_store = Qdrant(
                client=client,
                collection_name=_QDRANT_COLLECTION_NAME,
                embeddings=embeddings_model
            )
            
            # Add documents
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            qdrant_store.add_texts(texts=texts, metadatas=metadatas)
            
            logger.info(f"✅ Index Qdrant créé pour la collection '{_QDRANT_COLLECTION_NAME}'")
            return qdrant_store
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création de l'index: {e}")
            traceback.print_exc()
            return None
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la création de l'index Qdrant: {e}")
        traceback.print_exc()
        return None

def get_qdrant_index(rules_json: List[Dict] = None, force_reload: bool = False) -> Optional[Qdrant]:
    """
    Charge un index Qdrant existant ou en crée un nouveau.
    
    Args:
        rules_json: Liste des règles fiscales
        force_reload: Force le rechargement même si déjà en cache
        
    Returns:
        Qdrant: Instance du vectorstore Qdrant ou None en cas d'échec
    """
    try:
        client = get_qdrant_client()
        embeddings_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if _QDRANT_COLLECTION_NAME in collection_names and not force_reload:
            logger.info(f"📂 Utilisation de l'index Qdrant existant: '{_QDRANT_COLLECTION_NAME}'")
            return Qdrant(
                client=client,
                collection_name=_QDRANT_COLLECTION_NAME,
                embeddings=embeddings_model
            )
        else:
            logger.info("🔍 Aucun index Qdrant trouvé ou rechargement forcé, création d'un nouvel index...")
            return create_qdrant_index(rules_json, force_recreate=True)
    
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement de l'index Qdrant: {e}")
        traceback.print_exc()
        return create_qdrant_index(rules_json)

# =====================================================
# 3. RULE-BASED CLASSIFICATION
# =====================================================

def classify_by_rules_only(
    service: str, 
    montant: float, 
    year: int, 
    rules_json: List[Dict]
) -> Optional[Dict[str, Any]]:
    """
    Classifie une transaction en utilisant uniquement les règles statiques.
    
    Args:
        service: Description du service
        montant: Montant de la transaction
        year: Année de la transaction
        rules_json: Liste des règles fiscales
        
    Returns:
        Classification ou None si aucune règle ne correspond
    """
    try:
        if not rules_json:
            logger.warning("Aucune règle fiscale fournie pour la classification déterministe")
            return None
            
        # Convertir année en string pour correspondre au format dans les règles
        year_str = str(year)
        service_lower = service.lower()
        
        # 1. Filtre par année (exact ou plage)
        year_candidates = []
        for rule in rules_json:
            rule_year = str(rule.get('année', '')).strip()
            
            # Gestion des plages d'années (ex: "2020-2023")
            if '-' in rule_year:
                try:
                    start_year, end_year = rule_year.split('-')
                    if int(start_year) <= year <= int(end_year):
                        year_candidates.append(rule)
                except (ValueError, TypeError):
                    # Si le format n'est pas correct, on vérifie l'égalité simple
                    if rule_year == year_str:
                        year_candidates.append(rule)
            # Gestion des années individuelles
            elif rule_year == year_str:
                year_candidates.append(rule)
                
        if not year_candidates:
            logger.debug(f"Aucune règle trouvée pour l'année {year}")
            return None
            
        # 2. Filtre par seuil minimum (si défini)
        amount_candidates = []
        for rule in year_candidates:
            seuil = rule.get('seuil', 0)
            try:
                seuil_value = float(seuil) if seuil is not None else 0
                if montant >= seuil_value:
                    amount_candidates.append(rule)
            except (ValueError, TypeError):
                # Si le seuil n'est pas un nombre, on inclut la règle quand même
                amount_candidates.append(rule)
                
        if not amount_candidates:
            logger.debug(f"Aucune règle avec seuil <= {montant} pour l'année {year}")
            return None
            
        # 3. Match sur le type de service/revenus
        # D'abord essayer une correspondance exacte
        for rule in amount_candidates:
            rule_type = rule.get('type_de_revenu_ou_service', '').lower()
            if rule_type == service_lower:
                return _convert_rule_to_result(rule, "rule-based-exact")
                
        # Ensuite chercher une correspondance par inclusion
        for rule in amount_candidates:
            rule_type = rule.get('type_de_revenu_ou_service', '').lower()
            if rule_type in service_lower or service_lower in rule_type:
                return _convert_rule_to_result(rule, "rule-based-partial")
                
        # Si on arrive ici, c'est qu'on n'a pas trouvé de correspondance directe
        logger.debug(f"Aucune correspondance de type pour '{service}' parmi {len(amount_candidates)} règles")
        return None
        
    except Exception as e:
        logger.error(f"Erreur lors de la classification par règles: {e}")
        traceback.print_exc()
        return None

def _convert_rule_to_result(rule: Dict[str, Any], method: str = "rule-based") -> Dict[str, Any]:
    """
    Convertit une règle fiscale en résultat de classification standardisé.
    
    Args:
        rule: Règle fiscale
        method: Méthode de classification utilisée
    
    Returns:
        Dictionnaire formaté pour le résultat
    """
    # Extraction du taux
    taux = rule.get('taux', '')
    taux_num = None
    if isinstance(taux, (int, float)):
        taux_num = float(taux)
    elif isinstance(taux, str):
        # Essayer d'extraire le pourcentage
        match = re.search(r'(\d+(?:[.,]\d+)?)\s*%', taux)
        if match:
            try:
                taux_num = float(match.group(1).replace(',', '.'))
            except ValueError:
                pass
    
    return {
        "categorie": rule.get('type_de_revenu_ou_service', ''),
        "taux": taux_num,
        "ref": rule.get('référence_légale', ''),
        "doc": rule.get('document_source', ''),
        "date_app": rule.get('année', ''),
        "parag": rule.get('description', ''),
        "lien": rule.get('lien_source', ''),
        "benef": rule.get('bénéficiaire', ''),
        "methode_classification": method
    }

# =====================================================
# 4. LLM-BASED CLASSIFICATION
# =====================================================

def classify_transaction_with_llm(
    service: str, 
    montant: float, 
    year: int,
    description: str = "", 
    rules_json: List[Dict] = None
) -> Dict[str, Any]:
    """
    Classifie une transaction en utilisant un LLM pour déterminer le taux de retenue approprié.
    
    Args:
        service: Description du service
        montant: Montant de la transaction
        year: Année de la transaction
        description: Description additionnelle du service pour une meilleure classification
        rules_json: Liste de règles fiscales (optionnel)
        
    Returns:
        Dictionnaire contenant la classification ou un fallback en cas d'échec
    """
    try:
        # Essayer d'abord avec les règles déterministes
        if rules_json:
            rule_result = classify_by_rules_only(service, montant, year, rules_json)
            if rule_result:
                logger.info(f"✅ Classification réussie avec règles déterministes pour: {service}")
                return rule_result
        
        # Si pas de résultat avec les règles, on continue avec le LLM
        # Initialiser le modèle LLM
        try:
            llm = ChatOllama(model="deepseek-r1:8b")
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation du modèle LLM: {e}")
            # Fallback to rule-based if LLM initialization fails
            if rules_json and len(rules_json) > 0:
                # Find a suitable fallback rule based on the transaction amount
                fallback_rule = None
                for rule in rules_json:
                    if str(rule.get('année', '')) == str(year):
                        fallback_rule = rule
                        break
                
                if fallback_rule:
                    return _convert_rule_to_result(fallback_rule, "rule-based-fallback")
            
            # If no suitable rule found, return a default classification
            return {
                'categorie': 'Non catégorisé - LLM indisponible',
                'taux': None,
                'ref': 'Erreur LLM',
                'doc': 'Non disponible',
                'date_app': str(year),
                'parag': 'Erreur lors de l\'initialisation du modèle LLM',
                'lien': '',
                'benef': 'Non spécifié',
                'methode_classification': 'error'
            }

        # Rechercher des règles similaires dans Qdrant
        qdrant_index = get_qdrant_index(rules_json)
        similar_docs = []
        year_str = str(year)
        
        if qdrant_index:
            # Construire une requête basée sur le service et la description additionnelle
            query = f"Classification fiscale pour: {service}"
            if description:
                query += f" {description}"
            query += f" montant: {montant} année: {year}"
            
            logger.info(f"🔍 Recherche avec requête: '{query}'")
            
            try:
                # Essayer d'abord avec un filtre sur l'année (méthode optimale)
                client = get_qdrant_client()
                embeddings_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
                query_vector = embeddings_model.embed_query(query)
                
                filter_condition = Filter(
                    must=[
                        FieldCondition(
                            key="année",
                            match=MatchValue(value=year_str)
                        ),
                        FieldCondition(
                            key="seuil",
                            range=Range(lte=float(montant))
                        )
                    ]
                )
                
                search_result = client.search(
                    collection_name=_QDRANT_COLLECTION_NAME,
                    query_vector=query_vector,
                    limit=5,
                    query_filter=filter_condition
                )
                
                # Convertir les résultats Qdrant en documents
                if search_result:
                    for hit in search_result:
                        metadata = hit.payload
                        content = f"""
                        Type: {metadata.get('type', '')}
                        Taux: {metadata.get('taux', '')}
                        Référence: {metadata.get('référence', '')}
                        Année: {metadata.get('année', '')}
                        Bénéficiaire: {metadata.get('bénéficiaire', '')}
                        Description: {metadata.get('description', '')}
                        """
                        similar_docs.append(Document(page_content=content, metadata=metadata))
            except Exception as e:
                logger.warning(f"⚠️ Erreur lors de la recherche filtrée, fallback à la recherche simple: {e}")
                # Fallback à la méthode simple
                all_docs = qdrant_index.similarity_search(query, k=10)
                # Filtrer les documents par année
                similar_docs = [d for d in all_docs if str(d.metadata.get("année", "")) == year_str][:5]
            
            # Extraire le contexte des règles similaires
            context = "\n\nRÈGLES FISCALES SIMILAIRES:\n"
            if similar_docs:
                for i, doc in enumerate(similar_docs):
                    rule_data = json.loads(doc.metadata.get("original_rule", "{}"))
                    context += f"\nRègle {i+1}:\n"
                    context += f"- Type: {rule_data.get('type_de_revenu_ou_service', '')}\n"
                    context += f"- Taux: {rule_data.get('taux', '')}\n"
                    context += f"- Référence: {rule_data.get('référence_légale', '')}\n"
                    context += f"- Année: {rule_data.get('année', '')}\n"
                    context += f"- Bénéficiaire: {rule_data.get('bénéficiaire', '')}\n"
            else:
                context = "\n\nAUCUNE RÈGLE FISCALE SIMILAIRE TROUVÉE POUR L'ANNÉE SPÉCIFIÉE.\n"
        else:
            context = "\n\nATTENTION: Base de données des règles fiscales non disponible.\n"
        
        # Mise à jour du système de prompt avec tous les taux applicables en Tunisie
        system_message = """Tu es un expert fiscal spécialisé dans la classification des transactions pour déterminer les taux de retenue à la source (RAS) en Tunisie.

PRINCIPES GÉNÉRAUX DE CLASSIFICATION :
    1. ANALYSE PAR NATURE DE SERVICE :
    - Acquisition de biens/matériel/équipements           → retenue 1 %¹
    - Honoraires/commissions/loyers (activités non commerciales) → retenue 10 %¹
    - Honoraires servis aux assujettis au régime réel     → retenue 3 %²
    - Plus-values sur titres par non-résidents             → retenue 15 %³
    - Plus-values sur biens immobiliers par non-résidents  → retenue 10 %³
    - Revenus de capitaux mobiliers (intérêts, jetons de présence) → retenue 20 %⁴
    - Retenue sur TVA marché public (≥ 1000 DT)          → retenue 25 %⁵

2. MÉTHODOLOGIE D'ANALYSE :
   - Identifier la nature fondamentale (vente/prestation/mixte)
   - Vérifier le seuil de 1 000 DT hors TVA pour RAS libératoire
   - Considérer le statut fiscal du bénéficiaire (régime réel, non-résident…)
   - Appliquer les exceptions sectorielles et conventions internationales
   - Tenir compte de l'année fiscale (2021–2025+)

3. RÉFÉRENCES LÉGALES & DOCUMENTS SOURCES :
   ¹ Loi de finances 2021, art. 14 – acquisitions ≥ 1000 DT et honoraires/libératoires réduits
   ² ChaExpert (2021) – honoraires au régime réel
   ³ Ministère des Finances – retenues sur plus-values non-résidents
   ⁴ InFirst Auditors – retenue RAS définitive sur revenus de capitaux mobiliers
   ⁵ Finances.gov.tn – RAS TVA marchés publics ≥ 1000 DT

INSTRUCTIONS IMPORTANTES :
- Analyse chaque transaction de façon indépendante, NE CONSIDÈRE QUE l'année {year}.
- Précise toujours le rôle du « bénéficiaire » (acheteur/appliquant la retenue).
- Justifie chaque classification avec le texte de loi et la référence.
- En cas d'ambiguïté, détaille les interprétations possibles et mentionne la référence.
- Réponds exclusivement au format JSON défini ci-dessous.
"""

        human_message = f"""Classifie la transaction suivante pour déterminer le taux de retenue à la source approprié:

SERVICE: {service}
MONTANT: {montant:.2f}
ANNÉE FISCALE: {year}

{context}

Ton analyse doit être rigoureuse et détaillée. 

Réponds UNIQUEMENT au format JSON suivant:
{{
  "categorie": "Catégorie du service",
  "taux": 15.0,
  "ref": "Référence légale ou règle applicable",
  "doc": "Document source",
  "date_app": "Date d'application",
  "parag": "Explication détaillée de la classification",
  "lien": "Lien vers document source (si disponible)",
  "benef": "Type de bénéficiaire concerné"
}}

Le champ "taux" doit être un nombre sans symbole % (ex: 15.0, 0.0, 10.0, etc.).
""" 

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message)
        ]

        # Appel du modèle LLM
        response = llm.invoke(messages)
        response_text = response.content

        # Extraction du JSON de la réponse avec regex robuste
        try:
            # Rechercher le bloc JSON avec regex plus robuste
            json_match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', response_text)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
            else:
                # Si pas de JSON formaté, essayer de parser toute la réponse
                result = json.loads(response_text)
            
            # S'assurer que le taux est un nombre ou null
            if 'taux' in result:
                if result['taux'] is not None and result['taux'] != "null" and result['taux'] != "":
                    try:
                        taux_str = str(result['taux']).replace('%', '').strip()
                        if taux_str.lower() in ('null', 'none', ''):
                            result['taux'] = None
                        else:
                            result['taux'] = float(taux_str)
                    except (ValueError, TypeError):
                        logger.error(f"❌ Valeur de taux non valide: {result['taux']}, conversion à None")
                        result['taux'] = None
                else:
                    result['taux'] = None
            
            # Ajouter la méthode de classification
            result["methode_classification"] = "llm"
            
            # Vérifier si le lien source est manquant et qu'on a des documents similaires
            if (not result.get("lien") or result.get("lien") == "") and similar_docs:
                # Récupérer le lien du document le plus similaire
                result["lien"] = similar_docs[0].metadata.get("lien_source", "")
                if result["lien"]:
                    logger.info(f"⚠️ Lien manquant dans la réponse LLM, récupéré depuis document similaire")
            
            return result
            
        except json.JSONDecodeError:
            # Si impossible de parser le JSON, extraire manuellement les informations
            logger.error(f"Erreur de décodage JSON, tentative d'extraction manuelle: {response_text}")
            
            # Création d'un résultat de fallback
            result = {
                'categorie': 'Non catégorisé',
                'taux': None,
                'ref': 'Extraction manuelle',
                'doc': 'Non disponible',
                'date_app': str(year),
                'parag': response_text[:500],  # Limiter la taille
                'lien': '',
                'benef': 'Non spécifié',
                'methode_classification': 'llm-fallback'
            }
            
            # Tenter d'extraire le taux
            taux_match = re.search(r'(\d+(?:\.\d+)?)\s*%', response_text)
            if taux_match:
                result['taux'] = float(taux_match.group(1))
            
            # Si on a des documents similaires, utiliser le premier comme fallback
            if similar_docs:
                first_doc = similar_docs[0].metadata
                return _convert_rule_to_result(json.loads(first_doc.get("original_rule", "{}")), "llm-fallback")
            
            return result
    
    except Exception as e:
        logger.error(f"Erreur lors de la classification avec LLM: {e}")
        traceback.print_exc()
        
        # Toujours renvoyer un résultat, même en cas d'erreur
        return {
            'categorie': 'Erreur de classification',
            'taux': None,
            'ref': f'Erreur: {str(e)[:100]}',
            'doc': 'Non disponible',
            'date_app': str(year),
            'parag': 'Erreur lors du traitement de la requête',
            'lien': '',
            'benef': 'Non spécifié',
            'methode_classification': 'error'
        }

# =====================================================
# 5. SEARCH FUNCTIONS
# =====================================================

def search_similar_transactions(query_text: str, year: Optional[int] = None, qdrant_index: Optional[Qdrant] = None, top_k: int = 5) -> List[Document]:
    """
    Recherche des transactions similaires dans l'index Qdrant, avec filtrage optionnel par année.
    
    Args:
        query_text: Texte de la requête
        year: Année pour filtrage (optionnel)
        qdrant_index: Index Qdrant (si None, il sera chargé)
        top_k: Nombre de résultats à retourner
    
    Returns:
        Liste de documents similaires avec leurs métadonnées
    """
    try:
        if qdrant_index is None:
            qdrant_index = get_qdrant_index()
            
        if qdrant_index is None:
            logger.error("❌ Aucun index Qdrant disponible")
            return []
            
        # Si une année est spécifiée, essayer d'appliquer un filtre
        if year is not None:
            try:
                client = get_qdrant_client()
                embeddings_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
                query_vector = embeddings_model.embed_query(query_text)
                
                filter_condition = Filter(
                    must=[
                        FieldCondition(
                            key="année",
                            match=MatchValue(value=str(year))
                        )
                    ]
                )
                
                search_result = client.search(
                    collection_name=_QDRANT_COLLECTION_NAME,
                    query_vector=query_vector,
                    limit=top_k,
                    query_filter=filter_condition
                )
                
                # Convertir les résultats Qdrant en documents
                similar_docs = []
                if search_result:
                    for hit in search_result:
                        metadata = hit.payload
                        content = f"""
                        Type: {metadata.get('type', '')}
                        Taux: {metadata.get('taux', '')}
                        Référence: {metadata.get('référence', '')}
                        Année: {metadata.get('année', '')}
                        Bénéficiaire: {metadata.get('bénéficiaire', '')}
                        Description: {metadata.get('description', '')}
                        """
                        similar_docs.append(Document(page_content=content, metadata=metadata))
                    
                    return similar_docs
            except Exception as e:
                logger.warning(f"⚠️ Erreur lors de la recherche filtrée, fallback à la recherche simple: {e}")
        
        # Recherche de similarité standard (fallback)
        all_docs = qdrant_index.similarity_search(query_text, k=top_k * 2)  # Récupérer plus pour filtrer ensuite
        
        # Si une année est spécifiée, filtrer les résultats
        if year is not None:
            year_str = str(year)
            similar_docs = [d for d in all_docs if str(d.metadata.get("année", "")) == year_str][:top_k]
            if not similar_docs:
                # Si aucun résultat pour l'année spécifiée, retourner tous les résultats
                logger.warning(f"⚠️ Pas de résultat pour l'année {year}, retour de tous les résultats")
                similar_docs = all_docs[:top_k]
            return similar_docs
        else:
            return all_docs[:top_k]
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de la recherche de transactions similaires: {e}")
        traceback.print_exc()
        return []

# =====================================================
# 6. EXCEL EXPORT
# =====================================================

def export_to_excel(conn):
    """
    Exporte les données de la base vers un fichier Excel avec mise en forme avancée.
    
    Args:
        conn: Connexion à la base de données
        
    Returns:
        str: Nom du fichier généré ou None en cas d'échec
    """
    try:
        # Création d'un classeur Excel
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transactions_classifiees_{now}.xlsx"
        workbook = xlsxwriter.Workbook(filename)
        worksheet = workbook.add_worksheet("Transactions")
        
        # Styles pour le classeur - préparation de tous les formats nécessaires
        header_format = workbook.add_format({
            'bold': True, 
            'bg_color': '#C0C0C0',
            'border': 1
        })
        money_format = workbook.add_format({'num_format': '#,##0.00'})
        percent_format = workbook.add_format({'num_format': '0.00%'})
        desc_format = workbook.add_format({'text_wrap': True})
        
        # Récupération des données
        cursor = conn.cursor()
        cursor.execute("""
            SELECT t.id, t.service, t.montant, t.date_transaction, 
                   c.categorie, c.taux, c.reference, c.document_source, 
                   c.date_application, c.paragraphe, c.lien_source, c.beneficiaire,
                   c.methode_classification, c.description
            FROM transactions t
            LEFT JOIN classifications c ON t.id = c.transaction_id
            ORDER BY t.date_transaction DESC
        """)
        
        # Récupérer toutes les données en une seule fois
        rows = cursor.fetchall()
        
        # En-têtes
        headers = [
            "ID", "Service", "Montant", "Date", "Catégorie", "Taux", 
            "Référence", "Document", "Date app", "Explication", "Lien", "Bénéficiaire", 
            "Méthode", "Description"
        ]
        
        for col_num, header in enumerate(headers):
            worksheet.write(0, col_num, header, header_format)
        
        # Trouver l'index des colonnes spéciales
        montant_col = headers.index("Montant") if "Montant" in headers else 2
        taux_col = headers.index("Taux") if "Taux" in headers else 5 
        lien_col = headers.index("Lien") if "Lien" in headers else 10
        desc_col = headers.index("Description") if "Description" in headers else 13
        
        # Calcul des largeurs optimales des colonnes
        col_widths = [len(header) for header in headers]
        
        # Données
        for row_num, row in enumerate(rows):
            for col_num, cell_value in enumerate(row):
                # Mise à jour de la largeur maximale pour cette colonne
                if cell_value is not None:
                    col_widths[col_num] = max(col_widths[col_num], min(len(str(cell_value)), 50))
                
                # Formatage spécial pour certaines colonnes
                if col_num == montant_col:
                    # Format monétaire pour les montants
                    try:
                        montant_value = float(cell_value) if cell_value is not None else 0
                        worksheet.write_number(row_num + 1, col_num, montant_value, money_format)
                    except (ValueError, TypeError):
                        worksheet.write(row_num + 1, col_num, str(cell_value))
                elif col_num == taux_col:
                    # Format pourcentage pour les taux
                    if cell_value is not None and cell_value != "":
                        try:
                            taux_value = float(cell_value) / 100
                            worksheet.write_number(row_num + 1, col_num, taux_value, percent_format)
                        except (ValueError, TypeError):
                            worksheet.write(row_num + 1, col_num, str(cell_value))
                    else:
                        worksheet.write(row_num + 1, col_num, "N/A")
                elif col_num == lien_col:
                    # Écrire les liens comme des hyperliens
                    link = str(cell_value).strip() if cell_value else ""
                    if link and link != "#" and (link.startswith("http") or link.startswith("www")):
                        worksheet.write_url(row_num + 1, col_num, link, string=link)
                    else:
                        worksheet.write(row_num + 1, col_num, link)
                elif col_num == desc_col:
                    # Format spécial pour la colonne de description
                    description = cell_value if cell_value else ""
                    worksheet.write(row_num + 1, col_num, description, desc_format)
                else:
                    # Écriture normale pour les autres cellules
                    worksheet.write(row_num + 1, col_num, cell_value)
        
        # Ajustement des largeurs des colonnes
        for i, width in enumerate(col_widths):
            # Traitement spécial pour la colonne de description
            if i == desc_col:
                worksheet.set_column(i, i, 40)  # Largeur fixe plus grande pour la description
            else:
                # Limiter la largeur maximale et ajouter une marge
                adjusted_width = min(width, 30) * 1.1  # 10% de marge
                worksheet.set_column(i, i, adjusted_width)
            
        workbook.close()
        logger.info(f"✅ Export Excel terminé: {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'export Excel: {e}")
        traceback.print_exc()
        return None

# =====================================================
# 7. BATCH PROCESSING
# =====================================================

def process_batch_transactions(input_file: str, output_file: Optional[str] = None) -> Optional[str]:
    """
    Traite un lot de transactions à partir d'un fichier Excel.
    
    Args:
        input_file: Chemin vers le fichier Excel d'entrée
        output_file: Chemin vers le fichier Excel de sortie (optionnel)
        
    Returns:
        str: Chemin du fichier de sortie ou None en cas d'échec
    """
    try:
        import pandas as pd
        
        # Charger les règles fiscales
        rules = load_tax_rules()
        if not rules:
            logger.error("❌ Impossible de charger les règles fiscales")
            return None
            
        # Charger le fichier Excel
        logger.info(f"📂 Chargement du fichier: {input_file}")
        df = pd.read_excel(input_file)
        
        # Vérifier les colonnes requises
        required_columns = ["Description", "Montant HT", "Date"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"❌ Colonnes manquantes: {', '.join(missing_columns)}")
            return None
            
        # Préparer les résultats
        results = []
        total_rows = len(df)
        
        # Traiter chaque ligne
        for i, row in df.iterrows():
            svc = row["Description"]
            amt = float(row["Montant HT"])
            date = pd.to_datetime(row["Date"])
            year = date.year
            description = row.get("Description détaillée", "")
            
            logger.info(f"📝 Traitement {i+1}/{total_rows}: {svc}")
            
            # Essayer d'abord rule-based, puis LLM
            result = classify_transaction_with_llm(svc, amt, year, description, rules)
            results.append(result)
            
        # Créer un DataFrame avec les résultats
        results_df = pd.DataFrame(results)
        
        # Fusionner avec le DataFrame original
        output_df = pd.concat([df, results_df], axis=1)
        
        # Générer le nom du fichier de sortie si non spécifié
        if not output_file:
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"transactions_classifiees_{now}.xlsx"
            
        # Enregistrer le fichier
        output_df.to_excel(output_file, index=False)
        logger.info(f"✅ Traitement terminé: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du traitement par lots: {e}")
        traceback.print_exc()
        return None