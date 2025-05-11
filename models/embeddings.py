#!/usr/bin/env python
import sqlite3
import pandas as pd
from config import DB_PATH
import os, re, json
import hashlib
import uuid
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from models.utils import JSON_RULES, EMBED_MODEL
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

# Configuration Qdrant
QDRANT_URL = "http://localhost:6333"  # Modifiez si votre serveur Qdrant est ailleurs
QDRANT_COLLECTION = "projfinance_tax_rules"  # Nom de la collection
VECTOR_SIZE = 384  # Taille des vecteurs du modèle all-MiniLM-L6-v2

def init_db():
    """Initialise la base de données SQLite"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS transactions")
    c.execute("""
    CREATE TABLE transactions(
      idx INTEGER PRIMARY KEY,
      date TEXT,
      service TEXT,
      montant REAL,
      raw_taux TEXT,
      taux_applique REAL,
      statut TEXT,
      taux_attendu REAL,
      source_ref TEXT,
      document_source TEXT,
      date_application TEXT,
      paragraphe TEXT,
      lien_source TEXT,
      beneficiaire TEXT,
      seuil TEXT
    )""")
    conn.commit()
    return conn

def get_connection():
    """Retourne une connexion à la base de données avec row_factory configuré"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def execute_query(query, params=None):
    """Exécute une requête SQL et retourne les résultats"""
    conn = get_connection()
    try:
        if params:
            rows = conn.execute(query, params).fetchall()
        else:
            rows = conn.execute(query).fetchall()
        return rows
    except sqlite3.Error as e:
        print(f"Erreur SQL: {str(e)}")
        return None
    finally:
        conn.close()

def insert_transaction(idx, date_iso, service, montant, raw_taux, tap, 
                       statut, taux_att, ref, doc, date_app, parag, lien, benef, seuil):
    """Insère une transaction dans la base de données"""
    conn = get_connection()
    try:
        conn.execute("""
        INSERT INTO transactions
          (idx, date, service, montant, raw_taux, taux_applique,
           statut, taux_attendu, source_ref, document_source,
           date_application, paragraphe, lien_source, beneficiaire, seuil)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
          idx, date_iso, service, montant, raw_taux, tap,
          statut, taux_att, ref, doc,
          date_app, parag, lien, benef, seuil
        ))
        conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"Erreur d'insertion: {str(e)}")
        return False
    finally:
        conn.close()

def get_all_transactions():
    """Retourne toutes les transactions"""
    return execute_query("SELECT * FROM transactions")

"""
Vector embeddings using Qdrant instead of FAISS.
"""

# Initialize embeddings model with a try-except block to handle memory issues
try:
    # Try using a smaller model first
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("✅ Using all-MiniLM-L6-v2 embedding model")
except Exception as e:
    print(f"⚠️ Error loading primary model: {e}")
    try:
        # Fall back to an even smaller model
        embedder = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L3-v2")
        print("⚠️ Fallback to paraphrase-MiniLM-L3-v2 model")
    except Exception as e:
        print(f"❌ Critical error loading embedding models: {e}")
        # Create a dummy embedder that will warn but not crash the application
        from langchain_core.embeddings import Embeddings
        class DummyEmbedder(Embeddings):
            def embed_documents(self, texts):
                print("⚠️ Using dummy embedder - functionality limited")
                return [[0.0] * 384] * len(texts)
            def embed_query(self, text):
                return [0.0] * 384
        embedder = DummyEmbedder()

# Helper function to convert string IDs to valid UUID-compatible IDs
def string_to_uuid(string_id):
    """Convert a string to a deterministic UUID by hashing it"""
    hash_object = hashlib.md5(string_id.encode())
    hex_dig = hash_object.hexdigest()
    return hex_dig  # Return hexadecimal string directly

def build_qdrant():
    """Reconstruct the Qdrant collection from JSON rules."""
    print(f"🔄 Chargement des règles depuis {JSON_RULES}")
    try:
        with open(JSON_RULES, encoding="utf-8") as f:
            rules = json.load(f)
        print(f"✅ {len(rules)} règles chargées")
        
        # Afficher un aperçu des premières règles
        for i, r in enumerate(rules[:3]):
            print(f"\nRègle {i+1}:")
            print(f"  Année: {r.get('année')}")
            print(f"  Taux: {r.get('taux')}")
            print(f"  Date application: {r.get('date_application')}")
            print(f"  Bénéficiaire: {r.get('bénéficiaire')}")
            print(f"  Paragraphe: {r.get('paragraphe', '')[:50]}...")
            print(f"  Lien source: {r.get('lien_source', '')}")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du fichier JSON: {e}")
        print(f"Chemin actuel: {os.getcwd()}")
        print(f"Le fichier existe: {os.path.exists(JSON_RULES)}")
        return None
    
    # Initialize Qdrant client
    try:
        qdrant = QdrantClient(url=QDRANT_URL)
        print(f"✅ Connexion établie avec Qdrant sur {QDRANT_URL}")
    except Exception as e:
        print(f"❌ Erreur de connexion à Qdrant: {e}")
        print("Assurez-vous que le serveur Qdrant est en cours d'exécution.")
        return None
    
    # Create or recreate the collection
    try:
        qdrant.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"✅ Collection '{QDRANT_COLLECTION}' créée ou réinitialisée")
    except Exception as e:
        print(f"❌ Erreur lors de la création de la collection: {e}")
        return None
    
    # Process and upload documents
    points = []
    batch_size = 50
    processed_count = 0
    error_count = 0
    id_mapping = {}
    
    for idx, r in enumerate(rules):
        try:
            # Extraction améliorée de l'année
            annee_str = r.get("année", "0")
            annee = 0
            if isinstance(annee_str, str):
                match = re.search(r'\d+', annee_str)
                if match:
                    annee = int(match.group(0))
            
            # Contenu enrichi pour une meilleure recherche sémantique
            txt = (f"Catégorie: {r.get('categorie', 'Non spécifié')} | "
                   f"Type: {r.get('type_de_revenu_ou_service', 'Non spécifié')} | "
                   f"Taux: {r.get('taux', 'N/A')} | "
                   f"Bénéficiaire: {r.get('bénéficiaire', 'Non spécifié')} | "
                   f"Application: {r.get('date_application', 'Non spécifiée')}")
            
            original_id = f"rule_{idx}"
            point_id = string_to_uuid(original_id)
            id_mapping[point_id] = original_id
            
            # Generate embedding
            try:
                embedding = embedder.embed_documents([txt])[0]
            except Exception as embed_error:
                print(f"⚠️ Erreur d'embedding pour règle {idx}: {str(embed_error)}")
                error_count += 1
                continue
            
            # Create Qdrant point
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": txt,
                    "original_id": original_id,
                    "annee": annee,
                    "taux_num": r.get("taux_num"),
                    "reference": r.get("référence_légale", "Non spécifié"),
                    "source_file": r.get("source_file", "Non spécifié"),
                    "date_application": r.get("date_application", "Non spécifié"),
                    "paragraphe": r.get("paragraphe", "Non spécifié"),
                    "lien_source": r.get("lien_source", ""),
                    "beneficiaire": r.get("bénéficiaire", "Non spécifié"),
                    "seuil": r.get("seuil", "Non spécifié")
                }
            )
            points.append(point)
            processed_count += 1
            
            # Upload batch if it reaches the threshold
            if len(points) >= batch_size:
                qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
                print(f"✓ Lot chargé: {processed_count}/{len(rules)} règles ({error_count} erreurs)")
                points = []
        
        except Exception as e:
            print(f"❌ Erreur de traitement pour règle {idx}: {str(e)}")
            error_count += 1
    
    # Upload remaining points
    if points:
        qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
        print(f"✓ Lot final chargé: {processed_count}/{len(rules)} règles")
    
    print(f"✅ Collection Qdrant construite avec {processed_count}/{len(rules)} règles!")
    print(f"Total d'erreurs rencontrées: {error_count}")
    
    # Return the client object to be used as a retriever
    return qdrant

class QdrantRetriever:
    """Custom retriever class for Qdrant."""
    
    def __init__(self, client, collection_name, k=8):
        self.client = client
        self.collection_name = collection_name
        self.k = k
    
    def get_relevant_documents(self, query):
        """Search for similar documents based on the query."""
        try:
            # Generate query embedding
            query_vector = embedder.embed_query(query)
            
            # Search in Qdrant
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=self.k
            )
            
            # Convert results to Document objects
            documents = []
            for result in search_result:
                payload = result.payload
                documents.append(Document(
                    page_content=payload.get("text", ""),
                    metadata={
                        "score": result.score,
                        "annee": payload.get("annee", 0),
                        "taux_num": payload.get("taux_num"),
                        "reference": payload.get("reference", ""),
                        "source_file": payload.get("source_file", ""),
                        "date_application": payload.get("date_application", ""),
                        "paragraphe": payload.get("paragraphe", ""),
                        "lien_source": payload.get("lien_source", ""),
                        "beneficiaire": payload.get("beneficiaire", ""),
                        "seuil": payload.get("seuil", "")
                    }
                ))
            
            return documents
            
        except Exception as e:
            print(f"❌ Erreur de recherche dans Qdrant: {str(e)}")
            return []

def get_qdrant_retriever(rebuild=False):
    """Get a Qdrant retriever, connecting to existing collection or rebuilding if necessary."""
    try:
        # Connect to Qdrant
        qdrant = QdrantClient(url=QDRANT_URL)
        
        # Check if collection exists
        collections = qdrant.get_collections().collections
        collection_exists = any(c.name == QDRANT_COLLECTION for c in collections)
        
        if not collection_exists or rebuild:
            print(f"🔄 Collection '{QDRANT_COLLECTION}' n'existe pas ou reconstruction demandée")
            qdrant = build_qdrant()
            if qdrant is None:
                print("❌ Impossible de construire la collection Qdrant")
                return None
        else:
            print(f"✅ Connexion établie à la collection '{QDRANT_COLLECTION}'")
        
        # Return custom retriever
        return QdrantRetriever(qdrant, QDRANT_COLLECTION, k=8)
        
    except Exception as e:
        print(f"❌ Erreur de connexion à Qdrant: {e}")
        print("Assurez-vous que le serveur Qdrant est en cours d'exécution.")
        return None