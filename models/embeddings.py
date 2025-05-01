#!/usr/bin/env python
import sqlite3
import pandas as pd
from config import DB_PATH

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
Vector embeddings and FAISS operations.
"""
import os, re, json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from models.utils import JSON_RULES, FAISS_DIR, EMBED_MODEL

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

def build_faiss():
    """Reconstruct the FAISS index from JSON rules."""
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
    
    docs = []
    for r in rules:
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
        
        docs.append(Document(
            page_content=txt,
            metadata={
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
        ))
    
    print(f"🔄 Création de l'index FAISS avec {len(docs)} documents")
    vect = FAISS.from_documents(docs, embedder)
    FAISS.save_local(vect, str(FAISS_DIR))
    print("✅ Index FAISS reconstruit avec données enrichies.")
    return vect

def get_faiss_retriever():
    """Get a FAISS retriever, loading from disk or rebuilding if necessary."""
    try:
        vect = FAISS.load_local(
            str(FAISS_DIR),
            embeddings=embedder,
            allow_dangerous_deserialization=True
        )
        print("✅ Index FAISS chargé depuis le disque")
        return vect.as_retriever(search_kwargs={"k": 8})
    except Exception as e:
        print(f"❌ Erreur lors du chargement de l'index FAISS: {e}")
        print("🔄 Tentative de reconstruction de l'index...")
        vect = build_faiss()
        if vect is None:
            print("❌ Impossible de construire l'index FAISS. Vérifiez le fichier JSON.")
            return None
        return vect.as_retriever(search_kwargs={"k": 8})