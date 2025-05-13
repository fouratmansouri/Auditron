#!/usr/bin/env python
"""
Main application for tax rate verification processing + Chatbot RAG integration.
"""
import json, re, math, pathlib, argparse, os, uuid, sqlite3, hashlib
from typing import List, Dict
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response
from tqdm import tqdm
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
import time
# Import from modular components
from models.utils import XLS_IN, JSON_RULES, rs_or_pct, guess_cat
from models.db import export_to_excel
# Update imports to use Qdrant functions
from models.tax_classification import classify_transaction_with_llm, load_tax_rules, get_qdrant_index, create_qdrant_index, search_similar_transactions
from models.sql_query import extract_sql_query, direct_sql_query, setup_deepseek_agent

# ─── Début modifications pour intégration Chatbot ────────────────────────────
import sys
sys.path.append(os.path.dirname(__file__))  # pour trouver chatbot.py

# Importer main du chatbot (fonction principale pour le RAG)
try:
    from models.chatbot import main as chatbot_main
    print("✓ Importation réussie de la fonction main du chatbot")
except ImportError as e:
    print(f"⚠️ Erreur lors de l'importation de main du chatbot: {e}")
    # Fonction de secours si l'import échoue
    def chatbot_main(query):
        return f"Le chatbot n'a pas pu être chargé correctement. Erreur: {str(e)}. Veuillez réessayer plus tard."

# Ces fonctions ne sont pas utilisées dans la nouvelle implémentation
def clear_chat_history(session_id):
    """Fonction fictive pour compatibilité"""
    print(f"Session {session_id} réinitialisée (factice)")
    return True

def cleanup_inactive_sessions():
    """Fonction fictive pour compatibilité"""
    print("Nettoyage des sessions inactives (factice)")
    return True

# ─── Fin modifications Chatbot ───────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(__file__), "transactions.db")

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

# Global variable for Qdrant collection name
_QDRANT_COLLECTION_NAME = "tax_rules"

def initialize_qdrant():
    """Initialize Qdrant index for vector search"""
    try:
        # Chargement ou création de l'index Qdrant
        qdrant_index = get_qdrant_index()
        if qdrant_index:
            print("✅ Index Qdrant chargé avec succès")
            return qdrant_index
        else:
            print("❌ Échec du chargement de l'index Qdrant")
            return None
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation de Qdrant: {e}")
        return None

# ─── pipeline principal ────────────────────────────────────
def run_pipeline(rebuild: bool=False):
    # 1. Qdrant
    if rebuild:
        qdrant_index = create_qdrant_index()
        if qdrant_index is None:
            print("❌ Impossible de construire l'index Qdrant. Vérifiez le fichier JSON.")
            return
    
    qdrant_index = initialize_qdrant()
    if qdrant_index is None:
        print("❌ Impossible de charger l'index Qdrant.")
        return

    # 2. Lire Excel (concat.toutes feuilles)
    try:
        xl = pd.ExcelFile(XLS_IN)
        df = pd.concat([xl.parse(s) for s in xl.sheet_names], ignore_index=True)
        print(f"✅ Fichier Excel chargé: {len(df)} lignes")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du fichier Excel: {e}")
        return

    # 3. Détecter la colonne taux d'abord sur le nom
    taux_cols = [c for c in df.columns if "taux" in c.lower()]
    if taux_cols:
        TAUX_COL = taux_cols[0]
    else:
        TAUX_COL = next((c for c in df.columns
                       if df[c].astype(str).str.contains(r"%|rs", case=False, na=False).any()), None)
    
    if not TAUX_COL:
        print("❌ Impossible de trouver une colonne de taux dans le fichier Excel")
        return
    
    print(f"✅ Colonne de taux identifiée: {TAUX_COL}")

    # 4. Charger les règles fiscales depuis le JSON
    try:
        with open(JSON_RULES, encoding="utf-8") as f:
            rules_json = json.load(f)
        print(f"✅ {len(rules_json)} règles JSON chargées pour la classification")
    except Exception as e:
        print(f"❌ Erreur lors du chargement des règles JSON: {e}")
        rules_json = []  # Tableau vide en cas d'échec
    
    # Charger les règles fiscales
    tax_rules = load_tax_rules()

    # 5. Remplir la BDD
    conn = init_db()
    cur = conn.cursor()

    # Statistiques de classification
    stats = {"llm": 0, "total": 0, "fails": 0}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Traitement des transactions"):
        try:
            date_iso = pd.to_datetime(row["Date"]).date().isoformat()
            svc = str(row.get("intitulCompta", row.get("Service", "")))
            montant = float(str(row.get("MontantTTC", row.get("Montant", 0)))
                            .replace(" ","").replace(" ","").replace(",",".")) 
            raw_taux = str(row.get(TAUX_COL,"")).strip()
            tap = rs_or_pct(raw_taux)
            cat = guess_cat(svc)  # Gardons cette estimation initiale comme backup
            year = pd.to_datetime(row["Date"]).year

            stats["total"] += 1

            # Classification automatique avec LLM
            print(f"🔍 Classification de: '{svc}' (montant: {montant})")
            llm_result = classify_transaction_with_llm(svc, montant, year, rules_json)
            if llm_result:
                # Utiliser la classification du LLM
                stats["llm"] += 1
                taux_att = llm_result.get("taux", math.nan)
                ref = llm_result.get("ref", "Non spécifié")
                doc = llm_result.get("doc", "Non spécifié")
                date_app = llm_result.get("date_app", "Non spécifié")
                parag = llm_result.get("parag", "Non spécifié")
                lien = llm_result.get("lien", "")
                benef = llm_result.get("benef", "Non spécifié")
                seuil = "Non spécifié"
                
                print(f"✓ Classification LLM: {llm_result.get('categorie')}, taux={taux_att}%")
            else:
                # Échec de classification - uniquement des valeurs par défaut
                stats["fails"] = stats.get("fails", 0) + 1
                print(f"❌ Classification LLM échouée pour '{svc}'")
                taux_att = math.nan
                ref = "Classification échouée"
                doc = "Non disponible"
                date_app = "Non disponible"
                parag = "Classification automatique impossible"
                lien = ""
                benef = "Non spécifié"
                seuil = "Non spécifié"

            # Vérification stricte du taux (sans marge d'erreur)
            correct = (not math.isnan(tap) and not math.isnan(taux_att) and tap == taux_att)
            statut = "Correcte" if correct else "Incorrecte"

            # Insertion dans la base de données
            cur.execute("""
            INSERT INTO transactions
              (idx, date, service, montant, raw_taux, taux_applique,
               statut, taux_attendu, source_ref, document_source,
               date_application, paragraphe, lien_source, beneficiaire, seuil)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
              idx, date_iso, svc, montant, raw_taux, tap,
              statut, taux_att, ref, doc,
              date_app, parag, lien, benef, seuil
            ))
        except Exception as e:
            print(f"❌ Erreur lors du traitement de la ligne {idx}: {e}")

    # Statistiques de classification
    print(f"\nStatistiques de classification:")
    if stats["total"] > 0:
        print(f"  Total: {stats['total']} transactions")
        print(f"  LLM: {stats['llm']} transactions ({stats['llm']/stats['total']*100:.1f}%)")
        print(f"  Échecs: {stats.get('fails', 0)} transactions ({stats.get('fails', 0)/stats['total']*100:.1f}%)")
    else:
        print("  Aucune transaction traitée")
        
    # Mise à jour du fichier Excel avec les données enrichies
    conn.commit()
    export_to_excel(conn)
    conn.close()
    print("✅ Base SQLite et fichier Excel mis à jour avec données enrichies")

# ─── Flask ─────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def table():
    conn = get_connection()
    
    result = None
    query = ""
    sql_query = ""
    error = None
    rows = None
    filtered = False
    aggregate_result = None  #      Pour stocker les résultats d'agrégation
    show_detailed = True     # Toujours afficher les détails enrichis
    
    if request.method == "POST":
        query = request.form.get("query", "")
        if query:
            sql_found = False
            
            # Tenter de générer une requête SQL
            try:
                print(f"Tentative de génération directe SQL pour: {query}")
                direct_sql, direct_error = direct_sql_query(query)
                
                if direct_sql:
                    sql_query = direct_sql
                    sql_found = True
                    print(f"Requête SQL générée directement: {sql_query}")
                else:
                    print(f"La génération directe a échoué: {direct_error}")
                    
                    # Si la génération directe échoue, essayer avec l'agent DeepSeek
                    print(f"Tentative avec l'agent DeepSeek pour: {query}")
                    try:
                        agent = setup_deepseek_agent()
                        result_obj = agent.invoke({"input": query})
                        
                        # Imprimer la réponse brute pour le débogage
                        print(f"Réponse brute de l'agent: {result_obj}")
                        
                        # Extraire la requête SQL des logs ou du résultat
                        if hasattr(agent, "agent_executor") and hasattr(agent.agent_executor, "logs"):
                            logs = agent.agent_executor.logs
                            print(f"Logs de l'agent disponibles: {logs}")
                            extracted_sql = extract_sql_query(logs)
                            if extracted_sql:
                                sql_query = extracted_sql
                                sql_found = True
                        
                        # Sinon chercher dans le résultat brut
                        if not sql_found and result_obj:
                            result_text = str(result_obj)
                            print(f"Analyse du résultat brut: {result_text}")
                            extracted_sql = extract_sql_query(result_text)
                            if extracted_sql:
                                sql_query = extracted_sql
                                sql_found = True
                    except Exception as agent_error:
                        print(f"Erreur avec l'agent DeepSeek: {agent_error}")
            
            except Exception as e:
                error = f"Erreur d'exécution: {str(e)}"
                print(f"Erreur: {error}")
            
            # Si SQL trouvé, l'exécuter
            if sql_found and sql_query:
                # Nettoyer la requête - extraire uniquement jusqu'au premier point-virgule
                if ";" in sql_query:
                    sql_query = sql_query.split(";")[0].strip() + ";"
                
                # Convertir 'FETCH FIRST n ROWS' en 'LIMIT n' pour SQLite
                sql_query = re.sub(r"FETCH\s+FIRST\s+(\d+)\s+ROWS", r"LIMIT \1", sql_query, flags=re.IGNORECASE)
                
                print(f"Exécution de la requête SQL: {sql_query}")
                
                try:
                    # Vérifier si c'est une requête d'agrégation
                    is_aggregation = bool(re.search(r"(SUM|AVG|COUNT|MIN|MAX|TOTAL)\s*\(", sql_query, re.IGNORECASE))
                    
                    # Exécuter la requête
                    rows = execute_query(sql_query)
                    filtered = True
                    
                    # Traitement spécial pour les requêtes d'agrégation
                    if is_aggregation and len(rows) == 1:
                        # Récupérer le nom de la colonne et la valeur
                        first_row = rows[0]
                        column_names = first_row.keys()
                        column_name = column_names[0] if len(column_names) > 0 else "Résultat"
                        value = first_row[0] if len(first_row) > 0 else None
                        
                        # Personnaliser l'affichage selon le type d'agrégation
                        if ("SUM" in sql_query.upper() or "TOTAL" in sql_query.upper()) and "montant" in sql_query.lower():
                            aggregate_result = {
                                "type": "sum",
                                "label": "Somme totale des montants",
                                "value": value,
                                "formatted": f"{value:.2f}" if value is not None else "0.00"
                            }
                        elif "COUNT" in sql_query.upper():
                            aggregate_result = {
                                "type": "count",
                                "label": "Nombre de transactions",
                                "value": value,
                                "formatted": str(value) if value is not None else "0"
                            }
                        else:
                            # Format générique pour d'autres types d'agrégation
                            aggregate_result = {
                                "type": "generic",
                                "label": column_name.replace("_", " ").title(),
                                "value": value,
                                "formatted": str(value) if value is not None else "-"
                            }
                    
                    result = f"Requête SQL exécutée avec succès."
                except Exception as e:
                    error = f"Erreur SQL: {str(e)}"
                    rows = None
            else:
                error = "Aucune requête SQL valide n'a pu être générée."
    
    # Afficher toutes les transactions si aucune requête n'a été exécutée
    if rows is None:
        rows = execute_query("SELECT * FROM transactions")
        filtered = False
    
    return render_template(
        "table.html", 
        rows=rows, 
        year=datetime.now().year,
        result=result,
        query=query,
        sql_query=sql_query,
        error=error,
        filtered=filtered,
        aggregate_result=aggregate_result,
        show_detailed=show_detailed
    )

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    """Endpoint pour le chatbot avec support de streaming."""
    # Si la requête est JSON (depuis l'API)
    if request.is_json:
        user_message = request.json.get("query", "")
        session_id = request.json.get("session_id", "")
        use_streaming = request.json.get("streaming", True)  # Activer par défaut
    # Si la requête est form data (depuis le formulaire HTML)
    else:
        user_message = request.form.get("message", "")
        session_id = request.form.get("session_id", "")
        use_streaming = request.form.get("streaming", "true").lower() == "true"  # Convertir string en bool
    
    # Créer un nouvel ID de session si non fourni
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Action spéciale pour effacer l'historique
    if user_message.lower() in ["clear", "reset", "effacer", "nouvelle conversation"]:
        clear_chat_history(session_id)
        return jsonify({
            "response": "Conversation réinitialisée. Comment puis-je vous aider?",
            "session_id": session_id
        })
    
    # Si le streaming est demandé, utiliser l'endpoint de streaming
    if use_streaming:
        # Rediriger vers l'endpoint de streaming
        return redirect(url_for('chat_stream', 
                               message=user_message, 
                               session_id=session_id))
    
    # Sinon, utiliser la méthode standard (non-streaming)
    try:
        # Utiliser la fonction main du module chatbot
        bot_response = chatbot_main(user_message)
        
        return jsonify({
            "response": bot_response,
            "session_id": session_id
        })
    except Exception as e:
        print(f"❌ Error in chat endpoint: {str(e)}")
        return jsonify({
            "response": f"Désolé, une erreur s'est produite lors du traitement de votre message. Erreur: {str(e)}",
            "session_id": session_id,
            "error": True
        })

@app.route("/chat_stream")
def chat_stream():
    """Endpoint qui génère une réponse en streaming (mot par mot)."""
    user_message = request.args.get("message", "")
    session_id = request.args.get("session_id", str(uuid.uuid4()))
    
    def generate():
        """Générateur pour le streaming de la réponse."""
        try:
            # Préfixe qui indique le début du streaming
            yield "data: {\"type\": \"start\", \"session_id\": \"" + session_id + "\"}\n\n"
            
            # Obtenir la réponse complète du chatbot
            response = chatbot_main(user_message)
            
            # Simuler la génération mot par mot
            words = response.split()
            
            for word in words:
                # Échapper les guillemets pour le JSON
                escaped_word = word.replace('"', '\\"')
                yield f"data: {{\"type\": \"token\", \"content\": \"{escaped_word} \"}}\n\n"
                time.sleep(0.04)  # Petite pause entre les mots pour l'effet visuel
            
            # Indiquer que la génération est terminée
            yield "data: {\"type\": \"end\"}\n\n"
        
        except Exception as e:
            error_msg = str(e).replace('"', '\\"')
            yield f"data: {{\"type\": \"error\", \"content\": \"{error_msg}\"}}\n\n"
    
    # Configurer la réponse pour le streaming avec Server-Sent Events
    return Response(generate(), mimetype="text/event-stream")

# ─── CLI + lancement Flask ────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild", action="store_true",
                   help="reconstruit l'index Qdrant et la BDD")
    args = p.parse_args()
    run_pipeline(rebuild=args.rebuild)
    if not args.rebuild:
        app.run(debug=True, port=8000,use_reloader=False)