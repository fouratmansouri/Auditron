#!/usr/bin/env python
"""
Main application for tax rate verification processing.
"""
import json, re, math, pathlib, argparse, os
import pandas as pd
from flask import Flask, render_template, request
from tqdm import tqdm
from datetime import datetime

# Import from modular components
from models.utils import XLS_IN, JSON_RULES, rs_or_pct, guess_cat
from models.db import init_db, export_to_excel, get_connection, execute_query
from models.embeddings import get_faiss_retriever, build_faiss
from models.tax_classification import classify_transaction_with_llm, load_tax_rules
from models.sql_query import extract_sql_query, direct_sql_query, setup_deepseek_agent

# ─── pipeline principal ────────────────────────────────────
def run_pipeline(rebuild: bool=False):
    # 1. FAISS
    if rebuild:
        retriever = build_faiss()
        if retriever is None:
            print("❌ Impossible de construire l'index FAISS. Vérifiez le fichier JSON.")
            return
    
    retriever = get_faiss_retriever()
    if retriever is None:
        print("❌ Impossible de charger l'index FAISS.")
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
    aggregate_result = None  # Pour stocker les résultats d'agrégation
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

# ─── CLI + lancement Flask ────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild", action="store_true",
                   help="reconstruit l'index FAISS et la BDD")
    args = p.parse_args()
    run_pipeline(rebuild=args.rebuild)
    if not args.rebuild:
        app.run(debug=True, port=8000)