# filepath: c:\Users\friti\Downloads\projfinance\models\sql_agent.py
"""
Module pour les requêtes SQL et l'agent SQL.
"""
import re
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent, AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from models.utils import DB_PATH, COMMON_QUERIES, extract_clean_sql

def setup_deepseek_agent():
    """Configure et retourne un agent SQL avec deepseek"""
    # Connexion à la base de données SQLite
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
    
    # Création du modèle DeepSeek
    try:
        llm = ChatOllama(model="deepseek-r1:8b")
        print("✅ Utilisation de deepseek-r1:8b comme modèle LLM")
    except Exception:
        try:
            llm = ChatOllama(model="llama3.2:latest")
            print("✅ Utilisation de llama3.2:latest comme modèle LLM")
        except Exception:
            llm = ChatOllama(model="mistral:latest")
            print("✅ Utilisation de mistral:latest comme modèle LLM")
    
    # Prompt optimisé pour DeepSeek
    prompt = ChatPromptTemplate.from_template("""
    ### Tu es un expert SQL qui génère des requêtes précises pour analyser des données fiscales.
    
    ### Table de données:
    ```sql
    CREATE TABLE transactions(
      idx INTEGER PRIMARY KEY,
      date TEXT,                -- Date ISO
      service TEXT,             -- Description service
      montant REAL,             -- Montant
      raw_taux TEXT,            -- Taux brut original
      taux_applique REAL,       -- Taux normalisé (%)
      statut TEXT,              -- "Correcte" ou "Incorrecte"
      taux_attendu REAL,        -- Taux réglementaire
      source_ref TEXT,          -- Référence légale
      document_source TEXT,     -- Document source
      date_application TEXT,    -- Date d'application de la règle
      paragraphe TEXT,          -- Extrait pertinent du texte
      lien_source TEXT,         -- Lien vers le document source
      beneficiaire TEXT,        -- Bénéficiaire de la règle
      seuil TEXT                -- Seuil d'application
    )
    ```
    
    ### INSTRUCTIONS IMPORTANTES:
    - Génère UNIQUEMENT une requête SQL valide sans aucun texte additionnel
    - Pour les listes de données, utilise TOUJOURS "SELECT *" 
    - Pour les agrégations (somme, moyenne, comptage), utilise les fonctions appropriées comme SUM(), COUNT(), etc.
    - Attribue toujours un alias aux colonnes d'agrégation
    - Pour les champs textuels, ajoute toujours des conditions pour exclure les valeurs vides et "Non spécifié"
    
    ### Question: {input}
    
    {agent_scratchpad}
    """)
    
    # Création de la boîte à outils et de l'agent
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        prompt=prompt,
        max_iterations=2,
        handle_parsing_errors=True,
        early_stopping_method="force"
    )
    
    return agent

def extract_sql_query(text):
    """Extrait uniquement la requête SQL du texte, en ignorant tout le reste."""
    print(f"Texte brut à analyser: {text}")  # Debug
    
    # 0. Vérifier si on doit appliquer une requête prédéfinie
    for regex, query in COMMON_QUERIES.items():
        if re.search(regex, text, re.IGNORECASE):
            print(f"Requête prédéfinie reconnue: {query}")
            return query
    
    # Utiliser la fonction d'extraction propre
    clean_sql = extract_clean_sql(text)
    if clean_sql:
        print(f"Requête nettoyée extraite: {clean_sql}")
        return clean_sql
    
    # 4. NOUVELLE MÉTHODE: Rechercher des mots-clés et construire une requête simple
    if "transaction" in text.lower() and "montant" in text.lower() and re.search(r"(\d+|deux|trois|quatre|cinq)\s+(?:transaction|mont)", text.lower()):
        # Déterminer la limite
        limit = 5  # par défaut
        
        if match := re.search(r"(\d+|deux|trois|quatre|cinq)\s+(?:transaction|mont)", text.lower()):
            num_str = match.group(1).lower()
            if num_str == "deux":
                limit = 2
            elif num_str == "trois":
                limit = 3
            elif num_str == "quatre":
                limit = 4
            elif num_str == "cinq":
                limit = 5
            else:
                try:
                    limit = int(num_str)
                except:
                    pass
        
        # Construire une requête simple
        sql = f"SELECT * FROM transactions ORDER BY montant DESC LIMIT {limit};"
        print(f"Requête construite à partir des mots-clés: {sql}")  # Debug
        return sql
    
    # Recherche d'agrégation par mots-clés
    if any(word in text.lower() for word in ["somme", "total", "addition"]) and "montant" in text.lower():
        sql = "SELECT SUM(montant) AS total_montant FROM transactions;"
        print(f"Requête d'agrégation (somme) construite: {sql}")
        return sql
    
    if any(word in text.lower() for word in ["compt", "nombre", "combien"]) and "transaction" in text.lower():
        sql = "SELECT COUNT(*) AS nombre_transactions FROM transactions;"
        print(f"Requête d'agrégation (count) construite: {sql}")
        return sql
    
    # Recherche de requêtes liées aux champs enrichis
    if any(word in text.lower() for word in ["lien", "document", "source", "url"]):
        sql = "SELECT * FROM transactions WHERE lien_source IS NOT NULL AND lien_source != '' AND lien_source != 'Non spécifié';"
        print(f"Requête sur liens de documents: {sql}")
        return sql
        
    if "date_application" in text.lower() or "quand" in text.lower():
        sql = "SELECT * FROM transactions WHERE date_application IS NOT NULL AND date_application != '' AND date_application != 'Non spécifié';"
        print(f"Requête sur dates d'application: {sql}")
        return sql
        
    if "paragraphe" in text.lower() or "extrait" in text.lower():
        sql = "SELECT * FROM transactions WHERE paragraphe IS NOT NULL AND paragraphe != '' AND paragraphe != 'Non spécifié';"
        print(f"Requête sur paragraphes: {sql}")
        return sql
        
    if "bénéficiaire" in text.lower() or "beneficiaire" in text.lower():
        sql = "SELECT * FROM transactions WHERE beneficiaire IS NOT NULL AND beneficiaire != '' AND beneficiaire != 'Non spécifié';"
        print(f"Requête sur bénéficiaires: {sql}")
        return sql
    
    print("Aucune requête SQL n'a pu être extraite")  # Debug
    return None

def direct_sql_query(user_question):
    """Génère une requête SQL directement à partir de la question de l'utilisateur."""
    try:
        # Créer le modèle LLM (utiliser deepseek-r1:8b en priorité)
        try:
            llm = ChatOllama(model="llama3.2:latest")   
            print("✅ Utilisation de llama3.2:latest comme modèle LLM")
        except Exception as e:
            print(f"⚠️ Erreur avec llama3.2: {e}, tentative avec mistral")
            try:
                llm = ChatOllama(model="mistral:latest")
                print("⚠️ Fallback vers mistral:latest")
            except Exception as e2:
                print(f"⚠️ Erreur avec Mistral: {e2}, fallback sur llama3")
                llm = ChatOllama(model="llama3")
                print("⚠️ Fallback vers llama3")
        
        # Prompt optimisé pour DeepSeek
        prompt = """
        ### Tu es un expert SQL spécialisé dans la transformation de questions en requêtes SQL

        ### Ta tâche:
        Convertis la question en une requête SQL précise et RÉPONDS UNIQUEMENT AVEC LA REQUÊTE SQL.
        N'ajoute ni explication ni contexte. Uniquement le code SQL pur.

        ### Règles importantes:
        1. Pour les listes de données, utilise toujours SELECT * pour éviter les problèmes d'affichage.
        2. Pour les agrégations (somme, moyenne, comptage), utilise les fonctions appropriées:
          - Utilise SUM(montant) pour calculer des sommes
          - Utilise COUNT(*) pour compter des transactions
          - Utilise AVG(montant) pour calculer des moyennes
          - Attribue toujours un alias aux colonnes d'agrégation (ex: SUM(montant) AS total_montant)
        3. Pour les champs textuels comme date_application, paragraphe, lien_source, beneficiaire:
          - Ajoute toujours des conditions pour exclure les valeurs vides et "Non spécifié"
          - Exemple: WHERE date_application != '' AND date_application != 'Non spécifié'

        ### Schéma de la base de données:
        ```
        Table: transactions (
            idx INTEGER PRIMARY KEY,
            date TEXT,              -- Date ISO
            service TEXT,           -- Description   du service
            montant REAL,           -- Montant
            raw_taux TEXT,          -- Taux brut original
            taux_applique REAL,     -- Taux normalisé (%)
            statut TEXT,            -- "Correcte" ou "Incorrecte" 
            taux_attendu REAL,      -- Taux réglementaire
            source_ref TEXT,        -- Référence légale
            document_source TEXT,   -- Document source
            date_application TEXT,  -- Date d'application de la règle
            paragraphe TEXT,        -- Extrait pertinent
            lien_source TEXT,       -- URL du document source
            beneficiaire TEXT,      -- Bénéficiaire de la règle
            seuil TEXT              -- Seuil d'application éventuel
        )
        ```

        ### Question de l'utilisateur: {question}

        ### Requête SQL:
        """
        
        # Invoquer le modèle
        response = llm.invoke(prompt.format(question=user_question))
        response_text = response.content
        print(f"Réponse LLM brute: {response_text}")
        
        # Nettoyer et extraire la requête SQL
        sql = extract_clean_sql(response_text)
        
        if sql:
            return sql, None
        
        # Fallback spécifique pour les champs enrichis
        if any(word in user_question.lower() for word in ["lien", "source", "document", "url"]):
            sql = "SELECT * FROM transactions WHERE lien_source IS NOT NULL AND lien_source != '' AND lien_source != 'Non spécifié';"
            return sql, None
            
        if "date_application" in user_question.lower() or "quand" in user_question.lower():
            sql = "SELECT * FROM transactions WHERE date_application IS NOT NULL AND date_application != '' AND date_application != 'Non spécifié';"
            return sql, None
            
        if "paragraphe" in user_question.lower() or "extrait" in user_question.lower():
            sql = "SELECT * FROM transactions WHERE paragraphe IS NOT NULL AND paragraphe != '' AND paragraphe != 'Non spécifié';"
            return sql, None
            
        if "bénéficiaire" in user_question.lower() or "beneficiaire" in user_question.lower():
            sql = "SELECT * FROM transactions WHERE beneficiaire IS NOT NULL AND beneficiaire != '' AND beneficiaire != 'Non spécifié';"
            return sql, None
            
        # Autres fallbacks existants...
        if re.search(r"(\d+|deux|trois|quatre|cinq)\s+transaction.*mont", user_question.lower()):
            limit = 5  # par défaut
            if match := re.search(r"(\d+|deux|trois|quatre|cinq)", user_question.lower()):
                num_str = match.group(1).lower()
                if num_str == "deux":
                    limit = 2
                elif num_str == "trois":
                    limit = 3
                elif num_str == "quatre":
                    limit = 4
                elif num_str == "cinq":
                    limit = 5
                else:
                    try:
                        limit = int(num_str)
                    except:
                        pass
            sql = f"SELECT * FROM transactions ORDER BY montant DESC LIMIT {limit};"
            return sql, None
        
        # Fallback pour les questions d'agrégation
        if any(word in user_question.lower() for word in ["somme", "total", "addition"]) and "montant" in user_question.lower():
            sql = "SELECT SUM(montant) AS total_montant FROM transactions;"
            return sql, None
        
        if any(word in user_question.lower() for word in ["compt", "nombre", "combien"]) and "transaction" in user_question.lower():
            sql = "SELECT COUNT(*) AS nombre_transactions FROM transactions;"
            return sql, None
        
        return None, "Impossible de générer une requête SQL"
    
    except Exception as e:
        return None, f"Erreur lors de la génération de la requête: {str(e)}"