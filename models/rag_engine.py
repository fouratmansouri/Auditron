# filepath: c:\Users\friti\Downloads\projfinance\models\rag_engine.py
"""
Module de RAG (Retrieval-Augmented Generation) et classification LLM.
"""
import re, json, os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.chat_models import ChatOllama
from models.utils import BASE, FAISS_DIR, JSON_RULES, EMBED_MODEL

# Cache pour les classifications LLM
_classification_cache = {}

def build_faiss():
    """Reconstruit l'index FAISS à partir des règles fiscales."""
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
    
    # Créer les embeddings
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    
    print(f"🔄 Création de l'index FAISS avec {len(docs)} documents")
    vect = FAISS.from_documents(docs, embedder)
    FAISS.save_local(vect, str(FAISS_DIR))
    print("✅ Index FAISS reconstruit avec données enrichies.")
    return vect

def get_faiss_retriever():
    """Renvoie un retriever FAISS prêt à l'emploi."""
    try:
        embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
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

def classify_transaction_with_llm(service, montant, year, rules_json):
    """Classification d'une transaction avec LLM en utilisant le fichier JSON de règles."""
    
    # Cache key pour éviter de réinterroger le LLM pour des cas similaires
    cache_key = f"{service}|{montant}|{year}"
    if cache_key in _classification_cache:
        return _classification_cache[cache_key]
    
    # Détection directe pour électricité (avant même de consulter le cache)
    if re.search(r"(?:achat\s+)?elec[dt]?ri(?:c|qu|k)(?:i|it[ée])", service.lower().replace('é', 'e')):
        return {
            "categorie": "electricite",
            "taux": 1.5,
            "ref": "Code IRPP-IS Art. 52",
            "doc": "Code fiscal 2021",
            "date_app": "Ancien - en vigueur 2021-2025",
            "parag": "Retenue de 1.5% sur achats d'électricité selon l'article 52 du Code IRPP-IS",
            "lien": "https://www.finances.gov.tn/fr/apercu-general-sur-la-fiscalite",
            "benef": "Fournisseurs d'électricité"
        }
    
    # 3. Limitation du nombre de règles pour éviter un prompt trop long
    rules_sample = []
    if len(rules_json) > 5:
        # Prendre d'abord les règles qui correspondent au service
        service_lower = service.lower()
        matching_rules = [r for r in rules_json 
                         if service_lower in str(r.get("type_de_revenu_ou_service", "")).lower()]
        
        # Si on a trouvé des règles correspondantes, les utiliser
        if matching_rules:
            rules_sample = matching_rules[:5]
        else:
            # Sinon prendre simplement les 5 premières règles
            rules_sample = rules_json[:5]
    else:
        rules_sample = rules_json
    
    rules = json.dumps(rules_sample, ensure_ascii=False)
    
    # Construction du prompt pour le LLM avec les règles spécifiques
    prompt = f"""
    ### CONTEXTE: Classification fiscale pour retenues à la source en Tunisie (2021-2025)

    Tu es un expert fiscal chargé de déterminer la catégorie de service, le taux applicable, 
    et la règle fiscale correspondante pour chaque transaction.

    **Instructions** :
    - Cherche dans les règles fiscales fournies une correspondance précise sur type de revenu, bénéficiaire, taux et référence.
    - Si aucune correspondance exacte, applique ces règles spécifiques:
        - Location de bureaux → 15% (catégorie: loyers)
        - Consultant, Marketing, Transport, Maintenance, Nettoyage → 10% (catégorie: honoraires/prestations)
        - Services informatiques/IT → 10% (catégorie: honoraires)
        - Construction/BTP → 10% (catégorie: honoraires)
        - Fournitures, matériel médical → 1% (catégorie: marchandises) si IS 15%
        - Électricité, achat électricité, fourniture d'électricité → 1.5% (catégorie: electricite) selon Art 52 Code IRPP-IS
        - Revenus mobiliers → 20% sur dividendes (catégorie: mobiliers)
        - Frais bancaires → Exonérés - 0% (catégorie: exonere)
        - Assurance → Exonérée - 0% (catégorie: exonere)
        - Produits alimentaires → 1% (catégorie: alimentaire) conformément à la LF 2021
        - Autres prestations de services → 10% (catégorie: honoraires)
        - Autres achats de marchandises → 1% (catégorie: marchandises)
    - IMPORTANT: Appliquer 3% UNIQUEMENT pour les personnes physiques au régime réel ET SEULEMENT si l'article 14 de la loi 2020-46 est explicitement applicable.
    - Dans tous les autres cas de prestations de services, appliquer 10% même si "régime réel" est mentionné.
    - Attention aux montants : les acquisitions de marchandises ≤ 1 000 DT sont exonérées (franchise prévue par l'Art 17 de la LF 2021).
    - Toujours préférer une règle spécifique avec référence légale explicite avant d'utiliser une règle générale.
    - TOUJOURS INCLURE LE LIEN SOURCE SI DISPONIBLE.
    - TOUJOURS inclure un paragraphe explicatif citant l'article de loi pertinent.

    **Transaction à analyser**:
    - Service: {service}
    - Montant: {montant} DT
    - Année fiscale: {year}

    **Règles fiscales disponibles**:
    {rules}

    **Format strict de réponse JSON** :
    ```json
    {{
    "categorie": "marchandises, honoraires, loyers, electricite, mobiliers, exonere, etc.",
    "taux": 10.0,
    "ref": "référence légale (ex: Code IRPP-IS / Loi n°2020-46)",
    "doc": "document source",
    "date_app": "date d'application",
    "parag": "extrait de la règle avec citation précise de l'article de loi applicable",
    "lien": "lien source si disponible",
    "benef": "type de bénéficiaire concerné, préciser si personne physique au régime réel"
    }}
    ```
    Réponds uniquement au format JSON sans aucun texte additionnel."""
    
    try:
        # Vérifier si le LLM est défini
        if 'llm' not in globals():
            llm = ChatOllama(model="deepseek-r1:8b")
            print("✅ LLM initialisé pour la classification")
        
        # Invoquer le LLM
        response = llm.invoke(prompt)
        
        # Parser la réponse JSON
        response_text = response.content
        # Extraire le JSON de la réponse (si entouré de ```json...```)
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Essayer de trouver directement un objet JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                
                # Correction des échappements invalides
                json_str = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', json_str)
                
                # Traitement spécial pour les chemins Windows
                json_str = re.sub(r'\\\\([a-zA-Z]:\\\\)', r'\\\\\1', json_str)
            else:
                return None
        
        try:
            # Parser le JSON avec corrections
            classification = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"❌ Erreur JSON après première correction: {e}")
            try:
                # Solution de secours: double échappement de tous les backslashes
                json_str = json_str.replace('\\', '\\\\')
                # Mais éviter de doubler ceux déjà doublés
                json_str = json_str.replace('\\\\\\\\', '\\\\')
                classification = json.loads(json_str)
            except json.JSONDecodeError as e2:
                print(f"❌ Échec du parsing JSON même après corrections: {e2}")
                return None
        
        # Si lien est vide mais qu'on a une catégorie, chercher un lien dans les règles
        if (not classification.get("lien") or classification.get("lien") == "") and classification.get("categorie"):
            cat = classification.get("categorie").lower()
            # Chercher dans toutes les règles un lien correspondant à cette catégorie
            for rule in rules_json:
                rule_type = str(rule.get("type_de_revenu_ou_service", "")).lower()
                if cat in rule_type and rule.get("lien_source"):
                    classification["lien"] = rule.get("lien_source")
                    print(f"✅ Ajout automatique du lien source pour {cat}: {classification['lien']}")
                    break
        
        _classification_cache[cache_key] = classification  # Mettre en cache le résultat
        return classification
    
    except Exception as e:
        print(f"❌ Erreur lors de la classification LLM: {e}")
        return None