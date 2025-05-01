"""
Tax classification using LLM and rule-based approaches.
"""
import json, re
from langchain_community.chat_models import ChatOllama
from models.utils import JSON_RULES

# Cache for LLM classifications
_classification_cache = {}

def load_tax_rules():
    """Loads tax rules from JSON file."""
    try:
        with open(JSON_RULES, encoding="utf-8") as f:
            tax_rules = json.load(f)
            
        # Transform the format for easier access by category
        categorized_rules = {}
        for rule in tax_rules:
            category = rule.get("categorie", "").lower()
            if category and category not in categorized_rules:
                categorized_rules[category] = {
                    "taux": rule.get("taux_num", 0),
                    "ref": rule.get("référence_légale", "Non spécifié"),
                    "doc": rule.get("source_file", "Non spécifié"),
                    "date_app": rule.get("date_application", "Non spécifié"),
                    "parag": rule.get("paragraphe", "Non spécifié"),
                    "lien": rule.get("lien_source", ""),
                    "benef": rule.get("bénéficiaire", "Non spécifié"),
                    "seuil": rule.get("seuil", "Non spécifié")
                }
                
        print(f"✓ Règles fiscales chargées: {len(categorized_rules)} catégories configurées")
        return categorized_rules
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement des règles fiscales: {e}")
        return {}  # Dictionnaire vide en cas d'erreur

def classify_transaction_with_llm(service, montant, year, rules_json):
    """Classification d'une transaction avec LLM en utilisant le fichier JSON de règles."""
    
    # Cache key to avoid re-querying the LLM for similar cases
    cache_key = f"{service}|{montant}|{year}"
    if cache_key in _classification_cache:
        return _classification_cache[cache_key]
    
    # Direct detection for electricity (before even consulting the cache)
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
    
    # Limit the number of rules to avoid a prompt that's too long
    rules_sample = []
    if len(rules_json) > 5:
        # First take rules that match the service
        service_lower = service.lower()
        matching_rules = [r for r in rules_json 
                         if service_lower in str(r.get("type_de_revenu_ou_service", "")).lower()]
        
        # If we found matching rules, use them
        if matching_rules:
            rules_sample = matching_rules[:5]
        else:
            # Otherwise just take the first 5 rules
            rules_sample = rules_json[:5]
    else:
        rules_sample = rules_json
    
    rules = json.dumps(rules_sample, ensure_ascii=False)
    
    # Build the prompt for the LLM with the specific rules
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
        # Check if the LLM is defined
        llm = ChatOllama(model="deepseek-r1:8b")
        print("✅ LLM initialisé pour la classification")
        
        # Invoke the LLM
        response = llm.invoke(prompt)
        
        # Parse the JSON response
        response_text = response.content
        # Extract JSON from the response (if wrapped in ```json...```)
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find a JSON object directly
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                
                # Fix invalid escape sequences
                json_str = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', json_str)
                
                # Special handling for Windows paths
                json_str = re.sub(r'\\\\([a-zA-Z]:\\\\)', r'\\\\\1', json_str)
            else:
                return None
        
        try:
            # Parse JSON with corrections
            classification = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"❌ JSON error after first correction: {e}")
            try:
                # Fallback: double-escape all backslashes
                json_str = json_str.replace('\\', '\\\\')
                # But avoid doubling those already doubled
                json_str = json_str.replace('\\\\\\\\', '\\\\')
                classification = json.loads(json_str)
            except json.JSONDecodeError as e2:
                print(f"❌ JSON parsing failed even after corrections: {e2}")
                return None
        
        # If the link is empty but we have a category, look for a link in the rules
        if (not classification.get("lien") or classification.get("lien") == "") and classification.get("categorie"):
            cat = classification.get("categorie").lower()
            # Look for a link in all rules that match this category
            for rule in rules_json:
                rule_type = str(rule.get("type_de_revenu_ou_service", "")).lower()
                if cat in rule_type and rule.get("lien_source"):
                    classification["lien"] = rule.get("lien_source")
                    print(f"✅ Ajout automatique du lien source pour {cat}: {classification['lien']}")
                    break
        
        _classification_cache[cache_key] = classification  # Cache the result
        return classification
    
    except Exception as e:
        print(f"❌ Erreur lors de la classification LLM: {e}")
        return None