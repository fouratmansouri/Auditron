# filepath: c:\Users\friti\Downloads\projfinance\models\tax_rules.py
"""
Gestion des règles fiscales du projet.
"""
import json
from models.utils import BASE, JSON_RULES

def load_tax_rules():
    """Charge les règles fiscales du fichier JSON."""
    try:
        with open(JSON_RULES, encoding="utf-8") as f:
            tax_rules = json.load(f)
            
        # Transformer le format pour faciliter l'accès par catégorie
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