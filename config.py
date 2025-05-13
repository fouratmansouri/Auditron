#!/usr/bin/env python
import pathlib

# ─── chemins ──────────────────────────────────────────────
BASE        = pathlib.Path(__file__).parent
XLS_IN      = BASE / "testretenue_flat.xlsx"
JSON_RULES  = BASE / "retenues_final_enrichi.json"
FAISS_DIR   = BASE / "faiss_rules"
DB_PATH     = BASE / "transactions_verif.db"
XLS_OUT     = BASE / "transactions_enrichies.xlsx"

EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
MARCH_FRANCH = 1_000  # franchise marchandises Art 17 LF 2021

# ─── Requêtes SQL prédéfinies ─────────────────────────────────
COMMON_QUERIES = {
    r"(?:3|trois).*mont.*(?:élevé|plus)": "SELECT * FROM transactions ORDER BY montant DESC LIMIT 3;",
    r"(?:5|cinq).*mont.*(?:élevé|plus)": "SELECT * FROM transactions ORDER BY montant DESC LIMIT 5;",
    r".*incorrecte?.*": "SELECT * FROM transactions WHERE statut = 'Incorrecte';",
    r".*correcte?.*": "SELECT * FROM transactions WHERE statut = 'Correcte';",
    r".*référence.*2021": "SELECT DISTINCT source_ref FROM transactions WHERE source_ref LIKE '%2021%';",
    r".*somme.*total.*montant": "SELECT SUM(montant) AS total_montant FROM transactions;",
    r".*additionne.*montant": "SELECT SUM(montant) AS total_montant FROM transactions;",
    r".*compter.*transaction": "SELECT COUNT(*) AS nombre_transactions FROM transactions;",
    r".*lien.*document|.*document.*lien": "SELECT * FROM transactions WHERE lien_source IS NOT NULL AND lien_source != '' AND lien_source != 'Non spécifié';",
    r".*date.*application": "SELECT * FROM transactions WHERE date_application IS NOT NULL AND date_application != '' AND date_application != 'Non spécifié';",
    r".*beneficiaire|.*bénéficiaire": "SELECT * FROM transactions WHERE beneficiaire IS NOT NULL AND beneficiaire != '' AND beneficiaire != 'Non spécifié';",
    r".*paragraphe|.*extrait": "SELECT * FROM transactions WHERE paragraphe IS NOT NULL AND paragraphe != '' AND paragraphe != 'Non spécifié';"
}

# Catégories pour le fallback si LLM échoue
CATS = {
    "marchandises": [
        r"achat", r"consommable", r"alimentaire", r"marchand",
        r"auto.?factur", r"mati[èe]re", r"stock"
    ],
    "honoraires": [
        r"honorair", r"imprimer", r"print", r"consult",
        r"service", r"maintenance", r"location", r"loyer"
    ],
    "electricite": [
        r"electr", r"électr"
    ],
    "imprimerie": [
        r"imprimer", r"imprimerie"
    ],
    "alimentaire": [
        r"alimentaire", r"nourriture", r"consommable.*alimentaire"
    ]
}