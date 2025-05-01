"""
Utility functions and constants for tax processing.
"""
import re, math, unicodedata, pathlib
from datetime import datetime

# ─── chemins ──────────────────────────────────────────────
BASE        = pathlib.Path(__file__).parent.parent
XLS_IN      = BASE / "testretenue_flat_augmented.xlsx"
JSON_RULES  = BASE / "retenues_final_enrichi.json"
FAISS_DIR   = BASE / "faiss_rules"
DB_PATH     = BASE / "transactions_verif.db"
XLS_OUT     = BASE / "transactions_enrichies.xlsx"

EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
MARCH_FRANCH = 1_000  # franchise marchandises Art 17 LF 2021

# ─── utilitaires ───────────────────────────────────────────
RE_RS  = re.compile(r"rs[-\s]?(\d+[\,\.]?\d*)", re.I)
RE_NUM = re.compile(r"(\d+[\,\.]?\d*)")
def rs_or_pct(x) -> float:
    """Extract a numeric rate from a string."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return math.nan
    s = str(x)
    if m := RE_RS.search(s):
        # Ne pas multiplier par 10 - extraire directement la valeur numérique
        return float(m.group(1).replace(',', '.'))
    if m := RE_NUM.search(s):
        return float(m.group(1).replace(',', '.'))
    return math.nan

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

def guess_cat(lbl: str) -> str:
    """Classification basique des transactions selon leur libellé (fallback si LLM échoue)."""
    s = unicodedata.normalize("NFKD", lbl.lower()).encode("ascii", "ignore").decode()
    
    # Classifications prioritaires pour les cas spécifiques
    if re.search(r"electr|électr", s):
        return "electricite"
    if re.search(r"imprimer", s) or "imprimerie" in s:
        return "imprimerie"
    if "consommable produit alimentaire" in s:
        return "alimentaire"
    
    # Classification générale
    for cat, pats in CATS.items():
        if any(re.search(p, s) for p in pats):
            return cat
    
    return "honoraires"  # Valeur par défaut

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

def extract_clean_sql(text):
    """Extrait une requête SQL du texte et nettoie tout contenu superflu."""
    # Supprimer les balises think si présentes
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Extraire la requête SQL entre backticks
    sql_match = re.search(r'```(?:sql)?\s*(SELECT\s+.*?)```', text, re.DOTALL | re.IGNORECASE)
    if sql_match:
        sql = sql_match.group(1).strip()
    else:
        # Chercher directement une requête SQL
        sql_match = re.search(r'(SELECT\s+.*?FROM\s+.*?)(?:;|\n|$)', text, re.DOTALL | re.IGNORECASE)
        if sql_match:
            sql = sql_match.group(1).strip()
        else:
            # Dernier recours: prendre le premier SELECT
            if "SELECT" in text.upper():
                parts = text.upper().split("SELECT", 1)
                if len(parts) > 1:
                    sql = "SELECT" + parts[1].strip()
                    # Limiter jusqu'au prochain point-virgule ou nouvelle ligne
                    end_match = re.search(r'(.*?)(?:;|\n|$)', sql, re.DOTALL)
                    if end_match:
                        sql = end_match.group(1).strip()
                else:
                    return None
            else:
                return None
    
    # Nettoyer et valider la requête SQL
    # Ne pas forcer SELECT * pour les requêtes d'agrégation qui utilisent SUM, COUNT, etc.
    if not re.search(r"(SUM|AVG|COUNT|MIN|MAX)\s*\(", sql, re.IGNORECASE):
        # Pour les requêtes non-agrégation, forcer SELECT *
        if re.match(r"SELECT\s+(?!\*)", sql, re.IGNORECASE):
            sql = re.sub(r"SELECT\s+.*?\s+FROM", "SELECT * FROM", sql, flags=re.IGNORECASE)
            print(f"Requête modifiée pour SELECT *: {sql}")
    
    # S'assurer que la requête contient FROM
    if "FROM" not in sql.upper():
        sql += " FROM transactions"
    
    # S'assurer que la requête se termine par un point-virgule
    if not sql.endswith(';'):
        sql += ";"
    
    # Nettoyer d'éventuels textes supplémentaires après le premier point-virgule
    sql = sql.split(';')[0] + ";"
    
    return sql