"""
Database operations for tax processing application.
"""
import sqlite3
import pandas as pd
from models.utils import DB_PATH, XLS_OUT

def init_db():
    """Initializes the SQLite database."""
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

def export_to_excel(conn):
    """Exporte les données de la base vers un fichier Excel enrichi"""
    query = """SELECT 
        idx, date, service, montant, raw_taux, taux_applique, 
        statut, taux_attendu, source_ref, document_source,
        date_application, paragraphe, lien_source, beneficiaire, seuil
    FROM transactions"""
    
    df = pd.read_sql_query(query, conn)
    
    # Conversion des dates
    df['date'] = pd.to_datetime(df['date'])
    
    # Formatage des montants et taux
    df['montant'] = df['montant'].round(2)
    df['taux_applique'] = df['taux_applique'].fillna(0).round(2)
    df['taux_attendu'] = df['taux_attendu'].fillna(0).round(2)
    
    # Ajouter des émojis au statut pour une meilleure lisibilité
    df['statut'] = df['statut'].apply(lambda x: f"✅ {x}" if x == "Correcte" else f"❌ {x}")
    
    # Mise en forme avec couleurs conditionnelles
    writer = pd.ExcelWriter(XLS_OUT, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Transactions', index=False)
    
    # Accès à la feuille de calcul pour la mise en forme
    workbook = writer.book
    worksheet = writer.sheets['Transactions']
    
    # Format pour les statuts corrects et incorrects
    format_correct = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
    format_incorrect = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
    
    # Appliquer le formatage conditionnel
    statut_col = df.columns.get_loc('statut') + 1  # +1 car Excel est indexé à partir de 1
    worksheet.conditional_format(1, statut_col, len(df) + 1, statut_col, {
        'type': 'cell',
        'criteria': 'contains',
        'value': "Correcte",
        'format': format_correct
    })
    worksheet.conditional_format(1, statut_col, len(df) + 1, statut_col, {
        'type': 'cell',
        'criteria': 'contains',
        'value': "Incorrecte",
        'format': format_incorrect
    })
    
    # Adapter la largeur des colonnes
    for i, col in enumerate(df.columns):
        max_len = max(df[col].astype(str).apply(len).max(), len(col)) + 2
        worksheet.set_column(i, i, max_len)
    
    # Ajouter des hyperliens pour la colonne lien_source
    lien_col = df.columns.get_loc('lien_source') + 1
    for row_num, link in enumerate(df['lien_source']):
        if pd.notna(link) and link.strip():
            worksheet.write_url(row_num + 1, lien_col, link, string='Voir document')
    
    writer.close()
    print(f"✅ Fichier Excel exporté: {XLS_OUT}")
    # Afficher un aperçu des données exportées
    print("\nAperçu des données exportées:")
    for col in ['date_application', 'beneficiaire', 'paragraphe', 'lien_source']:
        non_empty = df[df[col].notna() & (df[col] != '')].shape[0]
        print(f"  - Colonne {col}: {non_empty}/{len(df)} valeurs non vides")
        
def get_connection():
    """Returns a database connection with row_factory configured."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def execute_query(query, params=None):
    """Executes an SQL query and returns the results."""
    conn = get_connection()
    try:
        if params:
            rows = conn.execute(query, params).fetchall()
        else:
            rows = conn.execute(query).fetchall()
        return rows
    except sqlite3.Error as e:
        print(f"SQL Error: {str(e)}")
        return None
    finally:
        conn.close()

def insert_transaction(idx, date_iso, service, montant, raw_taux, tap, 
                      statut, taux_att, ref, doc, date_app, parag, lien, benef, seuil):
    """Inserts a transaction into the database."""
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
        print(f"Insert Error: {str(e)}")
        return False
    finally:
        conn.close()

def get_all_transactions():
    """Returns all transactions."""
    return execute_query("SELECT * FROM transactions")