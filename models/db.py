"""
Database operations for tax processing application.
"""
import sqlite3
import pandas as pd
import datetime
import xlsxwriter
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
    """Exporte les données de la base vers un fichier Excel."""
    try:
        # Création d'un classeur Excel
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transactions_classifiees_{now}.xlsx"
        workbook = xlsxwriter.Workbook(filename)
        worksheet = workbook.add_worksheet("Transactions")
        
        # Styles pour le classeur
        header_format = workbook.add_format({
            'bold': True, 
            'bg_color': '#C0C0C0',
            'border': 1
        })
        
        # Récupération des données
        cursor = conn.cursor()
        # Ajuster cette requête en fonction de votre schéma de base de données
        cursor.execute("""
            SELECT idx, service, montant, date, 
                   'N/A', taux_attendu, source_ref, document_source, 
                   date_application, paragraphe, lien_source, beneficiaire,
                   statut
            FROM transactions
            ORDER BY idx ASC
        """)
        
        # En-têtes
        headers = [
            "ID", "Service", "Montant", "Date", "Catégorie", "Taux", 
            "Référence", "Document", "Date app", "Explication", "Lien", "Bénéficiaire", "Statut"
        ]
        
        for col_num, header in enumerate(headers):
            worksheet.write(0, col_num, header, header_format)
        
        # Identifier l'index de la colonne Lien
        lien_col = headers.index("Lien")
        
        # Données
        rows = cursor.fetchall()
        for row_num, row in enumerate(rows):
            for col_num, cell_value in enumerate(row):
                # Formatage spécial pour certaines colonnes
                if col_num == headers.index("Montant"):
                    # Format monétaire pour les montants
                    if cell_value is not None:
                        worksheet.write_number(row_num + 1, col_num, float(cell_value), 
                                            workbook.add_format({'num_format': '# ##0.00 "DT"'}))
                    else:
                        worksheet.write(row_num + 1, col_num, 0)
                elif col_num == headers.index("Taux"):
                    # Format pourcentage pour les taux
                    if cell_value is not None and str(cell_value).strip() != "":
                        try:
                            taux_val = float(cell_value)
                            worksheet.write_number(row_num + 1, col_num, taux_val / 100, 
                                                workbook.add_format({'num_format': '0.0%'}))
                        except (ValueError, TypeError):
                            worksheet.write(row_num + 1, col_num, str(cell_value))
                    else:
                        worksheet.write(row_num + 1, col_num, "N/A")
                elif col_num == lien_col:
                    # Correction pour les liens
                    link = str(cell_value).strip() if cell_value else ""
                    
                    # Vérification qu'il s'agit d'un lien valide
                    is_valid_link = (link and 
                                    link != "#" and 
                                    link != "Non spécifié" and
                                    ("http://" in link or "https://" in link))
                    
                    if is_valid_link:
                        try:
                            # Écrire comme URL avec texte personnalisé
                            worksheet.write_url(row_num + 1, col_num, link, string='Voir document')
                        except ValueError as e:
                            print(f"Erreur lors de l'écriture du lien '{link}': {e}")
                            # Écrire comme texte normal en cas d'erreur
                            worksheet.write(row_num + 1, col_num, link)
                    else:
                        # Écrire comme texte normal si ce n'est pas un lien valide
                        if link == "#" or not link:
                            worksheet.write(row_num + 1, col_num, "")
                        else:
                            worksheet.write(row_num + 1, col_num, link)
                else:
                    # Écriture normale pour les autres cellules
                    worksheet.write(row_num + 1, col_num, cell_value)
        
        # Ajustement automatique de la largeur des colonnes
        for i, header in enumerate(headers):
            # Calculer la largeur maximale basée sur le contenu
            max_width = len(header)
            for row_num in range(len(rows)):
                cell_value = str(rows[row_num][i]) if i < len(rows[row_num]) else ""
                max_width = max(max_width, min(len(cell_value), 50))  # Limiter à 50 caractères max
            
            # Ajouter une marge
            worksheet.set_column(i, i, max_width + 2)
        
        # Ajuster spécifiquement certaines colonnes
        worksheet.set_column(headers.index("Service"), headers.index("Service"), 25)  # Service
        worksheet.set_column(headers.index("Explication"), headers.index("Explication"), 40)  # Explication
        worksheet.set_column(lien_col, lien_col, 15)  # Lien
        
        workbook.close()
        print(f"✅ Export Excel terminé: {filename}")
        return filename
        
    except Exception as e:
        print(f"❌ Erreur lors de l'export Excel: {e}")
        import traceback
        traceback.print_exc()
        return None

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