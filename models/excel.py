#!/usr/bin/env python
import pandas as pd
from config import XLS_IN, XLS_OUT

def read_excel_file():
    """Lit le fichier Excel et retourne un DataFrame pandas."""
    try:
        xl = pd.ExcelFile(XLS_IN)
        df = pd.concat([xl.parse(s) for s in xl.sheet_names], ignore_index=True)
        print(f"✅ Fichier Excel chargé: {len(df)} lignes")
        return df
    except Exception as e:
        print(f"❌ Erreur lors du chargement du fichier Excel: {e}")
        return None

def find_taux_column(df):
    """Identifie la colonne de taux dans le DataFrame."""
    taux_cols = [c for c in df.columns if "taux" in c.lower()]
    if taux_cols:
        TAUX_COL = taux_cols[0]
    else:
        TAUX_COL = next((c for c in df.columns
                       if df[c].astype(str).str.contains(r"%|rs", case=False, na=False).any()), None)
    
    if not TAUX_COL:
        print("❌ Impossible de trouver une colonne de taux dans le fichier Excel")
        return None
    
    print(f"✅ Colonne de taux identifiée: {TAUX_COL}")
    return TAUX_COL

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