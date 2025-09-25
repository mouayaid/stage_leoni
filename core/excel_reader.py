import pandas as pd

def read_excel_file(file_path):
    required_columns = ['Reference', 'Findings', 'Reasons', 'Measures']
    try:
        # Read all rows as data
        df_all = pd.read_excel(file_path, header=None)
        header_row_idx = None

        # Search for the header row
        for i, row in df_all.iterrows():
            if all(col in row.values for col in required_columns):
                header_row_idx = i
                break

        if header_row_idx is None:
            raise ValueError("Required columns not found in any row.")

        # Re-read with the correct header row
        df = pd.read_excel(file_path, header=header_row_idx)
        return df[required_columns]

    except Exception as e:
        print(f"Erreur lors de la lecture du fichier: {e}")
        return None
