import pandas as pd
def read_excel_file(file_path):
    try:
        df = pd.read_excel(file_path)
        required_columns = ['Reference', 'Findings', 'Reasons', 'Measures']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in the file.")

        return df[required_columns]

    except Exception as e:
        print(f"Erreur lors de la lecture du fichier: {e}")
        return None
