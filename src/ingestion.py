import pandas as pd

def sheet_to_text(path, sheet_name):
    df = pd.read_excel(path, sheet_name=sheet_name, header=None)
    return df.to_string(index=False, na_rep="")