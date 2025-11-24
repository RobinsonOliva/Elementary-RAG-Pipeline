# loaders/xlsx_loader.py
import pandas as pd

def load_xlsx(path):
    sheets = pd.read_excel(path, sheet_name=None)
    text = []
    for name, df in sheets.items():
        text.append(f"--- {name} ---\n{df.to_string(index=False)}")
    return "\n".join(text)
