# storage/save_parquet.py
import pandas as pd
from pathlib import Path

def save_to_parquet(records, path):
    # Save a list of dictionaries in a Parquet file.
    if not records:
        print(f"⚠️ There is no data to save in {path}.")
        return

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)
    print(f"💾 Saving Parquet: {path}")

def save_index(index_records, index_path):
    # Save or update the global index of processed documents.
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    df_index = pd.DataFrame(index_records)

    # If it already exists, we combine it.
    if Path(index_path).exists():
        df_old = pd.read_parquet(index_path)
        df_index = pd.concat([df_old, df_index], ignore_index=True).drop_duplicates(subset=["file_name"])

    df_index.to_parquet(index_path, index=False)
    print(f"📘 index updated: {index_path}")
