import os
import faiss
import numpy as np
import pandas as pd

# Base path
BASE_DIR = "data/processed"
EMB_DIR = os.path.join(BASE_DIR, "embeddings")
INDEX_PATH = os.path.join(BASE_DIR, "index.faiss")
META_PATH = os.path.join(BASE_DIR, "metadata.parquet")

def construir_indice_vectorial():
    # Load embeddings from Parquet and create FAISS index.
    all_embeddings = []
    all_texts = []

    parquet_files = [f for f in os.listdir(EMB_DIR) if f.endswith("_emb.parquet")]

    for file in parquet_files:
        df = pd.read_parquet(os.path.join(EMB_DIR, file))
        df = df.dropna(subset=["embedding"])

        texto_col = next((c for c in df.columns if c.lower() in ["texto", "text", "content"]), None)
        if texto_col is None:
            raise ValueError(f"❌ The file {file} does not contain a valid text column: {df.columns.tolist()}")

        embeddings = np.array(df["embedding"].tolist()).astype("float32")
        all_embeddings.append(embeddings)

        for _, row in df.iterrows():
            all_texts.append({
                "texto": row[texto_col],
                "fuente": file
            })

    # Combine all embeddings in only one array
    all_embeddings = np.vstack(all_embeddings)

    # Create FAISS index
    index = faiss.IndexFlatL2(all_embeddings.shape[1])
    index.add(all_embeddings)

    # Save index and metadata
    faiss.write_index(index, INDEX_PATH)
    pd.DataFrame(all_texts).to_parquet(META_PATH, index=False)

    print(f"✅ FAISS index saved in: {INDEX_PATH}")
    print(f"✅ Metadata saved in: {META_PATH}")
    print(f"Total of embeddings indexed: {len(all_texts)}")



