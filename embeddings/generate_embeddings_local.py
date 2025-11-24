import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Path settings
BASE_DIR = "data/processed"
DOCS_DIR = os.path.join(BASE_DIR, "docs")
EMB_DIR = os.path.join(BASE_DIR, "embeddings")
os.makedirs(EMB_DIR, exist_ok=True)

# Loading free model
print("🧠 Loading a local model all-MiniLM-L6-v2...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("✅ Model loaded successfully.\n")

# Main function for embeddings
def generate_embeddings_for_file(parquet_path, output_path):
    # Read Parquet, generate embeddings save as another Parquet.
    df = pd.read_parquet(parquet_path)

    # Detects column of text
    texto_col = None
    for col in ["texto", "content", "text"]:
        if col in df.columns:
            texto_col = col
            break
    if texto_col is None:
        raise ValueError(f"❌ File {parquet_path} does not have a valid text column ('texto', 'content' o 'text').")

    textos = df[texto_col].astype(str).tolist()

    # Generate embeddings
    print(f"🚀 Generating embeddings to {os.path.basename(parquet_path)}...")
    embeddings = model.encode(textos, show_progress_bar=True, batch_size=32, convert_to_numpy=True)

    # Save results
    df["embedding"] = embeddings.tolist()
    df.to_parquet(output_path, index=False)
    print(f"✅ Embeddings saved on {output_path}\n")

# Procesar todos los archivos
def process_all_parquets():
    parquet_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".parquet")]
    if not parquet_files:
        print("⚠️ There is not files .parquet in data/processed/docs/")
        return

    for file in parquet_files:
        input_path = os.path.join(DOCS_DIR, file)
        output_path = os.path.join(EMB_DIR, file.replace(".parquet", "_emb.parquet"))
        generate_embeddings_for_file(input_path, output_path)

    print("🎉 All embeddings were generated successfully.")


