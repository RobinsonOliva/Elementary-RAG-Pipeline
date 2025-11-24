# Archivo: vectorIndex/retrieval.py (o similar)

import os
import faiss
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# Configuration (MUST MATCH vectorIndex/build_index.py)
BASE_DIR = "data/processed"
INDEX_PATH = os.path.join(BASE_DIR, "index.faiss")
META_PATH = os.path.join(BASE_DIR, "metadata.parquet")

class VectorRetriever:
    # Class to handle the FAISS index load and the search for relevant chunks.

    def __init__(self):
        self.index = None
        self.metadata_df = None
        self._load_index()

    def _load_index(self):
        # Load FAISS index and metadata if it found
        if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
            print(f"⚠️ Index files not found. Run build_vector_index first.")
            self.index = None
            self.metadata_df = None
            return

        print(f"🧠 Loading FAISS index from: {INDEX_PATH}")
        self.index = faiss.read_index(INDEX_PATH)
        
        print(f"📚 Loading metadata from: {META_PATH}")
        self.metadata_df = pd.read_parquet(META_PATH)
        
        # We added the 'id' column which matches the FAISS index
        self.metadata_df['faiss_id'] = range(len(self.metadata_df))
        print(f"✅ Index and metadata loaded. Total of documents: {len(self.metadata_df)}")

    def retrieve_chunks(self, 
                        query_vector: np.ndarray, 
                        k: int = 4) -> List[Dict[str, Any]]:
        """
        Search for the k chunks closest to the query vector in the FAISS index.

        Args:
            query_vector (np.ndarray): The query vector (ex. results from SophisticatedQueryEmbedder).
            k (int): Number of most relevants chunks to recover.

        Returns:
            List[Dict[str, Any]]: chunks list (texto, fuente, score) ordered by relevance.
        """
        if self.index is None or self.metadata_df is None:
            return []

        # FAISS requires that the query vector be a 2D (1, D) array of type float32
        query_vector = np.array(query_vector).astype('float32').reshape(1, -1)
        
        # 1. Perform the search in FAISS
        # D: Distances (scores), I: Chunk indices in the matrix (correspond to faiss_id)
        D, I = self.index.search(query_vector, k)
        
        retrieved_chunks = []
        
        # 2. Iterate on the FAISS results
        for rank, (score, doc_id) in enumerate(zip(D[0], I[0])):
            # Get the metadata and text of the chunk using the ID
            if doc_id < 0: # Ignore invalid IDs if FAISS returns -1 (possible in indexes that do not use all slots)
                continue
                
            metadata = self.metadata_df.iloc[doc_id]
            
            chunk_data = {
                "chunk_id_faiss": int(doc_id),
                "score_l2": float(score), # L2 distance (lower = closer)
                "rank": rank + 1,
                "texto": metadata['texto'],
                "fuente": metadata['fuente']
            }
            retrieved_chunks.append(chunk_data)
            
        return retrieved_chunks
