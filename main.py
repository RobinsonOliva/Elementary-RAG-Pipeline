# main.py
from pathlib import Path
from utils.file_finder import find_files
from utils.text_cleaner import TextCleaner
from utils.chunker import chunk_text
from loaders import txt_loader, docx_loader, pdf_loader, xlsx_loader
from storage.save_parquet import save_to_parquet, save_index
from embeddings.generate_embeddings_local import process_all_parquets
from vectorIndex.build_index import construir_indice_vectorial
from vectorIndex.retrieval import VectorRetriever
from llmGeneration.generation import RAGGenerator

import time

import numpy as np
from multiQueryEmbedding.query_embedding import SophisticatedQueryEmbedder 

# Loaders configuration
loaders = {
    ".txt": txt_loader.load_txt,
    ".docx": docx_loader.load_docx,
    ".pdf": pdf_loader.load_pdf,
    ".xlsx": xlsx_loader.load_xlsx,
}

cleaner = TextCleaner(target_lang="es")

# ==================================
# Main function
# ==================================
def import_all_docs(base_path="data/raw_docs", output_dir="data/processed/docs"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_records = []

    for path_str in find_files(base_path, extensions=loaders.keys()):
        path = Path(path_str)
        ext = next(ext for ext in loaders if path.suffix.lower() == ext)
        loader_fn = loaders[ext]

        try:
            text = loader_fn(path)
            if not text:
                print(f"⚠️ Empty or unreadable file: {path.name}")
                continue

            # Advanced cleaning
            cleaned = cleaner.clean_text(text)

            # Chunking
            chunks = chunk_text(cleaned)
            if not chunks:
                print(f"⚠️ No chunks were generated for: {path.name}")
                continue

            # Data structure
            file_chunks = [
                {
                    "source": path.name,
                    "chunk_id": f"{path.stem}_{i}",
                    "content": chunk
                }
                for i, chunk in enumerate(chunks)
            ]

            # Save Parquet for document
            parquet_path = output_dir / f"{path.stem}.parquet"
            save_to_parquet(file_chunks, parquet_path)

            # Metadata for global index
            index_records.append({
                "file_name": path.name,
                "parquet_path": str(parquet_path),
                "n_chunks": len(chunks),
                "extension": ext,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })

            print(f"✅ Processed: {path.name} ({len(chunks)} chunks)")

        except Exception as e:
            print(f"⚠️ Error processing {path}: {e}")

    # Saved global index
    if index_records:
        save_index(index_records, "data/processed/index.parquet")
        print(f"\n📘 Global index updated: data/processed/index.parquet")

        # Generate embeddings after Parquet saved
        print("\n🧠 Generating embeddings ...")
        process_all_parquets()
        construir_indice_vectorial()

    else:
        print("⚠️ No valid records were generated.")

    # We built the example query embedding with multi-query
    print("--- 🧠 Initializing SophisticatedQueryEmbedder (Loading Models) ---")
    try:
        LLM_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
        query_embedder = SophisticatedQueryEmbedder(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", 
        llm_rewrite_model_name="Qwen/Qwen2.5-0.5B-Instruct"
        )

        # Retriever
        retriever = VectorRetriever()
        # Initializing RAG generator
        rag_generator = RAGGenerator(llm_model_name=LLM_NAME)        
        
        if retriever.index is None or rag_generator.generation_pipeline is None:
            print("❌ The RAG query cannot be continued. Missing components.")
            return
           
        print("--- ✅ Complete Initialization ---\n")
    except Exception as e:
        # Error handling if models or libraries (such as bitsandbytes) fail
        print(f"\n❌ Fatal error during model initialization: {e}")
        exit()

    #PROCESS QUERY (Repeated Calls)
    user_question_1 = "¿dime de que trata el Algoritmo de compresión sin pérdidas bajo el enfoque de Huffman y RLE?"
    num_perspectives = 3 # Generate 3 variants in addition to the original
    num_k_chunks = 4 # We define how many chunks we want to retrieve
    print(f"\n--- 🔍 Processing query 1: '{user_question_1}' ---")
    # Calling to main function
    sophisticated_query_vector_1 = query_embedder.get_sophisticated_query_embedding(
        user_question_1,
        num_perspectives=num_perspectives
    )
    print("\n--- Results of Query 1 ---")
    print(f"Embedding Type: {type(sophisticated_query_vector_1)}") # Must be numpy.ndarray
    print(f"Dimension: {sophisticated_query_vector_1.shape}")         # Must be (384,)
    print(f"Vector (First 5 elements): {sophisticated_query_vector_1[:5]}\n")

    # b. Perform Retrieval (Vector Search)
    retrieved_chunks_1 = retriever.retrieve_chunks(
        sophisticated_query_vector_1,
        k=num_k_chunks
    )

    # c. Show Retrieval Results
    print("\n--- ✅ Results of Retrieval 1 ---")
    if retrieved_chunks_1:
        for chunk in retrieved_chunks_1:
            score_formatted = f"{chunk['score_l2']:.4f}"
            print(f"[{chunk['rank']}] Score L2: {score_formatted} | Fuente: {chunk['fuente']}")
            print(f"  Texto: {chunk['texto'][:100]}...\n")
    else:
        print("⚠️ No relevant chunks were found for Query 1.")

    # d. Response Generation
    print("✨ Synthesizing response with LLM...")
    response_1 = rag_generator.generate_answer(user_question_1, retrieved_chunks_1)

    # e. Show Final Results
    print("\n--- 🌟 Final Response RAG 1 ---")
    print(f"Question: {user_question_1}")
    print("---------------------------------")
    print(response_1['answer'])
    print("---------------------------------")
    print(f"Sources used: {response_1['sources']}")





    # Second Query (Quick, since the templates are loaded)
    user_question_2 = "¿Cómo se implementa el algoritmo anterior?"
    print(f"--- 🔍 Processing Query 2: '{user_question_2}' ---")
    # It's called the same function, but the models don't reload.
    sophisticated_query_vector_2 = query_embedder.get_sophisticated_query_embedding(
        user_question_2,
        num_perspectives=2 # We can use a different number of perspectives
    )
    print("\n--- Results of Query 2 ---")
    print(f"Dimension: {sophisticated_query_vector_2.shape}")
    print(f"Vector (First 5 elements): {sophisticated_query_vector_2[:5]}")
    # These are the vectors you would pass to your FAISS index for the search.
    # b. Perform Retrieval (Vector Search)
    retrieved_chunks_2 = retriever.retrieve_chunks(
        sophisticated_query_vector_2,
        k=num_k_chunks
    )
    
    # c. Show Retrieval Results
    print("\n--- ✅ Results of Retrieval 2 ---")
    if retrieved_chunks_2:
        for chunk in retrieved_chunks_2:
            score_formatted = f"{chunk['score_l2']:.4f}"
            print(f"[{chunk['rank']}] Score L2: {score_formatted} | Fuente: {chunk['fuente']}")
            print(f"  Texto: {chunk['texto'][:100]}...\n")
    else:
        print("⚠️ No relevant chunks were found for Query 2.")

    # d. Response Generation
    print("\n✨ Synthesizing response with LLM...")
    response_2 = rag_generator.generate_answer(user_question_2, retrieved_chunks_2)

    # e. Show Final Results
    print("\n--- 🌟 Final Response RAG 2 ---")
    print(f"Question: {user_question_2}")
    print("---------------------------------")
    print(response_2['answer'])
    print("---------------------------------")
    print(f"Sources used: {response_2['sources']}")


# ==================================
# Entry point
# ==================================
if __name__ == "__main__":
    import_all_docs()



