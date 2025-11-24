import os
import re
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, logging
from typing import List

# Import bitsandbytes (requered for load_in_4bit)
try:
    import bitsandbytes
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("⚠️ bitsandbytes is not installed. Load_in_4bit cannot be used.")

class SophisticatedQueryEmbedder:
#A class that encapsulates the logic for generating sophisticated query embeddings
#using the Multi-Query method with a local LLM.
   
    def __init__(self, 
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 llm_rewrite_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        
        self.EMBEDDING_MODEL_NAME = embedding_model_name
        self.LLM_REWRITE_MODEL_NAME = llm_rewrite_model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🚀 Using device for LLM and Embeddings: {self.device}")
        
        self._load_models()

    def _load_models(self):       
        # --- Load Embedding model ---
        print(f"🧠 Loading embedding model: {self.EMBEDDING_MODEL_NAME}...")
        self.embedding_model = SentenceTransformer(self.EMBEDDING_MODEL_NAME)
        print("✅ Embedding model loaded.")

        # --- Load LLM model for rewrite ---
        print(f"💬 Loading LLM model for rewrite: {self.LLM_REWRITE_MODEL_NAME}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.LLM_REWRITE_MODEL_NAME)

            model_llm = AutoModelForCausalLM.from_pretrained(
                self.LLM_REWRITE_MODEL_NAME,
                torch_dtype=torch.float32,
            )

            self.query_rewrite_pipeline = pipeline(
                "text-generation",
                model=model_llm,
                tokenizer=tokenizer,
                torch_dtype=torch.float32
            )
            self.tokenizer = tokenizer 
            print("✅ Pipeline LLM for rewrite has been loaded.")
        except Exception as e:
            print(f"⚠️ Error in loading LLM. Disabling Multi-Query functionality: {e}")
            self.query_rewrite_pipeline = None

    def _generate_multi_perspective_queries(self, original_query: str, num_variants: int) -> List[str]:
        #Use LLM to generate multiples versions (perspectives) for one query.
        if self.query_rewrite_pipeline is None:
             return [original_query]

        messages = [
            {"role": "user", "content": f"""Rewrite next user question in {num_variants} different perspectives to maximize posibilities to find relevant documents. Maintain the original meaning of the question.
            
Original query: {original_query}

Rewrited Perspectives:
1. """}
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)     
        outputs = self.query_rewrite_pipeline(
            prompt,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1
        )
        
        generated_text = outputs[0]["generated_text"][len(prompt):].strip()
        
        variants = [line.strip() for line in generated_text.split('\n') if line.strip() and re.match(r'^\d+\.', line)]
        
        if original_query not in variants:
            variants.insert(0, original_query)
            
        if len(variants) > num_variants + 1:
            variants = variants[:num_variants+1]

        return variants

    def _embed_multi_query(self, queries: List[str]) -> np.ndarray:
        # Generate embeddings for multiple queries and return average.
        if not queries:
            raise ValueError("The query list cannot be empty.")

        embeddings = self.embedding_model.encode(
            queries, 
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        avg_embedding = embeddings.mean(axis=0) 
        return avg_embedding
        
    def get_sophisticated_query_embedding(
        self, 
        original_query: str, 
        num_perspectives: int = 3
    ) -> np.ndarray:
        # Generate a sofisticated query embedding using Multi-Query.        
        expanded_queries = self._generate_multi_perspective_queries(
            original_query, 
            num_variants=num_perspectives
        )
        
        sophisticated_embedding = self._embed_multi_query(expanded_queries)
        
        return sophisticated_embedding