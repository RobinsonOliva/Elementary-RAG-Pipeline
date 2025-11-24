from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
os.environ["TRANSFORMERS_NO_ACCELERATE"] = "1"

class RAGGenerator:
    def __init__(self, llm_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.LLM_MODEL_NAME = llm_model_name
        self.device = "cpu"
        self.tokenizer = None
        self.generation_pipeline = None
        self._load_llm_model()

    def _load_llm_model(self):
        print(f"💬 Loading LLM model in CPU: {self.LLM_MODEL_NAME}...")
    
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.LLM_MODEL_NAME)
            model_llm = AutoModelForCausalLM.from_pretrained(
            self.LLM_MODEL_NAME,
            torch_dtype=torch.float32,
            )

            self.generation_pipeline = pipeline(
                "text-generation",
                model=model_llm,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float32,
            )

            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error: {e}")

    def _create_prompt(self, question: str, retrieved_chunks: List[Dict[str, Any]]):
        context_parts, sources = [], []
        for i, chunk in enumerate(retrieved_chunks):
            context_parts.append(f"[Doc {i+1}: {chunk['fuente']}]\n{chunk['texto']}")
            sources.append(chunk['fuente'])

        context = "\n---\n".join(context_parts)
        unique_sources = ", ".join(sorted(set(sources)))

        system_instruction = (
            "You're an expert RAG assistant. Use only CONTEXT to respond."
            "If you don't have the information, respond: "
            "'Sorry, I don't have any information about that in my documents.'"
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nResponse:"},
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt, unique_sources

    def generate_answer(self, question: str, retrieved_chunks: List[Dict[str, Any]]):
        if self.generation_pipeline is None:
            return {"answer": "Error: model not loaded.", "sources": ""}
        if not retrieved_chunks:
            return {"answer": "No relevant documents were recovered.", "sources": ""}

        prompt, sources = self._create_prompt(question, retrieved_chunks)
        outputs = self.generation_pipeline(
            prompt,
            max_new_tokens=400,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            return_full_text=False,
        )
        return {"answer": outputs[0]["generated_text"].strip(), "sources": sources}

