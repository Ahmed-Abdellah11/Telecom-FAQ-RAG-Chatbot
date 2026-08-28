"""
ingest_api.py
Build ChromaDB using Hugging Face Inference API (NO local PyTorch needed).
Run once: python ingest_api.py
"""
import json
import os

# On some Linux hosts (e.g. Streamlit Community Cloud) the system sqlite3
# is too old for chromadb. If pysqlite3-binary is installed, swap it in.
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from huggingface_hub import InferenceClient
from typing import List

# Config
DATA_PATH = "faq_data.json"
CHROMA_PATH = "chroma_db"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_API_KEY = os.getenv("HF_API_KEY", "")


class HFAPIEmbeddings(Embeddings):
    """Wrapper around HF Inference API for embeddings."""

    def __init__(self, api_key: str, model_name: str):
        self.client = InferenceClient(model=model_name, token=api_key, provider="hf-inference")
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i, text in enumerate(texts):
            print(f"  Embedding document {i+1}/{len(texts)}...")
            try:
                result = self.client.feature_extraction(text)
            except Exception as e:
                raise RuntimeError(f"HF API embedding failed: {e}")
            if hasattr(result, "tolist"):
                result = result.tolist()
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                result = result[0]
            embeddings.append(result)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        result = self.client.feature_extraction(text)
        if hasattr(result, "tolist"):
            result = result.tolist()
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
            result = result[0]
        return result


def load_faq_documents(path: str):
    with open(path, "r", encoding="utf-8") as f:
        faqs = json.load(f)

    documents = []
    for item in faqs:
        content = f"""Category: {item['category']}
Question (EN): {item['question_en']}
Answer (EN): {item['answer_en']}
Question (AR): {item['question_ar']}
Answer (AR): {item['answer_ar']}"""
        documents.append(
            Document(
                page_content=content,
                metadata={"category": item["category"], "source": "WE Synthetic FAQ"},
            )
        )
    return documents


def build_vectorstore():
    if not HF_API_KEY:
        raise ValueError("""HF_API_KEY not set!
1. Get a free token from https://huggingface.co/settings/tokens
2. Run: set HF_API_KEY=hf_your_token_here   (Windows CMD)
   Or:  $env:HF_API_KEY="hf_your_token_here"  (PowerShell)""")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("FAQ data not found at " + DATA_PATH)

    print("Loading documents...")
    documents = load_faq_documents(DATA_PATH)
    print("Loaded " + str(len(documents)) + " FAQ entries.")

    print("Initializing HF Inference API embeddings (" + EMBED_MODEL + ")...")
    embeddings = HFAPIEmbeddings(api_key=HF_API_KEY, model_name=EMBED_MODEL)

    print("Building ChromaDB index (this may take 1-2 minutes on first call)...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )
    print("Done. Vector store saved to ./" + CHROMA_PATH)


if __name__ == "__main__":
    build_vectorstore()
