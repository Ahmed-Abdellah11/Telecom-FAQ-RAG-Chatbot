"""
app_api.py
Streamlit RAG Chatbot using Hugging Face Inference API.
NO local PyTorch / Transformers / GPU needed.
Run with: streamlit run app_api.py
"""
import os
import time

# On some Linux hosts (e.g. Streamlit Community Cloud) the system sqlite3
# is too old for chromadb. If pysqlite3-binary is installed, swap it in.
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from huggingface_hub import InferenceClient
from typing import Any, List, Optional

# Config
CHROMA_PATH = "chroma_db"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct-1M"
# Tried in order if LLM_MODEL isn't currently served by any HF Inference
# Provider. HF's provider lineup shifts over time, so this adds resilience.
LLM_FALLBACK_MODELS = [
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "HuggingFaceH4/zephyr-7b-beta",
]
RETRIEVER_K = 3
MAX_NEW_TOKENS = 256
HF_API_KEY = os.getenv("HF_API_KEY", "")

SYSTEM_PROMPT = (
    "You are a helpful Telecom FAQ assistant for WE (Telecom Egypt).\n"
    "Answer the user question using ONLY the provided context.\n"
    "If the answer is not found in the context, say exactly:\n"
    "English: I do not have information about that in my knowledge base.\n"
    "Arabic: مش لاقي معلومات عن ده في قاعدة البيانات.\n"
    "Always respond in the same language as the user question."
)


class HFAPIEmbeddings(Embeddings):
    """Wrapper around HF Inference API for embeddings."""

    def __init__(self, api_key: str, model_name: str):
        self.client = InferenceClient(model=model_name, token=api_key, provider="hf-inference")
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            result = self.client.feature_extraction(text)
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


class HFAPILLM(LLM):
    """Wrapper around HF Inference API for text generation."""

    api_key: str
    model_name: str
    temperature: float = 0.1
    max_new_tokens: int = 256

    @property
    def _llm_type(self) -> str:
        return "huggingface_api"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        # No fixed provider here: let Hugging Face auto-route to whichever
        # provider (Together, Featherless, Groq, hf-inference, etc.) actually
        # serves this model. Most instruct/chat models today are only
        # reachable this way, not through the legacy hf-inference endpoint.
        models_to_try = [self.model_name] + [
            m for m in LLM_FALLBACK_MODELS if m != self.model_name
        ]
        last_error = None
        for model_name in models_to_try:
            client = InferenceClient(model=model_name, token=self.api_key)
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
                return completion.choices[0].message.content
            except Exception as e:
                last_error = e
                continue
        return f"[HF API Error: {str(last_error)}. Please try again in a moment.]"


@st.cache_resource(show_spinner=False)
def load_vectorstore():
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError("ChromaDB not found at ./" + CHROMA_PATH + ". Please run ingest_api.py first.")
    embeddings = HFAPIEmbeddings(api_key=HF_API_KEY, model_name=EMBED_MODEL)
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)


@st.cache_resource(show_spinner=False)
def load_llm():
    return HFAPILLM(
        api_key=HF_API_KEY,
        model_name=LLM_MODEL,
        temperature=0.1,
        max_new_tokens=MAX_NEW_TOKENS,
    )


def format_context(docs):
    return "\n\n".join([d.page_content for d in docs])


def build_prompt(question: str, context: str):
    return "Context:\n" + context + "\n\nQuestion: " + question + "\nAnswer:"


def answer_question(question: str, retriever, llm):
    docs = retriever.invoke(question)
    context = format_context(docs)
    prompt = build_prompt(question, context)
    raw_answer = llm.invoke(prompt)
    answer = raw_answer.strip()
    return answer, docs


st.set_page_config(page_title="WE Telecom FAQ Chatbot", page_icon="📡")
st.title("📡 WE Telecom FAQ Chatbot")
st.caption(
    "Synthetic Demo Data - Powered by HF Inference API (Qwen2.5-7B-Instruct-1M + paraphrase-multilingual-MiniLM-L12-v2) | "
    "Built with LangChain + ChromaDB + Streamlit | No local GPU needed"
)

if not HF_API_KEY:
    st.error(
        "HF_API_KEY not found!\n\n"
        "1. Get a free token from https://huggingface.co/settings/tokens\n"
        "2. Set it before running:\n"
        "   CMD: set HF_API_KEY=hf_your_token_here\n"
        "   PowerShell: $env:HF_API_KEY=\"hf_your_token_here\""
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("View Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown("**Source " + str(i) + "**")
                    st.text(src)

user_input = st.chat_input("Ask about internet packages, billing, recharge, SIM services...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer via HF API... (First call may take 30-60s while model loads)") as spinner:
            try:
                start_time = time.time()
                vectorstore = load_vectorstore()
                retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
                llm = load_llm()
                answer, docs = answer_question(user_input, retriever, llm)
                elapsed = time.time() - start_time

                st.markdown(answer)
                st.caption(f"Response time: {elapsed:.1f}s")

                sources = [d.page_content for d in docs]
                with st.expander("View Sources"):
                    for i, src in enumerate(sources, 1):
                        st.markdown("**Source " + str(i) + "**")
                        st.text(src)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error("An error occurred: " + str(e))
