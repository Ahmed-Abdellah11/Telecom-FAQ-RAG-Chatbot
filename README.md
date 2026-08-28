# 📡 WE Telecom FAQ Chatbot — RAG with Hugging Face Inference API

A Retrieval-Augmented Generation (RAG) chatbot that answers telecom customer-support questions in **Arabic and English**, built entirely on the **Hugging Face Inference API** — no local GPU, no PyTorch/Transformers install, no Docker.

## Overview

The bot retrieves relevant FAQ entries from a local knowledge base and uses an LLM to generate a grounded answer — only from the retrieved context, in the same language as the question. If the answer isn't in the knowledge base, it says so explicitly instead of guessing.

## Demo

| Question | Behavior |
|---|---|
| "ازاي اشحن رصيد WE؟" | Retrieves the Recharge FAQ entry, answers in Arabic |
| "What are the available WE internet packages?" | Retrieves the Internet Packages entry, answers in English |
| A question not covered by the FAQ | Replies with an explicit "not found" message instead of hallucinating |

## Stack

- **UI:** Streamlit chat interface
- **Orchestration:** LangChain
- **Vector store:** ChromaDB (persisted locally in `chroma_db/`)
- **Embeddings:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` via HF Inference API
- **LLM:** `Qwen/Qwen2.5-7B-Instruct-1M`, with automatic fallback to `Qwen2.5-Coder-32B-Instruct`, `Llama-3.1-8B-Instruct`, or `zephyr-7b-beta` if the primary model isn't currently served
- **Knowledge base:** synthetic bilingual telecom FAQ (`faq_data.json`) — internet packages, recharge, billing, technical issues, SIM services

Everything runs through Hugging Face's hosted Inference API, so there's no local model download and no GPU requirement — it runs fine on something as small as Streamlit Community Cloud.

## Project structure

```
├── app_api.py            # Streamlit chatbot app (run this)
├── ingest_api.py          # Builds the ChromaDB vector store from faq_data.json (run once)
├── debug_api.py           # Standalone diagnostic script — tests each step outside Streamlit
├── faq_data.json          # Synthetic bilingual (AR/EN) telecom FAQ knowledge base
└── requirements_api.txt   # Lightweight dependencies (no PyTorch/Transformers)
```

## Setup

### 1. Get a free Hugging Face API token
Create one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 2. Install dependencies
```bash
pip install -r requirements_api.txt
```

### 3. Set your API key
```bash
# Windows CMD
set HF_API_KEY=hf_your_token_here

# PowerShell
$env:HF_API_KEY="hf_your_token_here"

# macOS / Linux
export HF_API_KEY=hf_your_token_here
```

### 4. Build the vector store (run once)
```bash
python ingest_api.py
```
This embeds every FAQ entry via the HF Inference API and saves the index to `chroma_db/`.

### 5. Run the app
```bash
streamlit run app_api.py
```

## Troubleshooting

If the app crashes or behaves unexpectedly, run the standalone diagnostic script first — it walks through every step (API key, ChromaDB, embeddings, retrieval, LLM call) outside of Streamlit and prints exactly where it fails:

```bash
python debug_api.py
```

Common issues it catches:
- Missing or invalid `HF_API_KEY`
- `chroma_db/` not built yet (run `ingest_api.py` first)
- Embedding dimension mismatch between what's stored and what's freshly computed (usually means the embedding model changed — delete `chroma_db/` and re-run `ingest_api.py`)
- The primary LLM model not currently served by any HF Inference provider (the app automatically tries fallback models)

## How it works

1. **Ingest** (`ingest_api.py`): each FAQ entry is formatted with its category, English Q&A, and Arabic Q&A, embedded via the HF Inference API, and stored in ChromaDB.
2. **Retrieve**: on each user question, the top 3 most similar FAQ entries are pulled from ChromaDB.
3. **Generate**: the retrieved entries are passed as context to the LLM along with a system prompt instructing it to answer only from that context, in the user's language, and to say clearly when the answer isn't available.
4. **Display**: the Streamlit UI shows the answer, response time, and the retrieved source entries in an expandable panel.

## Notes

- This is a portfolio/demo project — the FAQ data is synthetic, not official WE (Telecom Egypt) information.
- The `pysqlite3` swap at the top of each script exists because some Linux hosts (e.g. Streamlit Community Cloud) ship a system `sqlite3` too old for ChromaDB.

## Author

Ahmed Abdellah — AI Engineer
