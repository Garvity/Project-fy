# Resume Analyzer Pro

Resume Analyzer Pro is a small AI-powered toolkit for analyzing PDF resumes, extracting structured sections, matching resumes to job descriptions, and providing a chat interface that uses retrieval-augmented generation (RAG) over local FAISS vector stores.

This repository contains a Streamlit frontend (`app.py`) and a FastAPI backend (`backend.py`) that communicates with Hugging Face inference APIs and local FAISS vector stores for semantic search.


## Features
- Upload PDF resumes and extract sections (skills, education, experience, projects, contact info).
- Semantic resume ↔ job-description matching with a numeric score and actionable feedback.
- Chat with your resume + job description context using a RAG approach.
- Uses local FAISS vector stores for fast retrieval (pre-built indexes in `vector_store/`).

## Repository layout

- `app.py` — Streamlit frontend (UI + client-side calls to backend endpoints).
- `backend.py` — FastAPI backend implementing endpoints: `/resume_details`, `/resume_matching`, `/chat_with_resume`.
- `requirements.txt` — Python dependencies.
- `test.py` — Small script to sanity-check Hugging Face Inference client connectivity.
- `data/` — example data (e.g., `job_title_des.csv`).
- `vector_store/` — pre-built FAISS indexes (job_faiss, resume_faiss).
- `notebooks/` — experimental notebooks (e.g., `rag_pipeline.ipynb`).

## Quick start

These instructions assume you have Python 3.9+ and `zsh` (macOS). They also assume you want to run both the backend and the frontend locally.

1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) Environment variables

Create a `.env` file in the project root with the following key(s):

```
HF_TOKEN=<your_huggingface_api_token>
```

Note: The backend code uses `huggingface_hub.InferenceClient` with a provider set to `featherless-ai` in this repository. If you have a different provider or want to use the default huggingface endpoint, adjust `backend.py` accordingly.

4) Start the backend (FastAPI)

```bash
# in one terminal
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```

5) Start the frontend (Streamlit)

```bash
# in another terminal
streamlit run app.py
```

The Streamlit app will open in your browser. The frontend posts uploaded PDFs and form data to the backend endpoints which in turn call the Hugging Face Inference API and local FAISS stores.

## Endpoints (backend)

- POST /resume_details — Form fields: `api_key` (string), file upload `file` (PDF). Returns parsed resume sections.
- POST /resume_matching — Form fields: `api_key`, `job_description`, file `file`. Returns match score and feedback.
- POST /chat_with_resume — Form fields: `api_key`, `query`, `job_description`, file `file`. Returns LLM response built using RAG context.

## Vector stores

Pre-built FAISS stores are expected under `vector_store/job_faiss` and `vector_store/resume_faiss`. The backend loads these at startup:

- `FAISS.load_local("vector_store/job_faiss", embeddings, ...)`

If you need to (re)create these vector stores, use the commented code at the bottom of `backend.py` as a starting point: chunk documents, embed them with `HuggingFaceEmbeddings`, then `save_local`.

## Notes on the LLM / Inference

- The code currently uses `mistralai/Mistral-7B-Instruct-v0.2` via `huggingface_hub.InferenceClient`.
- The `HF_TOKEN` environment variable must be set for the client to authenticate.
- The `test.py` script contains a small check that attempts to call the same client — useful for quickly verifying credentials.

## Troubleshooting

- If the Streamlit UI shows errors when calling the backend, check the backend terminal for tracebacks. Common issues:
	- Missing or invalid `HF_TOKEN` — endpoints will fail to generate responses.
	- FAISS index missing or corrupted — ensure `vector_store/*` folders exist.
	- Dependency issues — ensure `pip install -r requirements.txt` succeeded.

- If you get import errors for `langchain_community` or `langchain_huggingface`, install the packages listed in `requirements.txt` or update to compatible versions.

---

