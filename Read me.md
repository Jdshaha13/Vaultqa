# VaultQA

A small LangChain RAG service over an Obsidian vault, exposed as a plain
HTTP API — built to be called from Make.com (or curl, or anything else).

Ask a question, get an answer grounded in your own notes, plus which
notes it pulled from.

## How it works

```
your .md notes  --ingest.py-->  local Chroma vector index
                                         |
Make.com --HTTP POST /ask-->  FastAPI (main.py)  --retrieve + ChatAnthropic-->  answer + sources
```

- **Embeddings** run locally via `sentence-transformers` — free, no API
  calls, works offline.
- **Generation** uses Claude via `langchain-anthropic`.
- **Vector store** is Chroma, stored on disk (`chroma_db/` by default).

## Setup (local)

```bash
cd vaultqa
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: set ANTHROPIC_API_KEY, VAULTQA_API_KEY, VAULT_PATH

python ingest.py        # builds the vector index from VAULT_PATH
uvicorn main:app --reload
```

Test it:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_vaultqa_api_key" \
  -d '{"question": "What is my freelance strategy?"}'
```

Point `VAULT_PATH` at your real Obsidian vault folder once you've
confirmed it works against the sample notes.

## Deploying (so Make.com can reach it)

Any host that runs a Python web service works — Railway or Render are
the easiest for a project this size:

1. Push this folder to a GitHub repo.
2. Create a new Railway/Render service from that repo.
3. Set the environment variables from `.env.example` in the platform's
   dashboard (don't commit `.env`).
4. Set the start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. After first deploy, run ingestion once — either bake `python
   ingest.py` into the build step, or call `POST /reindex` after
   deploy (both are wired up in `main.py`).
6. Note the public URL Railway/Render gives you, e.g.
   `https://vaultqa-production.up.railway.app`

## Wiring it into Make.com

1. Add an **HTTP** module → **Make a request**.
2. URL: `https://<your-deployed-url>/ask`
3. Method: `POST`
4. Headers:
   - `Content-Type: application/json`
   - `X-API-Key: <your VAULTQA_API_KEY>`
5. Body (raw JSON):
   ```json
   { "question": "{{ your upstream text/question here }}" }
   ```
6. Parse the response — it returns:
   ```json
   { "answer": "...", "sources": ["Note Title 1", "Note Title 2"] }
   ```
7. Route `answer` and `sources` into whatever comes next — a Notion
   "Create Page", a Slack message, an email, etc.

For the **re-index after new/edited notes** use case: add a second
scenario that fires on a schedule (or on a vault-sync webhook if your
sync tool supports it) and hits `POST /reindex` the same way.

## Extending

- Swap Chroma for a hosted vector DB (Pinecone, Qdrant Cloud) once the
  vault gets large or you need it accessible from multiple deploys.
- Add an agent layer so Claude decides whether to search notes vs.
  answer from general knowledge — swap the LCEL chain in `main.py` for
  a LangChain agent with the retriever as a tool.
- Add per-note metadata (tags, folders) to `ingest.py` and filter
  retrieval by it.
