# gemini-embedding-2 + MongoDB Vector Search demo

Educational demo: upload any file (text, image, PDF, audio, video, ZIP, RAR) and search them semantically using `gemini-embedding-2` embeddings stored in MongoDB Atlas Local.

## Prerequisites

- Python 3.12+
- Docker (for MongoDB Atlas Local)
- ffmpeg (`apt install ffmpeg` / `brew install ffmpeg` / `winget install ffmpeg`)
- unrar (`apt install unrar` / `brew install rar` / `winget install RARLab.WinRAR`)
- Google Cloud SDK with `gcloud auth application-default login` completed
- A GCP project with Vertex AI API enabled

## Setup

```bash
git clone https://github.com/sergiomarquezdev/gemini-embeddings-2-mongo-demo.git
cd gemini-embeddings-2-mongo-demo
python -m venv .venv && source .venv/Scripts/activate    # or .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set GCP_PROJECT to your project ID
docker compose up -d mongo
gcloud auth application-default login
uvicorn app:app --reload
```

Open http://localhost:8000

## Smoke checklist (pre-recording)

1. `GET /health` returns `{mongo: "ok", dedup_index: "ready", vector_index: "ready", vertex: "configured"}`
2. Upload 3 short `.txt` (cats, soccer, JS), search `"animales domésticos"` → top result is the cats file
3. Upload images (beach, dog, pizza), search `"comida italiana"` → top result is pizza
4. Upload a 4-page scanned PDF, search text only present in the scanned image → matches (OCR works)
5. Upload a 15-page PDF → response shows `n_chunks: 5`
6. Upload the same `.txt` twice → second response is `status: "already_indexed"`
7. Upload a ZIP with `.txt + .png + .pdf + .exe` → 3 ok, 1 skipped
8. Upload a 30-second `.mp3` → search text from the audio → matches
9. Upload an `.exe` directly → 415 with explicit error
10. Upload a corrupt PDF → 422

## Running tests

```bash
docker compose up -d mongo
.venv/Scripts/python.exe -m pytest -v   # Windows
# or: pytest -v   (after activating venv on Linux/Mac)
```

Tests mock Vertex AI but use real MongoDB. They create disposable databases per test and clean up. One test (`tests/test_archives_rar.py::test_extract_rar_with_text_file`) skips automatically if `unrar` is not installed.

## Troubleshooting

- **`vector_index: "missing"`** — Atlas Local builds the index on first start; wait ~30s and retry `/health`.
- **`Vertex AI: 403`** — `gcloud auth application-default login` not completed, or Vertex AI API not enabled in the project.
- **`ffmpeg: not found`** — install ffmpeg system-wide; not pip-installable.
- **`unrar: not found`** — install via system package manager. Without it, `.rar` uploads fail with 500 and the RAR test is skipped.
- **`vector_search: command not recognized`** — you ran the standard `mongo` image instead of `mongodb/mongodb-atlas-local`. Check `docker-compose.yml`.

## Architecture references

- Design spec: `docs/superpowers/specs/2026-05-11-gemini-embedding-2-mongo-design.md`
- Implementation plan: `docs/superpowers/plans/2026-05-12-gemini-embedding-2-mongo-implementation.md`
