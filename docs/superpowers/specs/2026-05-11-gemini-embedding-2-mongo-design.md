# Design Spec — Demo educativa de `gemini-embedding-2` + MongoDB Vector Search

**Fecha:** 2026-05-11 (rev. 2026-05-11 review Claude / rev. 2026-05-12 review gpt-5.5)
**Autor:** smarq + Claude (sesión brainstorming + 2 revisiones independientes)
**Objetivo del proyecto:** Material didáctico (capítulos en `docs/` + demo funcional) que el autor convertirá en guion de video. La demo prueba en vivo cómo `gemini-embedding-2` indexa cualquier modalidad y cómo MongoDB con `$vectorSearch` resuelve búsqueda semántica multimodal.

> **Nota de versión:** ID canónico per doc oficial Vertex AI = `gemini-embedding-2` (GA). Algunos ejemplos del SDK `google-genai` aún muestran `gemini-embedding-2-preview` (escritos en era preview); si el SDK rechaza el ID GA, fallback a `-preview` y reportar issue al SDK.

---

## 1. Decisiones clave (resumen ejecutivo)

| Decisión | Elegido | Motivo |
|---|---|---|
| Stack | Python + FastAPI + Jinja/HTML | Familiar a audiencia ML/AI, async nativo |
| Auth GCP | Application Default Credentials (`gcloud auth application-default login`) | Lo más limpio para dev local |
| MongoDB | Docker local con `mongodb/mongodb-atlas-local:latest` | Soporta `$vectorSearch` sin cuenta cloud |
| Modalidades | Texto, imagen, PDF, audio, video (las 5) | Showcase completo de embedding-2 |
| Arquitectura | Monolito didáctico: `app.py` (endpoints + lógica) + 2 helpers cortos (`chunking.py`, `archives.py`) | Endpoints de un vistazo en cámara; helpers extraídos solo cuando suman legibilidad |
| Documentación | Capitulado temático + `00-scratchpad.md` | Materia prima organizada para guion |
| Embedding dim | 1536 (vía `output_dimensionality=1536` — MRL trunca desde 3072) | Sweet spot calidad/storage; comparable con OpenAI |
| Modelo | `gemini-embedding-2` en Vertex AI | Único endpoint multimodal de la familia (estado preview) |
| Flags embedding-2 | `document_ocr=True`, `audio_track_extraction=True` | Activan OCR de PDFs y mezcla audio+frames de video (soportados según SDK) |
| Task type | `RETRIEVAL_DOCUMENT`/`RETRIEVAL_QUERY` **best-effort** | El SDK los acepta en `EmbedContentConfig` pero NO hay evidencia de que el modelo multimodal los honre. Si Vertex devuelve `INVALID_ARGUMENT`, omitir y aceptar simetría doc/query |
| ADK | Excluido del MVP | Reservado para Cap. 2 (agente conversacional encima) |
| Idempotencia | SHA-256 del archivo, índice compuesto en Mongo | Evita reembedding y duplicados |
| Comprimidos | ZIP + RAR con extracción y salvaguardas | Caso de uso real en bulk uploads |

---

## 2. Arquitectura general

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────────────┐
│  Browser        │     │  FastAPI (app.py) │     │  Vertex AI               │
│  index.html     │────▶│                  │────▶│  gemini-embedding-2-     │
│  (drag-and-drop │     │  /upload         │     │  preview                 │
│   + search box) │◀────│  /search         │◀────│  (3072 nativo →          │
└─────────────────┘     └────────┬─────────┘     │   1536 vía MRL trunc)    │
                                                  └──────────────────────────┘
                                 │
                                 ▼
                        ┌────────────────────┐
                        │  MongoDB Atlas     │
                        │  Local (Docker)    │
                        │  - colección docs  │
                        │  - índice vector   │
                        │    cosine similarity│
                        └────────────────────┘
```

**Idea central:** cada archivo se proyecta a un vector de 1536 dims que vive en Mongo junto con metadata. La búsqueda toma una query (texto o archivo) y pide a Mongo los K más cercanos por coseno, agrupando chunks por documento padre.

**Fuera de scope MVP** (anotado en scratchpad):
- Autenticación / multi-usuario
- Reranking adicional (el modelo ya hace su parte)
- UI sofisticada (Tailwind CDN, sin build)
- Embeddings paralelos con `asyncio.gather`
- Borrado de docs vía API (se hace por Mongo Compass)
- Near-duplicate detection (MinHash/SimHash)

---

## 3. Componentes y dependencias

### Estructura de archivos

```
gemini_embeddings_2/
├── app.py                    # FastAPI: endpoints + lógica embedding + Mongo
├── chunking.py               # Helpers: split de texto/PDF/audio/video
├── archives.py               # Helpers: extracción ZIP/RAR + salvaguardas
├── templates/
│   └── index.html            # UI con drag-and-drop (Jinja2)
├── static/
│   └── styles.css            # Tailwind CDN, sin build
├── uploads/                  # Archivos subidos (gitignored)
├── docs/
│   ├── 00-scratchpad.md
│   ├── 01-que-es-un-embedding.md
│   ├── 02-gemini-embedding-2-overview.md
│   ├── 03-multimodalidad-y-task-instructions.md
│   ├── 04-mongodb-vector-search.md
│   ├── 05-arquitectura-y-chunking.md
│   ├── 06-implementacion-paso-a-paso.md
│   └── 07-comparativa-competidores.md
├── tests/
├── docker-compose.yml        # Levanta mongodb-atlas-local
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

### Dependencias Python

```
fastapi[standard]==0.115.*       # incluye uvicorn + multipart
jinja2==3.1.*
google-genai==1.*                # SDK oficial unificado Vertex AI / Gemini
pymongo==4.10.*                  # cliente MongoDB sync
python-dotenv==1.0.*             # cargar .env
pypdf==5.*                       # split PDF por páginas
ffmpeg-python==0.2.*             # chunking de audio Y video (requiere ffmpeg sistema)
filetype==1.2.*                  # MIME sniffing real (no confiar en Content-Type del browser)
rarfile==4.*                     # extracción RAR (requiere unrar/bsdtar sistema)
```

> **Nota:** descartamos `pydub` (deprecado en Py 3.13+, depende de `audioop` removido). Usamos `ffmpeg-python` para audio Y video — consolidamos en una sola dep.

### Binarios del sistema

- `ffmpeg` — para chunking de audio/video. Documentado en README.
- `unrar` o `bsdtar` — para extracción RAR. Documentado en README.
- `libmagic` (opcional) — `filetype` no lo necesita pero `python-magic` sí; nos quedamos con `filetype` para evitarlo.

### Servicios externos

- **Vertex AI** (`us-central1` por defecto). Auth vía ADC.
- **MongoDB Atlas Local** (Docker, puerto 27017). Imagen pineada `mongodb/mongodb-atlas-local:8.0.5` (NO `latest` — reproducibilidad del video).

> **Nota sync/async:** usamos `pymongo` (sync) dentro de FastAPI (async). Endpoints de Mongo van como `def` sync, los corre el threadpool de FastAPI. Aceptable para la demo. Para producción real cambiar a `motor` (async). Lo aclaramos en el video como tradeoff didáctico.

### Variables de entorno (`.env`)

```
GCP_PROJECT=tu-proyecto
GCP_LOCATION=us-central1
MONGO_URI=mongodb://localhost:27017/?directConnection=true
MONGO_DB=embeddings_demo
MONGO_COLLECTION=documents
EMBEDDING_DIM=1536
EMBEDDING_MODEL=gemini-embedding-2     # pinear versión exacta
MAX_UPLOAD_MB=50                                # cap por archivo individual
MAX_ARCHIVE_FILES=10                            # cap archivos dentro de ZIP/RAR
MAX_ARCHIVE_UNCOMPRESSED_MB=50                  # cap suma descomprimida
MAX_TOTAL_EMBED_SECONDS=1800                    # cap suma duración audio+video por upload (cost guard)
```

---

## 4. Flujo de datos

### Upload (`POST /upload`)

```
Browser                  FastAPI /upload              chunking.py            Vertex AI                MongoDB
  │                           │                           │                      │                       │
  │── POST file (multipart) ──▶│                           │                      │                       │
  │                           │── compute SHA-256          │                      │                       │
  │                           │── find_one(content_hash)──▶│                                              │
  │                           │     dedup hit? return 200 con status:already_indexed                       │
  │                           │── detect MIME type        │                      │                       │
  │                           │── archivo? extract ───────▶ archives.py          │                       │
  │                           │   (procesar cada extraído como upload anidado)   │                       │
  │                           │── needs chunking? ────────▶│                      │                       │
  │                           │◀── list of chunks ────────│                      │                       │
  │                           │── for each chunk:                                │                       │
  │                           │     embed(chunk, dim=1536, task=RETRIEVAL_DOC)──▶│                       │
  │                           │◀───────────── vector[1536] ──────────────────────│                       │
  │                           │── insert_many({...vector, metadata...}) ────────────────────────────────▶│
  │◀── 200 OK { doc_id, n_chunks } ─────────────────────────────────────────────────────────────────────│
```

**Decisiones:**
- `task=RETRIEVAL_DOCUMENT` al indexar.
- Un chunk = un documento Mongo. `parent_doc_id` agrupa chunks del mismo upload, `chunk_index` los ordena.
- Embedding por chunk en serie (legibilidad). Asyncio en scratchpad para v2.
- `preview_label` (primeros 500 chars o descripción del binario) para mostrar resultados sin re-leer original.

### Búsqueda (`POST /search`)

```
Browser                FastAPI /search          Vertex AI               MongoDB
  │                         │                       │                       │
  │── POST {query OR file} ─▶│                       │                       │
  │                         │── embed(input,                               │
  │                         │     task=RETRIEVAL_QUERY,                      │
  │                         │     dim=1536) ────────▶│                       │
  │                         │◀── vector[1536] ──────│                       │
  │                         │── $vectorSearch ──────────────────────────────▶│
  │                         │◀──────────── chunks + scores ──────────────────│
  │                         │── group by parent_doc_id, take max score       │
  │◀── results JSON ────────│                                                │
```

**Decisiones:**
- `task=RETRIEVAL_QUERY` al buscar (asimetría con indexing — clave para calidad).
- `numCandidates=400, limit=20` (ratio 20× recomendado por Mongo para recall).
- Búsqueda multimodal nativa: el endpoint acepta texto O archivo.
- Agrupar por `parent_doc_id` para no devolver el mismo PDF varias veces.

### Mapeo modalidad → estrategia de chunking

| Si subís... | Qué hacemos |
|---|---|
| `.txt`, `.md` | usar **`client.models.count_tokens(model, contents)`** (NO contar palabras — el ratio token/palabra varía mucho en español/código/tablas). Chunk si >7000 tokens (margen vs 8192). Overlap 500 tokens |
| `.pdf` | `pypdf` SOLO para contar páginas y dividir el binario; **NO extraemos texto local** — la gracia didáctica es el OCR del modelo. Split en bloques de **4 págs con 1 overlap** (margen vs límite 6); **fallback a 3 págs** si la llamada rebota por token-limit |
| `.png/.jpg/.webp/.bmp/.heic/.heif/.avif` | sin chunk, embedding directo. Hasta **6 imágenes pueden ir en una sola llamada** (batch opcional) |
| `.mp3/.wav` | `ffmpeg-python` lee duración, split en bloques de 170s con 10s overlap |
| `.mp4/.mpeg` | `ffmpeg -t 70 -ss N` extrae chunks de **70s con 10s overlap** (margen vs límite 81s con audio) |
| `.zip/.rar` | extraer y procesar cada archivo recursivamente (depth 1) |
| Otro MIME | rechazar con 415 + mensaje claro |

### Límites del modelo (de la doc oficial — verificados 2026-05-11)

| Modalidad | Límite por llamada |
|---|---|
| Texto | 8,192 tokens (~6k palabras) |
| PDF (`application/pdf`) | **1 archivo por prompt, 6 páginas/archivo** |
| Audio (`audio/mp3` o `audio/mpeg`, `audio/wav`) | 180 segundos. Aceptar AMBOS MIMEs para mp3 (browser puede mandar cualquiera) |
| Video con audio (`video/mp4`, `video/mpeg`) | ~81 s (limitado por 8192 tokens) |
| Video sin audio | 120 frames (≈120 s a 1 FPS) |
| Imagen (`image/png`, `jpeg`, `webp`, `bmp`, `heic`, `heif`, `avif`) | **6 por prompt** |

> **MIMEs NO soportados** (rechazar con 415 explícito): `audio/flac`, `audio/ogg`, `audio/m4a`, `video/webm`, `video/mkv`, `video/mov`, `video/quicktime`. Documentar la lista en error response.

---

## 5. Esquema Mongo + índice vectorial

### Documento en `documents`

```jsonc
{
  "_id": ObjectId("..."),
  "parent_doc_id": "uuid-v4-string",
  "chunk_index": 0,
  "n_chunks_total": 5,
  "modality": "pdf",                              // text | pdf | image | audio | video
  "filename": "informe_q3.pdf",
  "mime_type": "application/pdf",
  "size_bytes": 1245678,
  "storage_path": "uploads/uuid-v4/informe_q3.pdf",
  "preview_label": "Resumen ejecutivo del Q3...",  // texto real para .txt/.md/.pdf; etiqueta sintética para binarios ("imagen 800x600 png", "audio 45s mp3")
  "content_hash": "sha256:a3f5b2...",            // mismo en todos los chunks del padre
  "chunk_meta": {
    "page_start": 1, "page_end": 5
    // o "time_start": 0, "time_end": 75 (audio/video)
  },
  "source_archive": {                            // solo si vino de un .zip/.rar
    "filename": "informes_q3.zip",
    "extracted_at": "informes/marketing.pdf"
  },
  "vector": [0.0123, -0.0456, ...],              // length = 1536; NULL si status != "ok"
  "embedding_model": "gemini-embedding-2",
  "embedding_dim": 1536,
  "embedding_task": "RETRIEVAL_DOCUMENT",         // best-effort, ver Sec.1
  "embedding_flags": {
    "document_ocr": true,
    "audio_track_extraction": true
  },
  "status": "ok",                                 // "ok" | "failed" | "pending"
  "error": null,                                  // mensaje si status == "failed"
  "retry_count": 0,                               // intentos hasta 3 antes de marcar failed
  "created_at": ISODate("2026-05-11T...")
}
```

### Índices

**Vectorial (Atlas vectorSearch):**

```jsonc
{
  "name": "vector_index",
  "type": "vectorSearch",
  "definition": {
    "fields": [
      {"type": "vector", "path": "vector", "numDimensions": 1536, "similarity": "cosine"},
      {"type": "filter", "path": "modality"},
      {"type": "filter", "path": "parent_doc_id"},
      {"type": "filter", "path": "status"}
    ]
  }
}
```

**Dedup (B-tree compuesto, ÚNICO — anti-race):**

```python
# Índice único sobre (hash, model, dim, chunk_index) — bloquea inserts duplicados
# si dos uploads del mismo archivo llegan a la vez y ambos pasan el find_one.
db.documents.create_index(
    [("content_hash", 1), ("embedding_model", 1), ("embedding_dim", 1), ("chunk_index", 1)],
    name="dedup_idx",
    unique=True,
)
```

Compuesto a propósito: si cambia modelo o dim, el archivo se reembebe (es lo correcto — son embeddings distintos). **Unique** porque el `find_one` previo NO es atómico con el `insert_many`: bajo concurrencia (dos requests con el mismo archivo en paralelo) ambos pueden pasar la check. El insert del segundo falla con `DuplicateKeyError`, lo capturamos y devolvemos `status: "already_indexed"`.

> **Filtro `status: "ok"` en el pipeline de búsqueda:** agregamos `{$match: {status: "ok"}}` antes de `$vectorSearch` (o como `filter` del operador) para no contaminar resultados con chunks `failed`/`pending`. Requiere `status` como `filter` field del índice vectorial.

### Aggregation pipeline de búsqueda

```python
pipeline = [
    {"$vectorSearch": {
        "index": "vector_index",
        "path": "vector",
        "queryVector": query_embedding,
        "numCandidates": 400,                                # 20× limit (recomendado por Mongo para recall)
        "limit": 20,
        "filter": {                                          # filtros combinados
            "status": "ok",                                  # excluir chunks failed/pending
            "modality": {"$in": ["pdf", "text"]}            # opcional según UI
        }
    }},
    {"$project": {
        "vector": 0,
        "score": {"$meta": "vectorSearchScore"},
        "parent_doc_id": 1, "chunk_index": 1, "modality": 1,
        "filename": 1, "preview_label": 1, "storage_path": 1, "chunk_meta": 1
    }},
    {"$sort": {"parent_doc_id": 1, "score": -1}},
    {"$group": {
        "_id": "$parent_doc_id",
        "best_score": {"$first": "$score"},
        "best_chunk": {"$first": "$$ROOT"},
        "matched_chunks": {"$sum": 1}
    }},
    {"$sort": {"best_score": -1}},
    {"$limit": 10}
]
```

> **Nota UI:** `vectorSearchScore` está **normalizado a [0, 1]** por Mongo. NO es similaridad coseno cruda (que va en [-1, 1]). Si la UI muestra "82% match", aclarar al espectador que es score normalizado, no coseno directo.

> **Nota didáctica:** `numCandidates=400` es el valor correcto para producción según docs de Mongo (20× limit). Para una demo con <500 chunks el efecto sobre recall es marginal, pero usamos el valor correcto desde el inicio para no enseñar mal.

---

## 6. Manejo de errores y casos borde

### En `/upload`

| Caso | Respuesta | Por qué |
|---|---|---|
| MIME no soportado (sniff real con `filetype`) | **415** + `{error, supported_types}` | Falla rápido y accionable. Sniff real, NO confiar en `Content-Type` del browser (un `.exe` renombrado a `.pdf` debe rebotar) |
| Archivo > `MAX_UPLOAD_MB` (50 MB default) | **413** | Límite nuestro antes que el de Vertex |
| PDF corrupto | **422** | Error del archivo, no de la API |
| Vertex AI timeout (>30s/chunk) | **502** + chunk marcado `failed` | Doc queda parcial pero usable |
| Vertex AI 429 | retry exponencial 1s/2s/4s, luego **503** | Quotas de Vertex |
| Vertex AI 401/403 | **500** + log claro | Bug de config |
| Vertex AI `INVALID_ARGUMENT` por token-limit en PDF | **fallback automático**: re-split a 3 págs y reintentar | Caso conocido por margen ajustado |
| Mongo down | **503** | Diferenciable de Vertex |
| ffmpeg no instalado | **500** + *"ffmpeg required: see README"* | Setup, no runtime |
| Mongo `DuplicateKeyError` (race idempotencia) | capturar, devolver **200** con `status: "already_indexed"` | Race protection — segundo upload en paralelo del mismo archivo |
| Suma de duración audio+video > `MAX_TOTAL_EMBED_SECONDS` | **413** + `{error: "embedding budget exceeded", limit_seconds, requested_seconds}` | Cost guard antes de empezar a llamar Vertex |

### En `/search`

| Caso | Respuesta |
|---|---|
| Query vacía y sin archivo | **400** + `{error: "provide query or file"}` |
| Cero resultados | **200** + `{results: [], message: "no matches"}` |
| Embedding falla | **502** específico |
| Filtro inválido | **400** + lista de modalidades válidas |

### Patrón global

```python
{
  "error": "Vertex AI rate limit",
  "code": "VERTEX_RATE_LIMIT",
  "details": "Retry after 60s",
  "request_id": "uuid-..."
}
```

`request_id` logueado en cada request para correlación.

### Idempotencia (hash SHA-256)

- Computar SHA-256 antes de chunkear (~100ms para 50MB).
- `find_one({content_hash, embedding_model, embedding_dim})`.
- Si existe → `200 OK` con `status: "already_indexed"` (NO 409 — el resultado deseado se cumple).
- **Race protection**: el `find_one` NO es atómico con `insert_many`. Si dos requests del mismo archivo llegan en paralelo, ambos pueden pasar la check. **El índice `unique=True` (Sec. 5)** bloquea el segundo insert con `DuplicateKeyError` → lo capturamos y devolvemos `already_indexed` igualmente.
- Misma lógica para cada archivo extraído de ZIP/RAR.

### Seguridad de archivos servidos

- **NO exponer `storage_path` crudo en respuestas API**: devolver un endpoint `/files/{doc_id}` que valida acceso y resuelve a `storage_path` server-side.
- **Sanitizar nombres** al guardar: `secure_filename()` o regex `[^a-zA-Z0-9._-]` → `_`. El nombre original queda en `filename` (metadata), el archivo en disco usa `{uuid}.{ext}`.
- **Sin `StaticFiles` directo sobre `uploads/`**: nada de directory listing. Endpoint dedicado con `FileResponse` y MIME explícito.
- **`Content-Disposition: attachment`** al servir, evita ejecución inline en browser de tipos peligrosos.

### Archivos comprimidos (ZIP + RAR)

**Cap duro de tamaño (decisión consciente — evita timeout de browser en demo grabada):**

| Salvaguarda | Valor |
|---|---|
| **Cantidad de archivos** | **≤ 10** (config: `MAX_ARCHIVE_FILES`) |
| **Tamaño descomprimido total** | **≤ 50 MB** (config: `MAX_ARCHIVE_UNCOMPRESSED_MB`) |
| **Anti-zip-bomb (PRE-extracción)** | sumar `ZipInfo.file_size` de cada entry **ANTES** de extraer; si suma > cap → **413** sin tocar disco |
| **Path traversal** | Descartar entries con `..`, paths absolutos, o que resuelvan fuera del temp dir post-`os.path.realpath` |
| **Profundidad** | 1 (no recursar en archivos anidados; skipear con warning) |
| **Encriptados full** | **422** *"encrypted archives not supported"* |
| **ZIP con headers legibles + contenido encriptado** | detectar vía `ZipInfo.flag_bits & 0x1` antes de extraer (no esperar a fallar en read) |

> **Por qué el cap duro y no async/job-queue:** un upload con 100 archivos × 5 chunks promedio = 500 llamadas seriales × 1.5s ≈ 12 min, que time-outea el browser y rompe la grabación. Job queue se anota como mejora para Cap. 3 (deploy a Cloud Run con Cloud Tasks).

**Respuesta `/upload` con archivo:**

```json
{
  "archive": "informes_q3.zip",
  "extracted": [
    {"doc_id": "...", "filename": "marketing.pdf", "n_chunks": 3, "status": "ok"},
    {"doc_id": "...", "filename": "logo.png", "status": "already_indexed"},
    {"doc_id": null, "filename": "data.xlsx", "status": "skipped", "reason": "unsupported MIME"},
    {"doc_id": null, "filename": "video.mkv", "status": "skipped", "reason": "MKV not supported by gemini-embedding-2"}
  ],
  "summary": {"total": 4, "ok": 1, "already_indexed": 1, "skipped": 2, "failed": 0}
}
```

---

## 7. Plan de pruebas

### Smoke tests manuales — checklist en cámara

Sirven como QA pre-grabación y como guion de los segmentos del video.

**Setup:**
1. `docker compose up -d` → mongo-atlas-local arranca, healthcheck OK
2. `gcloud auth application-default login` → ADC OK
3. `pip install -r requirements.txt`
4. `uvicorn app:app --reload`
5. `GET /health` → `{vertex: ok, mongo: ok, index: ready}`

**Caso 1 — Texto puro:** subir 3 `.txt` (gatos, fútbol, JS), buscar `"animales domésticos"` → top = gatos.

**Caso 2 — Magia multimodal (clip viral):** subir imágenes (playa, perro, pizza), buscar texto `"comida italiana"` → top = pizza.

**Caso 3 — PDF con OCR:** subir factura escaneada, buscar texto solo presente en imagen escaneada.

**Caso 4 — Chunking PDF largo:** subir PDF de 15 págs → 3 chunks, buscar contenido de pág 12 → encontrado con `chunk_index: 2`.

**Caso 5 — Idempotencia:** subir mismo `.txt` dos veces → segunda devuelve `status: "already_indexed"`. Verificar 1 entrada en Mongo.

**Caso 6 — ZIP mix:** subir `.zip` con `.txt + .png + .pdf + .exe` → 3 procesados, 1 skipped.

**Caso 7 — Audio:** subir `.mp3` 30s, buscar texto del podcast → match.

**Caso 8 — Errores controlados:** `.exe` suelto → 415; PDF corrupto → 422; query vacía → 400.

### Tests automatizados — cobertura mínima

`pytest` + `httpx.AsyncClient`. **Mockear Vertex AI** (no gastar quota), **NO mockear Mongo** (queremos verificar `$vectorSearch` real).

```
tests/
├── conftest.py                # fixtures
├── test_chunking.py           # unit: PDF 1 pág, audio <1s, video sin track, texto =8192 tokens (off-by-one)
├── test_upload.py             # integración: Vertex mockeado, sniff MIME real
├── test_search.py             # integración: vectores fake, filtros válidos e inválidos
├── test_dedup.py              # subir dos veces el mismo archivo
├── test_zip.py                # mix válidos/inválidos/skipped
└── test_archive_safety.py     # zip bomb sintética (rechazo por headers PRE-extract), path traversal (`../`), nombre UTF-8, ZIP con encrypted-content + clear-headers
```

**Lo que no testeamos automáticamente:**
- Calidad real de embeddings (manual + ojo humano).
- Llamadas reales a Vertex (caro, flaky).
- UI (sin e2e en MVP).

**Cómo correr:**
```
docker compose up -d mongo
pytest -v
```

---

## 8. Roadmap futuro (anotado en scratchpad)

- **Cap. 2:** Agente conversacional con Google ADK encima del vector search.
- **Cap. 3:** Deploy a Cloud Run con autenticación + **job queue async** (Cloud Tasks o `BackgroundTasks` + endpoint `/jobs/{id}` para uploads grandes sin cap).
- **v2 técnica:** embeddings paralelos con `asyncio.gather`; near-duplicate detection (MinHash/SimHash); endpoint `DELETE`; **batch de 6 imágenes por llamada** (optimización de costo).
- **Comparativa para video:** medir contra `text-embedding-3-large` (OpenAI), `voyage-multimodal-3` (Voyage), `cohere-embed-v3` (Cohere) en un set de prueba pequeño.
- **Pre-grabación del video:** **pre-warm de la quota Vertex** en `us-central1` (correr 5-10 embeddings de prueba antes de grabar para evitar throttling visible en cámara). Considerar pedir cuota-quota-bump a GCP si hace falta.

---

## 9. Convenciones del proyecto

- Comentarios y commits en inglés.
- Conventional commits.
- Sin emojis en código ni docs.
- Documentación de proceso: `docs/00-scratchpad.md` para ideas crudas, `docs/0X-*.md` para capítulos.

---

## 10. Changelog del spec

- **rev. 2026-05-11 (initial)** — diseño aprobado en sesión brainstorming.
- **rev. 2026-05-11 (post review #1 — superpowers:code-reviewer Claude)** — críticos C1-C4 (anti-zip-bomb pre-extract, cap duro ZIP, MIME sniff `filetype`, pinear modelo); mejoras M1-M8 (ffmpeg-python en vez de pydub, numCandidates 20×, etc.); V1 corregido (6 imágenes/prompt, no 1).
- **rev. 2026-05-12 (post review #2 — gpt-5.5 vía Codex)**:
  - **G1** índice dedup `unique=True` + captura de `DuplicateKeyError` (race protection)
  - **G2** chunking de texto por **tokens vía `count_tokens`**, no por palabras (ratio variable en español/código/tablas)
  - **G3** schema con `status`/`error`/`retry_count`; filtro `status: "ok"` en vector search
  - **G4** model ID corregido a `gemini-embedding-2` (GA per doc oficial), no `-preview`
  - **G5** `task_type RETRIEVAL_DOCUMENT/QUERY` marcado como **best-effort** (SDK lo acepta pero sin evidencia de que el modelo multimodal lo honre — fallback a omisión si Vertex devuelve `INVALID_ARGUMENT`)
  - Mongo image pineada (`8.0.5`, no `latest`); nota explícita `pymongo` sync en FastAPI async
  - Security uploads: endpoint `/files/{doc_id}` con FileResponse + `Content-Disposition: attachment`; sanitizar nombres; no exponer `storage_path` crudo
  - `MAX_TOTAL_EMBED_SECONDS=1800` como cost guard pre-Vertex
  - PDF: aclarado que `pypdf` solo divide binario, NO extrae texto (OCR es del modelo)
  - Renames: `text_preview` → `preview_label`; `monolito único` → `monolito + helpers`
  - MIMEs audio: aceptar `audio/mp3` y `audio/mpeg` (browser puede mandar cualquiera)
  - Flags Python: `True` (no `true`) en código
