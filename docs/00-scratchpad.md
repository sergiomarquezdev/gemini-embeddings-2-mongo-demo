# Scratchpad — ideas, dudas, anotaciones crudas

## Ideas para capítulos posteriores (Cap. 2 / Cap. 3)

- **Cap. 2:** agente conversacional con Google ADK encima del vector search.
- **Cap. 3:** deploy a Cloud Run + job queue async (Cloud Tasks o BackgroundTasks + `/jobs/{id}` polling) para uploads grandes sin cap.

## Hallazgos de auditorías independientes (Codex gpt-5.5 + code-reviewer) — diferidos a v2

- **Streaming size guard**: actualmente `file.file.read()` consume el upload entero antes de aplicar `_ensure_within_size`, así que un archivo de 1 GB se sube a memoria/spool antes de devolver 413. v2: leer en chunks con un running counter y abortar al superar `MAX_UPLOAD_MB + 1`.
- **`MAX_TOTAL_EMBED_SECONDS` por archivo, no por archive**: un ZIP con 10 audios de 170s cada uno bypassa el cap "total" porque solo se chequea por entry. v2: acumular duración total a través de las entries de un archive y abortar al superar el cap.
- **`$vectorSearch.limit = limit * 2` antes del `$group`**: si un PDF tiene N chunks y N >= limit*2, esos chunks ocupan toda la ventana y dejan fuera otros documentos. v2: subir `numCandidates` y/o `limit` interno proporcional a la cardinalidad esperada.
- **PDF/audio/video corruptos pueden 500**: si `chunk_pdf`/`chunk_audio`/`chunk_video` propaga una excepción (ffmpeg crash, pypdf parse error), `/upload` devuelve 500 con stack trace en log. v2: catch en cada `_ingest_*` y devolver 422 con mensaje genérico (loguear interno).
- **Zip-bomb defense de defensa-en-profundidad**: el guard actual confía en `i.file_size` del header local del ZIP. Un ZIP crafted puede declarar `file_size=0` y aun así descomprimir GB. Para un demo con `MAX_UPLOAD_MB=50` el daño máximo es manejable, pero v2 debería usar streaming decompression con counter.

## Mejoras técnicas v2 (no MVP)

- **Cleanup de orphan files en race condition de dedup**: actualmente si dos uploads simultáneos del mismo archivo llegan al mismo tiempo, el primero gana el `_save_uploaded_file` + `_ingest_text`, el segundo cae en el `DuplicateKeyError` catch y devuelve `already_indexed`, pero el directorio `uploads/{uuid}/` del segundo queda huérfano. Solución v2: borrar el directorio en el `except DuplicateKeyError` block.
- Embeddings paralelos con `asyncio.gather` (cuidado con quota Vertex).
- Batch de hasta 6 imágenes en una sola llamada (optimización costo).
- Near-duplicate detection con MinHash/SimHash.
- Endpoint `DELETE /files/{doc_id}` que limpia tanto Mongo como `uploads/`.
- Cambiar `pymongo` → `motor` para async puro.
- `init_indexes` actualmente hace `try/except Exception: log.warning` alrededor de la creación del vector index para tolerar transient errors de Atlas Local Search Index Management. Considerar un retry/backoff explícito o un mecanismo de readiness probe en `/health` que distinga "index aún en INITIAL_SYNC" de "index nunca creado".

## Pre-grabación del video

- Pre-warm de quota Vertex en `us-central1`: correr 5-10 embeddings de prueba antes de grabar.
- Considerar pedir cuota-bump a GCP si la cuenta es nueva.
- Tener fixtures de prueba listos: 3 .txt, 3 imágenes, 1 PDF escaneado, 1 PDF largo, 1 mp3 corto.
- Verificar que `unrar` está instalado y la ruta esté en `PATH` antes de grabar el caso ZIP/RAR.

## Comparativa para video (datos a recolectar)

- `text-embedding-3-large` (OpenAI) en mismo set de prueba.
- `voyage-multimodal-3` (Voyage) si hay tiempo.
- `cohere-embed-v3` (Cohere) si hay tiempo.

## Dudas pendientes

- ¿`task_type` finalmente se honra o se ignora silenciosamente en `gemini-embedding-2`? Validar empíricamente con embeddings A/B (mismo input, una con task una sin → comparar coseno).
- ¿El `vectorSearchScore` normalizado [0,1] depende de la dimensión o es estable cross-dim?
- ¿`filetype.guess()` retorna `audio/x-wav` para WAV en algunas versiones y `audio/wav` en otras? Verificar con varios WAVs reales (sample rate, mono/stereo, 16/24/32 bit).

## Decisiones pequeñas tomadas durante la implementación (no en spec)

- `docker-compose.yml`: `hostname: mongo` añadido para estabilidad de RS primary election en Atlas Local.
- `init_indexes` envuelve la creación del vector index en `try/except` con `logger.warning` para tolerar errores transitorios del Atlas Search Index Management service.
- `_vector_search` envuelve el `aggregate(pipeline)` en `try/except` con `logger.warning` para devolver `[]` cuando el índice está en INITIAL_SYNC en una colección nueva.
- `_ingest_extracted` envuelve el dispatch de modalidades en un outer `try/except Exception` que captura cualquier fallo de un archivo dentro de un archive y lo reporta como `{"status": "failed", "reason": str(exc)}` en el summary, en vez de propagar y reventar todo el upload.
- `audio/x-wav` añadido a `SUPPORTED_AUDIO` porque `filetype.guess()` lo retorna en lugar de `audio/wav` en este entorno.
- Los tests usan `monkeypatch.setattr(app_module, "MONGO_DB", test_db_name)` y `monkeypatch.setattr("app.MAX_TOTAL_EMBED_SECONDS", ...)` en vez de `monkeypatch.setenv(...)` porque las constantes se leen una vez en import-time. Esto es la diferencia entre lo que decía el plan y lo que terminó funcionando.
