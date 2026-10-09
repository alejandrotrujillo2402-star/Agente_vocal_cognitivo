# Agente Vocal Cognitivo — Kognia Labs, Reto 01

Habla con cualquier documento, en tiempo real. Subes un archivo (o consultas la API de IPS de datos.gov.co), el agente lo indexa como **única fuente de conocimiento**, presenta un brief con preguntas sugeridas y conversa por voz con baja latencia, mientras muestra la **transcripción diarizada** y el **análisis de sentimiento y emociones** en vivo.

## Stack

| Capa | Tecnología | Por qué |
|---|---|---|
| Backend | Python 3.12 · FastAPI · Uvicorn | Streaming, WebSocket y despliegue simple |
| LLM | Groq `openai/gpt-oss-120b` (razonamiento bajo) · respaldo `llama-3.3-70b-versatile` | Muy baja latencia de primer token; API compatible con OpenAI (cambiable por Azure OpenAI con 3 variables) |
| Sentimiento y emociones | Groq `llama-3.1-8b-instant` en modo JSON | Rápido y barato: se ejecuta en cada intervención |
| Voz a texto + diarización | Deepgram `nova-2` en streaming (`diarize`, `interim_results`, `utterance_end_ms`) | Separa hablantes con marca de tiempo en vivo |
| Texto a voz | Deepgram Aura-2 (`aura-2-celeste-es`) · respaldo voz del navegador | Voz natural en español, audio en streaming |
| Recuperación | BM25 propio sobre fragmentos de 900 caracteres con solape | Indexa en milisegundos, sin GPU ni modelos pesados |
| Tablas | pandas + herramienta `consultar_tabla` (tool calling) | Conteos y sumas exactos; el LLM no "calcula de memoria" |
| Datos abiertos | API SODA3 de datos.gov.co con App Token (respaldo SODA2), paginada de 1.000 en 1.000 | Requisito del reto; el dataset tiene más de 41.000 filas |
| Frontend | HTML + JS sin framework (un archivo) | Carga instantánea, cero instalación |
| Despliegue | Render (web service) | Proceso persistente con WebSocket |

## Arquitectura

```
Navegador ── micrófono (webm/opus) ──► /ws/stt ──► Deepgram (STT + diarización)
   │  ◄──── transcripciones parciales y finales por hablante ────┘
   │
   ├─ turno terminado ─► POST /api/preguntar ─► BM25 (top 5 fragmentos) + brief
   │                                          └► LLM en streaming (+ consultar_tabla si es tabla)
   │  ◄── NDJSON: fuentes · texto por fragmentos · herramientas
   │
   ├─ cada frase completa ─► GET /api/tts ─► Deepgram Aura (mp3 en streaming), cola ordenada
   └─ cada intervención ───► POST /api/analizar ─► sentimiento, polaridad y 9 emociones
```

Archivos: `app.py` (rutas, prompts, proxy STT/TTS) · `documentos.py` (lectura PDF/DOCX/PPTX/XLSX/CSV/JSON/TXT/HTML, fragmentos, BM25, consultas a tablas) · `ips.py` (cliente datos.gov.co) · `llm.py` (cliente del modelo con respaldo) · `static/index.html` (interfaz) · `tests/` (15 pruebas sin red).

## Decisiones técnicas

- **Latencia**: el LLM responde en streaming y cada frase se envía a TTS apenas termina, en paralelo, así el agente empieza a hablar antes de terminar de generar. La interfaz muestra la latencia real (fin de la voz del usuario → primer audio).
- **Interrupciones**: si alguien habla mientras el agente responde, se cancela la generación y el audio. Un filtro de eco descarta transcripciones que repiten lo que el agente acaba de decir.
- **Fidelidad y honestidad**: el prompt prohíbe el conocimiento externo; si BM25 no encuentra fragmentos, el modelo lo sabe y responde "eso no aparece en el documento". Los fragmentos usados se muestran como evidencia, con su página.
- **Seguridad**: las claves viven solo en el servidor. El navegador habla con Deepgram a través de un proxy WebSocket y nunca ve la clave.
- **Diarización**: cada palabra trae `speaker`; se agrupan en segmentos por hablante con su marca de tiempo. Las intervenciones del agente se agregan como un hablante más.
- **Tablas grandes (IPS)**: el dataset se agrupa en una ficha por sede para preguntas de detalle, y el modelo usa `consultar_tabla` para cuántos, totales y rankings.
- **Robustez**: si Deepgram no está disponible, se usa el reconocimiento y la voz del navegador; si el modelo principal da 429/503, se usa el de respaldo; si el brief falla, hay un brief básico.

## Ejecutar

```bash
pip install -r requirements.txt
cp .env.example .env      # LLM_API_KEY, DEEPGRAM_API_KEY, DATOS_GOV_TOKEN
uvicorn app:app --reload
pytest -q
```

Despliegue: Render → New → Blueprint con este repo (`render.yaml`) y cargar las tres claves como variables de entorno.

## Uso de IA (declaración, regla M04)

El código se desarrolló con asistencia de Claude (Anthropic): estructura del backend, interfaz, prompts y pruebas fueron generados con IA y revisados, probados y ajustados por el equipo. Las decisiones de arquitectura (BM25 en lugar de embeddings, proxy de STT, voz por frases, herramienta de tablas) se discutieron y validaron con pruebas.
