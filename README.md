# Agente Vocal Cognitivo — Kognia Labs, Reto 01

Agente de voz que conversa en tiempo real con los datos abiertos de salud de Colombia: **"Relación de IPS públicas y privadas según nivel de atención y capacidad instalada"** (datos.gov.co, `s2ru-bqt6`, 41.427 registros). Muestra la transcripción diarizada, el sentimiento y las emociones en vivo, y el rastro de cada decisión que toma.

## Stack

| Capa | Tecnología | Por qué |
|---|---|---|
| Datos | API SODA de datos.gov.co, paginada de 1.000 en 1.000 (6 en paralelo), validada contra `count(*)`; caché parquet de respaldo | La API es la única fuente; la caché evita depender de la red en la demo |
| Cálculo | pandas en memoria, menos de 5 ms por consulta | Cifras exactas y deterministas: el modelo nunca cuenta por su cuenta |
| Razonamiento | Groq `openai/gpt-oss-120b`, `reasoning_effort=low` | Tool calling confiable y primer token en ~2 s (medido abajo) |
| Emociones y charla social | Groq `openai/gpt-oss-20b` en modo JSON | Rápido y con un límite de uso separado del modelo principal |
| Voz a texto + diarización | Deepgram `nova-2` en streaming (`diarize`, `interim_results`, `utterance_end_ms`) | Separa hablantes con marca de tiempo en vivo |
| Texto a voz | Deepgram Aura-2 `aura-2-celeste-es`, frase por frase; las cifras se convierten a palabras | El agente empieza a hablar antes de terminar de pensar |
| App | FastAPI + HTML/JS sin framework, desplegada en Render | Un solo proceso con WebSocket; cero instalación |

## Arquitectura


**Herramientas:** `consultar_estadistica` (prestadores, sedes o suma de capacidad, con filtros y ranking), `buscar_ips` (búsqueda tolerante a errores), `listar_valores` e `info_dataset`, más `escalar_a_humano`.

**Trampas del dataset que el agente maneja:**
- prestador ≠ sede ≠ fila;
- el nivel de atención solo existe para IPS públicas;
- Cali, Barranquilla, Cartagena, Santa Marta y Buenaventura figuran como departamentos aparte;
- la UCI aparece con varios nombres;
- sumar capacidad sin indicar el grupo mezcla camas, consultorios y ambulancias, así que se obliga a corregir.

## Evaluación (`python evaluar.py`)

9 casos con respuesta esperada calculada desde los datos: resumen, fuera de alcance, trampas del dataset y usuario molesto.

| Modelo | Aciertos | Cifra exacta | Herramienta correcta | Honestidad fuera de alcance | Latencia p50 al primer token |
|---|---|---|---|---|---|
| openai/gpt-oss-120b | **100 %** | 100 % | 100 % | 100 % | **2,1 s** |

El proceso de evaluación mejoró el agente:
- Groq rechazaba parámetros nulos y el modelo sumaba capacidad sin indicar el grupo (3.158 en lugar de 1.646 camas). Ahora los esquemas aceptan nulos y la herramienta exige el grupo.
- El modelo ponía "UCI" en el campo equivocado. Ahora la herramienta lo reubica y lo anota.

En el caso de la trampa (privadas de nivel 3), el agente detectó el resultado sospechoso, ajustó el filtro y lo explicó.

## Ejecutar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # LLM_API_KEY (Groq), DEEPGRAM_API_KEY, DATOS_GOV_TOKEN
pytest -q               # pruebas sin red
uvicorn app:app --port 8001
```

## Uso de IA (regla M04)

El código se desarrolló con asistencia de Claude (Anthropic) y Cursor: backend, interfaz, prompts y pruebas fueron generados con IA y revisados, probados y ajustados por el equipo. El equipo tomó las decisiones de arquitectura (precarga frente a SoQL en vivo, herramientas cerradas, ciclo de autocorrección, emociones en el contexto, escalamiento) y las validó con la evaluación anterior.
