"""Conexión al modelo de lenguaje (cualquier API compatible con OpenAI: Groq, OpenAI, Azure).
Escalamiento de modelos: si el principal responde 429/5xx, pasa al siguiente de la cadena.
"""
import json
import os

from dotenv import load_dotenv
from openai import APIStatusError, AsyncOpenAI

load_dotenv()

CLAVE = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY", "")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
MODELO = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")                 # decisiones con herramientas
MODELO_RAPIDO = os.getenv("LLM_MODEL_RAPIDO", "llama-3.1-8b-instant")  # sentimiento y charla social
RESPALDOS = [m for m in os.getenv("LLM_RESPALDOS", "openai/gpt-oss-20b,llama-3.3-70b-versatile").split(",") if m]

cliente = AsyncOpenAI(api_key=CLAVE or "sin-clave", base_url=BASE_URL, max_retries=0, timeout=30)
ULTIMO_MODELO = {"nombre": MODELO}


def _extra(modelo):
    # gpt-oss razona antes de responder: con esfuerzo bajo la latencia de voz baja mucho.
    return {"reasoning_effort": "low"} if "gpt-oss" in modelo else {}


async def _crear(modelo, **kw):
    cadena = [modelo] + [m for m in RESPALDOS if m != modelo]
    ultimo_error = None
    for m in cadena:
        try:
            r = await cliente.chat.completions.create(model=m, **_extra(m), **kw)
            ULTIMO_MODELO["nombre"] = m
            return r
        except APIStatusError as e:
            ultimo_error = e
            if e.status_code not in (429, 500, 502, 503, 504):
                raise
    raise ultimo_error


async def json_rapido(messages, modelo=None):
    """Llamada corta que devuelve un dict (análisis de sentimiento, brief)."""
    r = await _crear(modelo or MODELO_RAPIDO, messages=messages, temperature=0,
                     response_format={"type": "json_object"})
    return json.loads(r.choices[0].message.content or "{}")


async def stream(messages, tools=None, modelo=None):
    """Genera eventos: ("texto", fragmento) o ("tools", [llamadas completas])."""
    kw = {"messages": messages, "temperature": 0.2, "stream": True}
    if tools:
        kw["tools"] = tools
    respuesta = await _crear(modelo or MODELO, **kw)
    llamadas = {}
    async for parte in respuesta:
        if not parte.choices:
            continue
        d = parte.choices[0].delta
        if d.content:
            yield "texto", d.content
        for tc in d.tool_calls or []:
            c = llamadas.setdefault(tc.index, {"id": "", "nombre": "", "args": ""})
            c["id"] = tc.id or c["id"]
            if tc.function:
                c["nombre"] += tc.function.name or ""
                c["args"] += tc.function.arguments or ""
    if llamadas:
        yield "tools", [llamadas[i] for i in sorted(llamadas)]
