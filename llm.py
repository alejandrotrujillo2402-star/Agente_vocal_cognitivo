"""Conexión al modelo de lenguaje (cualquier API compatible con OpenAI: Groq, OpenAI, Azure).
Escalamiento de modelos: si el principal responde 429/5xx, pasa al siguiente de la cadena.
"""
import asyncio
import json
import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import APIStatusError, AsyncOpenAI, DefaultAsyncHttpxClient

load_dotenv(Path(__file__).with_name(".env"))

CLAVE = (os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY", "")).strip()
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").strip()
MODELO = os.getenv("LLM_MODEL", "openai/gpt-oss-120b").strip()                 # decisiones con herramientas
MODELO_RAPIDO = os.getenv("LLM_MODEL_RAPIDO", "openai/gpt-oss-20b").strip()  # sentimiento y charla social
RESPALDOS = [m.strip() for m in os.getenv("LLM_RESPALDOS", "openai/gpt-oss-20b").split(",") if m.strip()]

# Conexiones vivas 2 min (por defecto son 5 s): la pregunta siguiente no repite el handshake TLS.
cliente = AsyncOpenAI(api_key=CLAVE or "sin-clave", base_url=BASE_URL, max_retries=0, timeout=30,
                      http_client=DefaultAsyncHttpxClient(limits=httpx.Limits(
                          max_connections=100, max_keepalive_connections=20, keepalive_expiry=120)))
ULTIMO_MODELO = {"nombre": MODELO}


async def calentar():
    """Llamada mínima para abrir la conexión con el proveedor antes de la primera pregunta real."""
    if not CLAVE:
        return
    try:
        await cliente.chat.completions.create(model=MODELO, messages=[{"role": "user", "content": "ok"}], max_tokens=1)
    except Exception as e:  # aunque el proveedor la rechace, la conexión ya quedó abierta
        print("Calentamiento del LLM:", str(e)[:120])


def _espera(e):
    """Segundos sugeridos por el proveedor tras un 429 (cabecera retry-after o texto 'try again in 1.2s')."""
    try:
        return float(e.response.headers.get("retry-after"))
    except Exception:
        m = re.search(r"try again in ([\d.]+)(m?s)", str(e))
        if m:
            return float(m.group(1)) / (1000 if m.group(2) == "ms" else 1)
    return 99


def _extra(modelo):
    # gpt-oss razona antes de responder: con esfuerzo bajo la latencia de voz baja mucho.
    return {"reasoning_effort": "low"} if "gpt-oss" in modelo else {}


async def _crear(modelo, **kw):
    cadena = [modelo] + [m for m in RESPALDOS if m != modelo]
    ultimo_error = None
    for m in cadena:
        for intento in range(2):
            try:
                r = await cliente.chat.completions.create(model=m, **_extra(m), **kw)
                ULTIMO_MODELO["nombre"] = m
                return r
            except APIStatusError as e:
                ultimo_error = e
                espera = _espera(e)
                # 429 con espera corta: vale más esperar un instante que cambiar a un modelo más débil
                if e.status_code == 429 and intento == 0 and espera <= 4:
                    await asyncio.sleep(espera)
                    continue
                break
        # 404 = modelo no disponible en la cuenta; 429/5xx = saturado: se pasa al siguiente de la cadena
        if ultimo_error is not None and ultimo_error.status_code not in (404, 429, 500, 502, 503, 504):
            raise ultimo_error
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
