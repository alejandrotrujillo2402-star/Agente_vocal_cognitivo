"""Conexión al modelo de lenguaje (cualquier API compatible con OpenAI: Groq, OpenAI, Azure).
Escalamiento de modelos: si el principal responde 429/5xx, pasa al siguiente de la cadena.
"""
import asyncio
import json
import os
import re
import time
from contextvars import ContextVar
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

# Consumo de tokens: total acumulado desde el arranque (se ve en /api/estado) y un registro por petición en logs/tokens.jsonl
CONSUMO = {"llamadas": 0, "prompt_tokens": 0, "completion_tokens": 0, "peticiones": 0, "por_tipo": {}}
LOG_TOKENS = Path(__file__).with_name("logs") / "tokens.jsonl"
# Petición en curso: {"tipo", "cuenta", "max_tokens"}. La fija quien llama (agente, análisis) sin cambiar las firmas.
PETICION: ContextVar = ContextVar("peticion", default=None)


def contar(uso):
    """Suma el usage de una llamada al total, a su tipo (pregunta, emocion, brief...) y a la cuenta de la petición."""
    if not uso:
        return
    pet = PETICION.get() or {}
    tipo, cuenta = pet.get("tipo", "otro"), pet.get("cuenta")
    p, c = int(getattr(uso, "prompt_tokens", 0) or 0), int(getattr(uso, "completion_tokens", 0) or 0)
    for d in (CONSUMO, CONSUMO["por_tipo"].setdefault(tipo, {"llamadas": 0, "prompt_tokens": 0, "completion_tokens": 0}),
              *([cuenta] if cuenta is not None else [])):
        d["llamadas"] = d.get("llamadas", 0) + 1
        d["prompt_tokens"] = d.get("prompt_tokens", 0) + p
        d["completion_tokens"] = d.get("completion_tokens", 0) + c


def registrar_peticion(tipo, texto, cuenta):
    """Una línea por petición: cuánto costó cada pregunta (o cada análisis)."""
    CONSUMO["peticiones"] += 1
    try:
        LOG_TOKENS.parent.mkdir(exist_ok=True)
        with LOG_TOKENS.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "tipo": tipo, "texto": str(texto)[:120],
                                "llamadas": cuenta.get("llamadas", 0), "prompt_tokens": cuenta.get("prompt_tokens", 0),
                                "completion_tokens": cuenta.get("completion_tokens", 0)}, ensure_ascii=False) + "\n")
    except OSError:
        pass


async def calentar():
    """Llamada mínima para abrir la conexión con el proveedor antes de la primera pregunta real."""
    if not CLAVE:
        return
    for modelo in dict.fromkeys([MODELO, MODELO_RAPIDO]):
        try:
            await cliente.chat.completions.create(model=modelo, messages=[{"role": "user", "content": "ok"}],
                                                  max_tokens=1, **_extra(modelo))
        except Exception as e:  # aunque el proveedor la rechace, la conexión ya quedó abierta
            print("Calentamiento del LLM:", modelo, str(e)[:120])


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
    contar(getattr(r, "usage", None))
    return json.loads(r.choices[0].message.content or "{}")


async def stream(messages, tools=None, modelo=None):
    """Genera eventos: ("texto", fragmento) o ("tools", [llamadas completas]). El usage va a la petición en curso."""
    kw = {"messages": messages, "temperature": 0.2, "stream": True, "stream_options": {"include_usage": True}}
    if tools:
        kw["tools"] = tools
    pet = PETICION.get() or {}
    if pet.get("max_tokens"):
        kw["max_tokens"] = pet["max_tokens"]
    if tools and pet.get("tool_choice"):
        kw["tool_choice"] = pet["tool_choice"]
    respuesta = await _crear(modelo or MODELO, **kw)
    llamadas, uso = {}, None
    async for parte in respuesta:
        if getattr(parte, "usage", None):
            uso = parte.usage   # Groq lo repite en dos trozos: se cuenta una sola vez al final
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
    contar(uso)
    if llamadas:
        yield "tools", [llamadas[i] for i in sorted(llamadas)]
