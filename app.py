"""API del Agente Vocal Cognitivo (fuente única: API de IPS de datos.gov.co).
Rutas: / · /health · /api/estado · /api/brief · /api/preguntar (streaming NDJSON) · /api/analizar ·
       /api/escalamientos · /api/tts · /ws/stt (transcripción diarizada en vivo, proxy a Deepgram)
"""
import asyncio
import json
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse, StreamingResponse
from num2words import num2words
from pydantic import BaseModel, Field

import agente
import datos_ips
import llm

load_dotenv()
DEEPGRAM = os.getenv("DEEPGRAM_API_KEY", "").strip()
STT_MODELO = os.getenv("STT_MODEL", "nova-2").strip() or "nova-2"
TTS_VOZ = os.getenv("TTS_VOZ", "aura-2-celeste-es").strip() or "aura-2-celeste-es"
ESTATICOS = Path(__file__).parent / "static"
DG_ERROR = {"motivo": None, "hora": None}
HTTP: httpx.AsyncClient | None = None


def cliente_http():
    global HTTP
    if HTTP is None or HTTP.is_closed:
        HTTP = httpx.AsyncClient(timeout=30, limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=120))
    return HTTP


@asynccontextmanager
async def ciclo(app):
    cliente_http()
    try:
        await datos_ips.cargar()
    except Exception as e:
        print("No se pudieron cargar los datos al arrancar:", e)
    calentamiento = asyncio.create_task(llm.calentar())
    yield
    calentamiento.cancel()
    if HTTP is not None:
        await HTTP.aclose()


app = FastAPI(title="Agente Vocal Cognitivo", lifespan=ciclo)


@app.get("/", include_in_schema=False)
def inicio():
    return FileResponse(ESTATICOS / "index.html")


@app.get("/health")
def health():
    return {"ok": True, "llm": bool(llm.CLAVE), "modelo": llm.MODELO, "modelo_rapido": llm.MODELO_RAPIDO,
            "respaldos": llm.RESPALDOS, "stt": bool(DEEPGRAM), "tts": bool(DEEPGRAM), "stt_modelo": STT_MODELO,
            "stt_error": DG_ERROR["motivo"], "stt_error_hora": DG_ERROR["hora"],
            "stt_motivo": None if DEEPGRAM else "DEEPGRAM_API_KEY no configurada en el servidor",
            "voz": TTS_VOZ, "datos": datos_ips.estado()}


@app.get("/api/estado")
def estado():
    return datos_ips.estado()


@app.post("/api/recargar")
async def recargar():
    await datos_ips.cargar()
    return datos_ips.estado()


@app.get("/api/brief")
def brief():
    try:
        return datos_ips.brief()
    except Exception as e:
        raise HTTPException(503, f"Datos aún no disponibles: {e}")


# ---------------- conversación ----------------

class Pregunta(BaseModel):
    sesion: str
    pregunta: str = Field(min_length=1, max_length=2000)


@app.post("/api/preguntar")
async def preguntar(p: Pregunta):
    if not datos_ips.estado().get("filas"):
        raise HTTPException(503, "Los datos de datos.gov.co aún no están cargados.")

    async def generar():
        try:
            async for ev in agente.responder(p.sesion, p.pregunta):
                yield json.dumps(ev, ensure_ascii=False, default=str) + "\n"
        except Exception as e:
            yield json.dumps({"t": "error", "x": str(e)[:300]}) + "\n"
            yield json.dumps({"t": "delta", "x": "Tuve un problema técnico al consultar. ¿Puedes repetir la pregunta?"}) + "\n"
            yield json.dumps({"t": "fin", "x": {}}) + "\n"

    return StreamingResponse(generar(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/escalamientos")
def escalamientos():
    return agente.ESCALAMIENTOS[-20:][::-1]


@app.post("/api/reiniciar")
def reiniciar(p: dict):
    agente.SESIONES.pop(p.get("sesion", ""), None)
    return {"ok": True}


# ---------------- sentimiento y emociones ----------------

EMOCIONES = ["alegria", "confianza", "interes", "sorpresa", "confusion", "frustracion", "enojo", "tristeza", "miedo"]

PROMPT_EMOCION = f"""Analiza el sentimiento y las emociones de una intervención en una conversación hablada en español con un asistente.
Devuelve SOLO un JSON: {{"sentimiento": "positivo"|"neutral"|"negativo", "polaridad": número entre -1 y 1,
"emociones": {{{", ".join(f'"{e}": 0 a 1' for e in EMOCIONES)}}}, "dominante": emoción más alta o "neutral"}}.
Una pregunta informativa normal es neutral con interés moderado. Considera el tono y la intención, no el tema."""


class Analisis(BaseModel):
    texto: str = Field(min_length=1, max_length=2000)
    hablante: str = ""
    sesion: str = ""


def _num(v, lo, hi):
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return 0.0


async def analizar_texto(texto, hablante=""):
    try:
        r = await asyncio.wait_for(llm.json_rapido([{"role": "system", "content": PROMPT_EMOCION},
                                                    {"role": "user", "content": f"{hablante}: {texto}"}]), 3)
    except Exception:
        r = {}
    emociones = {e: round(_num((r.get("emociones") or {}).get(e, 0), 0, 1), 2) for e in EMOCIONES}
    sentimiento = r.get("sentimiento") if r.get("sentimiento") in ("positivo", "neutral", "negativo") else "neutral"
    dominante = r.get("dominante") if r.get("dominante") in EMOCIONES else (
        max(emociones, key=emociones.get) if max(emociones.values()) >= 0.4 else "neutral")
    return {"sentimiento": sentimiento, "polaridad": round(_num(r.get("polaridad", 0), -1, 1), 2),
            "emociones": emociones, "dominante": dominante}


@app.post("/api/analizar")
async def analizar(a: Analisis):
    humano = a.sesion and a.hablante != "Agente"
    s = agente.sesion(a.sesion) if humano else None
    tarea = asyncio.ensure_future(analizar_texto(a.texto, a.hablante))
    if s:
        s.pendientes.add(tarea)
    try:
        res = await tarea
    finally:
        if s:
            s.pendientes.discard(tarea)
    if s:
        agente.registrar_emocion(s, res)
    return res


# ---------------- voz del agente (TTS) ----------------

def para_voz(texto):
    """Cifras a palabras para que la voz no lea '41.427' como 'cuarenta y uno punto...'."""
    def entero(m):
        try:
            return num2words(int(m.group(0).replace(".", "").replace(" ", "")), lang="es")
        except Exception:
            return m.group(0)

    def decimal(m):
        try:
            a, b = m.group(1).replace(".", ""), m.group(2)
            return f"{num2words(int(a), lang='es')} coma {num2words(int(b), lang='es')}"
        except Exception:
            return m.group(0)

    texto = texto.replace("%", " por ciento")
    texto = re.sub(r"(\d[\d.]*),(\d+)", decimal, texto)
    texto = re.sub(r"\d{1,3}(?:\.\d{3})+|\d+", entero, texto)
    texto = re.sub(r"\bveintiuno (mil|millones)", r"veintiún \1", texto)
    return re.sub(r"\buno (mil|millones)", r"un \1", texto)


@app.get("/api/tts")
async def tts(texto: str, voz: str = ""):
    if not DEEPGRAM:
        raise HTTPException(503, "TTS no configurado")
    texto = para_voz(texto.strip()[:1200])
    if not texto:
        raise HTTPException(400, "texto vacío")
    http = cliente_http()
    pedido = http.build_request("POST", "https://api.deepgram.com/v1/speak",
                                params={"model": voz or TTS_VOZ, "encoding": "mp3"},
                                headers={"Authorization": f"Token {DEEPGRAM}"}, json={"text": texto})
    r = await http.send(pedido, stream=True)
    if r.status_code != 200:
        detalle = (await r.aread())[:200]
        await r.aclose()
        raise HTTPException(502, f"TTS falló: {detalle!r}")

    async def audio():
        try:
            async for trozo in r.aiter_bytes():
                yield trozo
        finally:
            await r.aclose()

    return StreamingResponse(audio(), media_type="audio/mpeg", headers={"Cache-Control": "no-store"})


# ---------------- transcripción diarizada (STT) ----------------

PARAMS_STT = {"language": "es", "diarize": "true", "smart_format": "true",
              "punctuate": "true", "interim_results": "true", "utterance_end_ms": "1200",
              "vad_events": "true", "endpointing": "400"}


def registrar_error_dg(motivo):
    DG_ERROR["motivo"] = motivo
    DG_ERROR["hora"] = datetime.now().strftime("%H:%M:%S") if motivo else None
    if motivo:
        print("Deepgram falló:", motivo)


def motivo_dg(e):
    """Texto legible de un fallo de Deepgram; en el handshake incluye el código HTTP y el cuerpo."""
    resp = getattr(e, "response", None)
    if resp is not None and hasattr(resp, "status_code"):
        cuerpo = getattr(resp, "body", b"") or b""
        cuerpo = cuerpo.decode("utf-8", "replace") if isinstance(cuerpo, bytes) else str(cuerpo)
        dg = resp.headers.get("dg-error") if getattr(resp, "headers", None) else None
        return f"HTTP {resp.status_code}: {dg or cuerpo or 'sin detalle'}"[:200]
    return (str(e) or type(e).__name__)[:200]


async def conectar_dg():
    """Abre el WebSocket con Deepgram; si el modelo es rechazado, reintenta una vez con el otro."""
    from websockets.asyncio.client import connect
    modelos = [STT_MODELO, "nova-3" if STT_MODELO != "nova-3" else "nova-2"]
    for i, modelo in enumerate(modelos):
        url = "wss://api.deepgram.com/v1/listen?" + "&".join(f"{k}={v}" for k, v in {"model": modelo, **PARAMS_STT}.items())
        try:
            return await connect(url, additional_headers={"Authorization": f"Token {DEEPGRAM}"}, max_size=None)
        except Exception as e:
            motivo = motivo_dg(e)
            if i == 0 and "model" in motivo.lower():
                continue
            raise RuntimeError(motivo) from e


@app.websocket("/ws/stt")
async def ws_stt(ws: WebSocket):
    """Puente navegador <-> Deepgram: la clave nunca llega al navegador."""
    await ws.accept()
    if not DEEPGRAM:
        await ws.send_text(json.dumps({"type": "Error", "description": "DEEPGRAM_API_KEY no configurada en el servidor"}))
        await ws.close()
        return
    try:
        dg = await conectar_dg()
        registrar_error_dg(None)
        async with dg:
            async def subir():
                while True:
                    m = await ws.receive()
                    if m["type"] == "websocket.disconnect":
                        await dg.send(json.dumps({"type": "CloseStream"}))
                        return
                    if m.get("bytes"):
                        await dg.send(m["bytes"])
                    elif m.get("text"):
                        await dg.send(m["text"])

            async def bajar():
                async for m in dg:
                    await ws.send_text(m if isinstance(m, str) else m.decode())

            tareas = [asyncio.create_task(subir()), asyncio.create_task(bajar())]
            hechas, pendientes = await asyncio.wait(tareas, return_when=asyncio.FIRST_COMPLETED)
            for t in pendientes:
                t.cancel()
            bajada = tareas[1]
            if bajada in hechas and bajada.exception() is not None:
                raise bajada.exception()
    except Exception as e:
        motivo = motivo_dg(e)
        registrar_error_dg(motivo)
        try:
            await ws.send_text(json.dumps({"type": "Error", "description": motivo}))
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass
