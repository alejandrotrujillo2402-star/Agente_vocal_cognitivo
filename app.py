"""API del Agente Vocal Cognitivo (fuente única: API de IPS de datos.gov.co).
Rutas: / · /health · /api/estado · /api/brief · /api/preguntar (streaming NDJSON) · /api/analizar ·
       /api/escalamientos · /api/tts · /ws/stt (transcripción diarizada en vivo, proxy a Deepgram)
"""
import asyncio
import json
import os
import re
from collections import Counter, OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlencode

from dotenv import load_dotenv

# El .env se lee antes que cualquier otra cosa (y desde la carpeta del proyecto, no desde el cwd).
load_dotenv(Path(__file__).with_name(".env"))

import httpx  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket  # noqa: E402
from fastapi.responses import FileResponse, Response, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from num2words import num2words  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import agente  # noqa: E402
import datos_ips  # noqa: E402
import documentos  # noqa: E402
import llm  # noqa: E402

DEEPGRAM = os.getenv("DEEPGRAM_API_KEY", "").strip()
TTS_VOZ = os.getenv("TTS_VOZ", "aura-2-celeste-es").strip()
ESTATICOS = Path(__file__).parent / "static"
STT = {"modelo": "nova-3", "idioma": "es-419", "terminos": 0, "error": ""}   # lo que quedó activo en Deepgram (/health)
HTTP: httpx.AsyncClient | None = None       # cliente reutilizable para el TTS: sin handshake TLS por frase


def http():
    global HTTP
    if HTTP is None or HTTP.is_closed:
        HTTP = httpx.AsyncClient(timeout=30, limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=120))
    return HTTP


@asynccontextmanager
async def ciclo(app):
    http()
    calentamiento = [asyncio.create_task(llm.calentar()), asyncio.create_task(precargar_voz())]
    try:
        await datos_ips.cargar()
    except Exception as e:
        print("No se pudieron cargar los datos al arrancar:", e)
    yield
    for t in calentamiento:
        t.cancel()
    await http().aclose()


app = FastAPI(title="Agente Vocal Cognitivo", lifespan=ciclo)
app.mount("/static", StaticFiles(directory=ESTATICOS), name="static")


@app.get("/", include_in_schema=False)
def inicio():
    return FileResponse(ESTATICOS / "index.html")


@app.get("/health")
def health():
    return {"ok": True, "llm": bool(llm.CLAVE), "modelo": llm.MODELO, "modelo_rapido": llm.MODELO_RAPIDO,
            "respaldos": llm.RESPALDOS, "stt": bool(DEEPGRAM), "tts": bool(DEEPGRAM), "stt_modelo": STT["modelo"],
            "stt_idioma": STT["idioma"], "stt_terminos": STT["terminos"], "stt_error": STT["error"] or None, "voz": TTS_VOZ, "datos": datos_ips.estado()}


@app.post("/api/calentar")
async def calentar():
    """La interfaz lo llama al iniciar la conversación: la primera pregunta y la primera frase
    encuentran abiertas las conexiones con el LLM y con Deepgram (el handshake TLS cuesta ~0,7 s)."""
    asyncio.create_task(llm.calentar())
    asyncio.create_task(calentar_voz())
    return {"ok": True}


@app.get("/api/estado")
def estado(sesion: str = ""):
    """Fuente de la sesión: la API, o el archivo que cargó (tipo "ips" o "documento")."""
    return agente.estado_fuente(agente.SESIONES.get(sesion))


@app.post("/api/recargar")
async def recargar():
    await datos_ips.cargar()
    return datos_ips.estado()


@app.get("/api/brief")
def brief(sesion: str = ""):
    try:
        return agente.brief_fuente(agente.SESIONES.get(sesion))
    except Exception as e:
        raise HTTPException(503, f"Datos aún no disponibles: {e}")


@app.post("/api/documento")
async def cargar_documento(sesion: str = Form(..., min_length=1), archivo: UploadFile = File(...)):
    """Archivo del usuario (CSV, XLSX, JSON, PDF, DOCX, PPTX, TXT; máx. 25 MB) como fuente de esta sesión.
    Con las columnas del dataset de IPS reemplaza los datos; si no, se indexa como documento.
    Responde nombre, tipo ("ips" o "documento"), filas o páginas, fragmentos, brief y el estado completo."""
    datos = await archivo.read(documentos.MAX_BYTES + 1)
    if len(datos) > documentos.MAX_BYTES:
        raise HTTPException(413, "El archivo supera el máximo de 25 MB.")
    try:
        fuente = await asyncio.to_thread(documentos.cargar, archivo.filename, datos)
    except documentos.ArchivoInvalido as e:
        raise HTTPException(e.codigo, str(e))  # 422: PDF escaneado · 413: más de 25 MB · 400: formato
    if fuente["tipo"] == "documento":
        fuente["brief"] = await documentos.brief(fuente["doc"])
    agente.cambiar_fuente(sesion, fuente)
    s = agente.sesion(sesion)
    # con una tabla de IPS el brief se calcula con pandas: fuera del event loop
    brief_nuevo = await asyncio.to_thread(agente.brief_fuente, s)
    e = await asyncio.to_thread(agente.estado_fuente, s)
    return {"nombre": fuente["nombre"], "tipo": fuente["tipo"], "filas": e.get("filas"), "paginas": e.get("paginas"),
            "fragmentos": e.get("fragmentos"), "brief": brief_nuevo, "estado": e}


@app.delete("/api/documento")
def quitar_documento(sesion: str):
    """Vuelve a la API de datos.gov.co."""
    agente.cambiar_fuente(sesion, None)
    s = agente.sesion(sesion)
    return {"estado": agente.estado_fuente(s), "brief": agente.brief_fuente(s)}


# ---------------- conversación ----------------

class Pregunta(BaseModel):
    sesion: str
    pregunta: str = Field(min_length=1, max_length=2000)


@app.post("/api/preguntar")
async def preguntar(p: Pregunta):
    s = agente.SESIONES.get(p.sesion)
    if not (s and s.fuente) and not datos_ips.estado().get("filas"):
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
    res = await analizar_texto(a.texto, a.hablante)
    if a.sesion and a.hablante != "Agente":  # el agente lo usa en el turno siguiente, sin esperarlo
        agente.registrar_emocion(agente.sesion(a.sesion), res)
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
    # "1 646" con espacio fino o no separable como separador de miles -> "1.646" (si no, se lee "uno seiscientos...")
    texto = re.sub(r"(?<=\d)[   ](?=\d{3}\b)", ".", texto)
    # también con espacio normal o coma ("1 245", "1,245"), solo si el grupo final tiene exactamente 3 dígitos
    texto = re.sub(r"(?<!\d)(\d{1,3})((?:[ ,]\d{3})+)(?![\d,])", lambda m: m.group(1) + re.sub(r"[ ,]", ".", m.group(2)), texto)
    texto = re.sub(r"(\d[\d.]*),(\d+)", decimal, texto)
    texto = re.sub(r"\d{1,3}(?:\.\d{3})+|\d+", entero, texto)
    texto = re.sub(r"\bveintiuno (mil|millones)", r"veintiún \1", texto)
    return re.sub(r"\buno (mil|millones)", r"un \1", texto)


AUDIO_CORTO: "OrderedDict[tuple, bytes]" = OrderedDict()  # saludo, rellenos y respuestas sociales: sin red
URL_TTS = "https://api.deepgram.com/v1/speak"
# Deben coincidir con SALUDO_INICIAL y RELLENOS de static/index.html
FRASES_FRECUENTES = ["Hola, ¿en qué puedo ayudarte?", "Déjame revisar.", "Un momento, lo consulto.", "Con gusto."]


async def precargar_voz():
    """Al arrancar: sintetiza el saludo y los rellenos para que suenen sin esperar al TTS."""
    if not DEEPGRAM:
        return
    for frase in FRASES_FRECUENTES:
        texto = para_voz(frase)
        try:
            r = await http().post(URL_TTS, params={"model": TTS_VOZ, "encoding": "mp3"},
                                  headers={"Authorization": f"Token {DEEPGRAM}"}, json={"text": texto})
        except Exception as e:
            print("No se pudo precargar la voz:", e)
            return
        if r.status_code != 200:
            print("No se pudo precargar la voz:", r.status_code, r.text[:120])
            return
        AUDIO_CORTO[(TTS_VOZ, texto)] = r.content


async def calentar_voz():
    if DEEPGRAM:
        try:  # cualquier respuesta (aquí 405) deja la conexión abierta en el pool
            await http().head(URL_TTS, headers={"Authorization": f"Token {DEEPGRAM}"})
        except Exception:
            pass


@app.get("/api/tts")
async def tts(texto: str, voz: str = ""):
    if not DEEPGRAM:
        raise HTTPException(503, "TTS no configurado")
    texto = para_voz(texto.strip()[:1200])
    if not texto:
        raise HTTPException(400, "texto vacío")
    clave = (voz or TTS_VOZ, texto)
    if clave in AUDIO_CORTO:
        AUDIO_CORTO.move_to_end(clave)
        return Response(AUDIO_CORTO[clave], media_type="audio/mpeg", headers={"Cache-Control": "no-store"})
    cliente = http()
    pedido = cliente.build_request("POST", URL_TTS,
                                   params={"model": voz or TTS_VOZ, "encoding": "mp3"},
                                   headers={"Authorization": f"Token {DEEPGRAM}"}, json={"text": texto})
    r = await cliente.send(pedido, stream=True)
    if r.status_code != 200:
        detalle = (await r.aread())[:200]
        await r.aclose()
        raise HTTPException(502, f"TTS falló: {detalle!r}")

    async def audio():
        trozos, completo = [], False
        try:
            async for trozo in r.aiter_bytes():
                trozos.append(trozo)
                yield trozo
            completo = True
        finally:
            await r.aclose()
            if completo and len(texto) <= 80:
                AUDIO_CORTO[clave] = b"".join(trozos)
                while len(AUDIO_CORTO) > 64:
                    AUDIO_CORTO.popitem(last=False)

    return StreamingResponse(audio(), media_type="audio/mpeg", headers={"Cache-Control": "no-store"})


# ---------------- transcripción diarizada (STT) ----------------

PARAMS_STT = {"smart_format": "true", "punctuate": "true", "numerals": "true", "diarize": "true",
              "interim_results": "true", "filler_words": "false", "vad_events": "true",
              "endpointing": "300", "utterance_end_ms": "1000"}
# Se prueba en orden; si Deepgram rechaza una combinación se pasa a la siguiente
COMBINACIONES = [("nova-3", "es-419"), ("nova-2", "es-419"), ("nova-2", "es")]
URL_LISTEN = "https://api.deepgram.com/v1/listen"


def params_modelo(modelo, idioma, terminos):
    """Vocabulario del dominio: nova-3 usa keyterm (uno por término); nova-2, keywords con intensidad 2."""
    p = {"model": modelo, "language": idioma}
    if modelo.startswith("nova-3"):
        p["keyterm"] = terminos
    else:
        p["keywords"] = [f"{t}:2" for t in terminos]
    return p


# ---------------- vocabulario de la sesión ----------------

VOCAB_BASE = ["IPS", "EPS", "UCI", "REPS", "prestador", "prestadores", "sede", "sedes", "nivel de atención",
              "capacidad instalada", "camas", "ambulancias", "medicalizada", "consultorios", "quirófanos",
              "datos.gov.co", "Kognia", "Manizales", "Dosquebradas"]
DEPARTAMENTOS = ["Amazonas", "Antioquia", "Arauca", "Atlántico", "Bogotá", "Bolívar", "Boyacá", "Caldas", "Caquetá",
                 "Casanare", "Cauca", "Cesar", "Chocó", "Córdoba", "Cundinamarca", "Guainía", "Guaviare", "Huila",
                 "La Guajira", "Magdalena", "Meta", "Nariño", "Norte de Santander", "Putumayo", "Quindío", "Risaralda",
                 "San Andrés", "Santander", "Sucre", "Tolima", "Valle del Cauca", "Vaupés", "Vichada"]
MAX_TERMINOS = 100
_PARTICULAS = {"de", "del", "la", "las", "los", "el", "y", "e", "en"}
_LEGALES = {"SAS", "SA", "ESE", "LTDA", "IPS"}
_VOCAB: "OrderedDict[tuple, dict]" = OrderedDict()


def nombre_propio(t):
    """'SANTA MARTA' -> 'Santa Marta' · 'CARMEN DE VIBORAL' -> 'Carmen de Viboral'."""
    pal = str(t).strip().lower().split()
    return " ".join(w if i and w in _PARTICULAS else w[:1].upper() + w[1:] for i, w in enumerate(pal))


def nombre_prestador(t):
    """'DUMIAN MÉDICAL S.A.S' -> 'Dumian Médical' (sin sufijos legales, máximo 5 palabras)."""
    pal = [w for w in str(t).split() if re.sub(r"[^A-Za-z]", "", w).upper() not in _LEGALES][:5]
    while pal and pal[-1].lower() in _PARTICULAS:  # "Subred Integrada de Servicios de" -> "... de Servicios"
        pal.pop()
    return nombre_propio(" ".join(pal)).strip(" .,-")


def terminos_documento(doc, n=30):
    """Lo más frecuente de un archivo subido: siglas, nombres propios a mitad de frase y cifras con unidad."""
    texto = " ".join(f["texto"] for f in doc.fragmentos[:2000])
    c = Counter(m.group() for m in re.finditer(r"\b[A-ZÁÉÍÓÚÑ]{2,6}\b", texto))
    c.update(m.group() for m in re.finditer(
        r"(?<=[a-záéíóúñ0-9,;:] )[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,}(?: (?:de |del |la |las |los )?[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,})*", texto))
    c.update(m.group() for m in re.finditer(r"\b\d[\d.,]*\s?(?:%|[a-záéíóúñ]{3,})", texto)
             if documentos.norm(m.group().split()[-1]) not in documentos.VACIAS)
    return [t for t, _ in c.most_common(n)]


def terminos_tabla(df, n=30):
    """Términos frecuentes de una tabla IPS subida: prestadores, sedes, siglas y cifras con unidad."""
    cols = [c for c in ("nombre_prestador", "nom_sede_ips", "departamento", "municipio", "nom_descripcion_capacidad")
            if c in df.columns]
    if not cols:
        return []
    texto = " ".join(str(x) for x in df[cols].fillna("").to_numpy().ravel() if str(x).strip())
    c = Counter(m.group() for m in re.finditer(r"\b[A-ZÁÉÍÓÚÑ]{2,6}\b", texto))
    c.update(m.group() for m in re.finditer(r"\b\d[\d.,]*\s?(?:camas|ambulancias|consultorios|quirófanos|sedes)\b", texto, re.I))
    c.update(nombre_propio(x) for col in cols for x in df[col].dropna().astype(str) if len(str(x).split()) <= 5)
    return [t for t, _ in c.most_common(n)]


def vocabulario(sesion):
    """{"terminos": hasta 100 para Deepgram (keyterm) y para la corrección en el navegador: base del dominio, términos
    del archivo subido, los 40 municipios con más sedes, los 33 departamentos y los 30 prestadores con más camas;
    "extra": todos los municipios y departamentos, que el navegador usa solo para correcciones seguras
    ("2 quebradas" -> "Dosquebradas")}."""
    s = agente.SESIONES.get(sesion)
    doc = s.fuente["doc"] if agente.es_documento(s) else None
    archivo_ips = bool(s and s.fuente and s.fuente["tipo"] == "ips")
    with agente.datos_de(s):
        try:
            df = datos_ips._df()
        except RuntimeError:
            df = None
    llave = (id(doc), id(df))
    if llave in _VOCAB:
        return _VOCAB[llave]
    terminos, extra = VOCAB_BASE + (terminos_documento(doc) if doc else []), list(DEPARTAMENTOS)
    if df is not None and len(df):
        if archivo_ips:
            terminos += terminos_tabla(df)
        extra += sorted({nombre_propio(m) for m in df["municipio"].unique() if m})
        terminos += [nombre_propio(m) for m in df.groupby("municipio")["sede_id"].nunique().nlargest(40).index if m]
        terminos += DEPARTAMENTOS
        camas = df[df["nom_grupo_capacidad_n"] == "camas"].groupby("nombre_prestador")["cantidad"].sum().nlargest(30)
        terminos += [nombre_prestador(x) for x in camas.index]
    else:
        terminos += DEPARTAMENTOS
    vistos, out = set(), []
    for t in terminos:
        k = documentos.norm(t)
        if t and k not in vistos:
            vistos.add(k)
            out.append(t)
    _VOCAB[llave] = {"terminos": out[:MAX_TERMINOS], "extra": [t for t in extra if documentos.norm(t) not in vistos]}
    while len(_VOCAB) > 16:
        _VOCAB.popitem(last=False)
    return _VOCAB[llave]


@app.get("/api/vocabulario")
async def ver_vocabulario(sesion: str = ""):
    return await asyncio.to_thread(vocabulario, sesion)


def motivo_deepgram(e):
    """Texto legible del fallo: 'HTTP 401 ... (clave rechazada)', 'HTTP 400: No such model...', etc."""
    r = getattr(e, "response", None)
    if r is not None and getattr(r, "status_code", None):
        cuerpo = (getattr(r, "body", b"") or b"").decode(errors="ignore").strip()
        try:
            cuerpo = json.loads(cuerpo).get("err_msg") or cuerpo
        except Exception:
            pass
        txt = f"HTTP {r.status_code}" + (f": {cuerpo}" if cuerpo else "")
        if r.status_code in (401, 403):
            txt += " (clave de Deepgram rechazada)"
        return txt[:200]
    return (str(e) or type(e).__name__)[:200]


async def conectar_deepgram(terminos):
    """Abre la conexión con Deepgram: nova-3 + es-419; si rechaza la combinación, nova-2 + es-419 y nova-2 + es."""
    from websockets.asyncio.client import connect
    for i, (modelo, idioma) in enumerate(COMBINACIONES):
        url = "wss://api.deepgram.com/v1/listen?" + urlencode(
            {**params_modelo(modelo, idioma, terminos), **PARAMS_STT}, doseq=True)
        try:
            dg = await connect(url, additional_headers={"Authorization": f"Token {DEEPGRAM}"},
                               max_size=None, open_timeout=10)
            STT.update(modelo=modelo, idioma=idioma, terminos=len(terminos))
            return dg
        except Exception as e:
            clave_mala = getattr(getattr(e, "response", None), "status_code", 0) in (401, 403)
            if i + 1 < len(COMBINACIONES) and not clave_mala:
                print(f"Deepgram rechazó {modelo} + {idioma} ({motivo_deepgram(e)}); pruebo la siguiente")
                continue
            raise


@app.websocket("/ws/stt")
async def ws_stt(ws: WebSocket):
    """Puente navegador <-> Deepgram: la clave nunca llega al navegador.
    Si Deepgram no acepta la conexión se envía {"type": "Error"} (la interfaz pasa al navegador);
    si se cae a mitad de la conversación se envía {"type": "Aviso"} y la interfaz reconecta."""
    from websockets.exceptions import ConnectionClosed
    await ws.accept()
    try:
        if not DEEPGRAM:
            raise RuntimeError("DEEPGRAM_API_KEY no está configurada en el servidor")
        terminos = (await asyncio.to_thread(vocabulario, ws.query_params.get("sesion", "")))["terminos"]
        dg = await conectar_deepgram(terminos)
    except Exception as e:
        STT["error"] = motivo_deepgram(e)
        print("Deepgram no disponible:", STT["error"])
        try:
            await ws.send_text(json.dumps({"type": "Error", "description": STT["error"]}))
            await ws.close()
        except Exception:
            pass
        return
    STT["error"] = ""

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
        try:
            async for m in dg:
                await ws.send_text(m if isinstance(m, str) else m.decode())
        except ConnectionClosed as e:
            STT["error"] = motivo_deepgram(e)
            print("Deepgram cerró la conexión:", STT["error"])
            await ws.send_text(json.dumps({"type": "Aviso", "description": STT["error"]}))

    tareas = [asyncio.create_task(subir()), asyncio.create_task(bajar())]
    try:
        await asyncio.wait(tareas, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for t in tareas:
            t.cancel()
        await asyncio.gather(*tareas, return_exceptions=True)
        await dg.close()
        try:
            await ws.close()
        except Exception:
            pass


# ---------------- refinado del turno (Deepgram pregrabado) ----------------

@app.post("/api/refinar")
async def refinar(sesion: str = Form(""), audio: UploadFile = File(...)):
    """Audio de un turno del usuario (WAV del navegador) -> transcripción pregrabada, más precisa que la de streaming.
    No bloquea al agente: la interfaz lo pide en segundo plano y solo mejora el texto en pantalla."""
    if not DEEPGRAM:
        raise HTTPException(503, "STT no configurado")
    datos = await audio.read(10 * 1024 * 1024 + 1)
    if not datos or len(datos) > 10 * 1024 * 1024:
        raise HTTPException(413, "Audio vacío o mayor de 10 MB")
    terminos = (await asyncio.to_thread(vocabulario, sesion))["terminos"]
    comunes = {"smart_format": "true", "punctuate": "true", "numerals": "true", "diarize": "true", "filler_words": "false"}
    for modelo, idioma in COMBINACIONES:
        r = await http().post(URL_LISTEN, params={**params_modelo(modelo, idioma, terminos), **comunes}, content=datos,
                              headers={"Authorization": f"Token {DEEPGRAM}", "Content-Type": audio.content_type or "audio/wav"})
        if r.status_code == 200 or r.status_code in (401, 403):
            break
    if r.status_code != 200:
        raise HTTPException(502, f"Deepgram no pudo refinar: HTTP {r.status_code} {r.text[:150]}")
    STT.update(modelo=modelo, idioma=idioma, terminos=len(terminos), error="")
    alt = r.json()["results"]["channels"][0]["alternatives"][0]
    palabras = [{"w": w.get("punctuated_word") or w["word"], "conf": round(w.get("confidence", 1), 3),
                 "start": w["start"], "end": w["end"], "hablante": w.get("speaker", 0)} for w in alt.get("words", [])]
    segmentos = []
    for p in palabras:
        if segmentos and segmentos[-1]["hablante"] == p["hablante"]:
            s = segmentos[-1]
            s.update(fin=p["end"], texto=s["texto"] + " " + p["w"], _c=s["_c"] + [p["conf"]])
        else:
            segmentos.append({"hablante": p["hablante"], "inicio": p["start"], "fin": p["end"], "texto": p["w"], "_c": [p["conf"]]})
    for s in segmentos:
        c = s.pop("_c")
        s["confianza"] = round(sum(c) / len(c), 3)
    return {"modelo": modelo, "idioma": idioma, "texto": alt.get("transcript", ""), "palabras": palabras, "segmentos": segmentos}
