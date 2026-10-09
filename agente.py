"""Cerebro del agente: contexto por capas, enrutador de modelos, ciclo de decisión con
autocorrección, emociones que cambian el comportamiento y escalamiento a humano.
"""
import asyncio
import json
import random
import re
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime

import datos_ips
import llm

MAX_PASOS = 4
NEGATIVAS = ("frustracion", "enojo", "confusion")
MAX_ARCHIVOS = 4   # sesiones con archivo propio en memoria; al pasar el límite, la más antigua vuelve a la API

HABLA = """Cómo hablas:
- Es voz: 1 a 3 frases cortas, máximo 60 palabras, español natural, sin markdown, listas ni emojis. Escribe las cifras con dígitos.
- Adapta el tono al ESTADO EMOCIONAL: con frustración o confusión, reconoce en pocas palabras, simplifica y ofrece un camino concreto; con enojo, no discutas ni te justifiques.
- Si aparece la instrucción ESCALAR, llama escalar_a_humano y luego informa el número de caso una sola vez.
- Si te piden silencio, no respondas. Si el mensaje es un fragmento incompleto, pide en máximo 6 palabras que complete la idea."""

SISTEMA = f"""Eres el agente de voz de datos abiertos de salud de Colombia. Tu ÚNICA fuente es el dataset "Relación de IPS públicas y privadas según nivel de atención y capacidad instalada" de datos.gov.co, al que accedes SOLO con herramientas.

Cómo decides:
1. Para cualquier cifra, nombre o dato del dataset, llama una herramienta. Nunca digas un número que no venga de un resultado.
2. Si un resultado trae "error" o "sugerencias", corrígete tú mismo: vuelve a llamar con la sugerencia más parecida, sin preguntarle al usuario si la corrección es obvia.
3. Si trae "sospechoso": true, aplica la "recomendacion" y di brevemente qué ajustaste.
4. Si tras intentarlo no hay datos, dilo con honestidad ("Eso no aparece en el dataset") y ofrece algo que sí puedes responder.
5. Si la pregunta es ambigua y las interpretaciones dan respuestas muy distintas, pregunta en una frase corta; si no, elige la más razonable y dilo (por ejemplo "conté sedes, no prestadores").
6. Lo que el dataset no contiene (médicos, EPS, precios, calidad, citas, datos posteriores a la fecha de corte): responde que no está, sin llamar herramientas.
7. Usa el CONTEXTO ACTIVO para entender referencias como "¿y en Pereira?" o "esa IPS".

{HABLA}"""

SISTEMA_DOC = f"""Eres un agente de voz que responde preguntas sobre un documento que cargó el usuario. Tu ÚNICA fuente es ese documento, al que accedes SOLO con herramientas.

Cómo decides:
1. Antes de responder sobre el contenido, llama buscar_en_documento con palabras clave. Si no encuentras, reintenta una vez con sinónimos o con los términos sugeridos.
2. Responde solo con lo que digan los fragmentos. Si el resultado trae "pagina", menciónala ("según la página 4"); si no, no hables de fragmentos ni de números de fila. Nunca completes con conocimiento general.
3. Para contar, sumar, promediar o comparar datos de una tabla usa consultar_tabla; no calcules a partir de fragmentos.
4. Si el documento no lo dice, dilo con honestidad ("Eso no aparece en el documento") y ofrece algo que sí contenga.
5. Usa el historial para entender referencias como "¿y eso cuándo fue?".

{HABLA}"""

TOOL_ESCALAR = {
    "type": "function",
    "function": {
        "name": "escalar_a_humano",
        "description": "Transfiere la conversación a un analista humano con todo el contexto para que el usuario no repita nada.",
        "parameters": {
            "type": "object",
            "properties": {
                "motivo": {"type": "string", "enum": ["frustracion", "sin_respuesta", "solicitud_usuario"]},
                "resumen": {"type": "string", "description": "Qué necesita el usuario, en una frase"},
                "pendiente": {"type": "string", "description": "Qué quedó sin resolver"},
            },
            "required": ["motivo", "resumen", "pendiente"],
        },
    },
}

PIDE_HUMANO = re.compile(r"\b(humano|asesor|persona real|una persona|alguien real|supervisor|analista|operador)\b")
SILENCIO = re.compile(r"\b(callate|callado|callada|silencio|para ya|detente|deja de hablar|no hables|espera|stop|shh)\b")
INTERROGATIVA = re.compile(r"(?<!\w)(qué|cuánt[oa]s?|cuál(es)?|dónde|cómo|quién(es)?|cuándo)(?!\w)")

# Vocabulario social: un turno es social solo si TODO él está hecho de estas frases
# ("hola, ¿cuántas sedes hay en Caldas?" no lo es y va al modelo con herramientas).
_SALUDO = r"hola|holi|hey|buenas|buen dia|buenos dias|buenas tardes|buenas noches|que tal|saludos"
_COMO = r"como estas|como esta|como te va|como vas|como andas|que mas|todo bien"
_GRACIAS = r"gracias|muchas gracias|mil gracias|te agradezco|muy amable"
_DESPEDIDA = r"adios|chao|chau|bye|hasta luego|hasta pronto|hasta manana|nos vemos|hasta la proxima|eso es todo"
_RELLENO = (r"ok|okay|vale|listo|perfecto|bien|muy bien|genial|excelente|super|claro|bueno|pues|si|agente|asistente|"
            r"amigo|amiga|y tu|y usted|igualmente|por favor|a ti|a usted|tambien|de nuevo")
_OTRO_SOCIAL = r"quien eres|que eres|que haces|que puedes hacer|como funcionas|entiendo|de acuerdo"


def _solo(*grupos):
    return re.compile(rf"(?:\b(?:{'|'.join(grupos)})\b\s*)+")


INMEDIATA = _solo(_SALUDO, _COMO, _GRACIAS, _DESPEDIDA, _RELLENO)
SOCIAL = _solo(_SALUDO, _COMO, _GRACIAS, _DESPEDIDA, _RELLENO, _OTRO_SOCIAL)
CATEGORIAS = [(c, re.compile(rf"\b(?:{rx})\b")) for c, rx in
              (("despedida", _DESPEDIDA), ("gracias", _GRACIAS), ("como", _COMO), ("saludo", _SALUDO))]
PLANTILLAS = {
    "saludo": ["Hola, ¿en qué puedo ayudarte?", "Hola, ¿qué quieres consultar?", "¡Hola! Cuéntame, ¿en qué te ayudo?"],
    "como": ["Muy bien, gracias. ¿En qué puedo ayudarte?", "Todo bien, gracias. ¿Qué quieres consultar?"],
    "gracias": ["Con gusto.", "Con mucho gusto.", "A la orden.", "Para servirte."],
    "despedida": ["Hasta luego.", "Hasta pronto, que estés muy bien.", "Fue un gusto. Hasta luego."],
}
SALUDO_HORA = {"buenos dias": "Buenos días", "buen dia": "Buen día", "buenas tardes": "Buenas tardes", "buenas noches": "Buenas noches"}


def norm(t):
    t = unicodedata.normalize("NFKD", str(t).lower())
    return "".join(c for c in t if not unicodedata.combining(c)).strip(" ¿?¡!.,")


def plano(t):
    """Minúsculas, sin tildes ni puntuación: 'Hola, ¿cómo estás?' -> 'hola como estas'."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", norm(t)).split())


def pide_silencio(pregunta):
    """'cállate', 'quédate callado'... Solo 'espera' + una pregunta ('espera, ¿y en Pereira?') no es un comando."""
    hallados = {m.group(0) for m in SILENCIO.finditer(norm(pregunta))}
    if not hallados:
        return False
    return not (hallados == {"espera"} and ("?" in pregunta or INTERROGATIVA.search(pregunta.lower())))


def tiene_contenido(texto):
    """False para respuestas vacías o de solo puntuación ('...', '¿?')."""
    return bool(re.sub(r"[\W_]+", "", texto or ""))


@dataclass
class Sesion:
    historial: list = field(default_factory=list)      # [{"role", "content"}]
    emociones: list = field(default_factory=list)      # análisis de intervenciones humanas
    sociales: dict = field(default_factory=dict)       # respuestas instantáneas dadas, por categoría
    fallos: int = 0                                    # turnos seguidos sin respuesta válida
    entidades: dict = field(default_factory=dict)      # contexto activo
    ticket: dict | None = None
    fuente: dict | None = None                         # archivo propio (documentos.cargar); None = API datos.gov.co


SESIONES: "OrderedDict[str, Sesion]" = OrderedDict()
ESCALAMIENTOS: list = []
CON_ARCHIVO: "OrderedDict[str, None]" = OrderedDict()


def sesion(sid) -> Sesion:
    if sid not in SESIONES:
        SESIONES[sid] = Sesion()
        while len(SESIONES) > 200:
            SESIONES.popitem(last=False)
    SESIONES.move_to_end(sid)
    return SESIONES[sid]


# ---------------- fuente de conocimiento por sesión ----------------

def cambiar_fuente(sid, fuente):
    """Archivo propio de la sesión (o None para volver a la API). El contexto de la fuente anterior ya no aplica."""
    s = sesion(sid)
    s.fuente, s.fallos = fuente, 0
    s.entidades.clear()
    s.historial.clear()
    CON_ARCHIVO.pop(sid, None)
    if fuente:
        CON_ARCHIVO[sid] = None
    while len(CON_ARCHIVO) > MAX_ARCHIVOS:  # cada archivo vive en memoria: se libera el más antiguo
        viejo, _ = CON_ARCHIVO.popitem(last=False)
        if viejo in SESIONES:
            SESIONES[viejo].fuente = None


def es_documento(s: Sesion | None):
    return bool(s and s.fuente and s.fuente["tipo"] == "documento")


def datos_de(s: Sesion | None):
    """Contexto en el que las herramientas de IPS ven el archivo de la sesión (o la API)."""
    return datos_ips.con_fuente(s.fuente["datos"] if s and s.fuente and s.fuente["tipo"] == "ips" else None)


def estado_fuente(s: Sesion | None):
    if es_documento(s):
        return {"fuente": f"archivo {s.fuente['nombre']}", **s.fuente["doc"].estado()}
    with datos_de(s):
        e = datos_ips.estado()
    return {**e, "tipo": "ips", "archivo": s.fuente["nombre"]} if s and s.fuente else {**e, "tipo": "api"}


def brief_fuente(s: Sesion | None):
    if es_documento(s):
        return s.fuente["brief"]
    with datos_de(s):
        return datos_ips.brief()


def herramientas(s: Sesion | None):
    return s.fuente["doc"].herramientas() if es_documento(s) else datos_ips.TOOLS


def ejecutar_herramienta(s: Sesion, nombre, args):
    if es_documento(s):
        return s.fuente["doc"].ejecutar(nombre, args)
    with datos_de(s):
        return datos_ips.ejecutar(nombre, args)


# ---------------- emociones ----------------

def registrar_emocion(s: Sesion, analisis: dict):
    s.emociones.append(analisis)
    s.emociones[:] = s.emociones[-10:]


def tension(a):
    e = a.get("emociones", {})
    return max(e.get(k, 0) for k in NEGATIVAS)


def estado_emocional(s: Sesion):
    if not s.emociones:
        return "sin datos todavía"
    a = s.emociones[-1]
    dom = a.get("dominante", "neutral")
    nivel = a.get("emociones", {}).get(dom, 0)
    texto = f"última intervención {a.get('sentimiento', 'neutral')}, emoción dominante {dom} ({nivel:.1f})"
    if len(s.emociones) >= 2 and all(tension(x) >= 0.6 for x in s.emociones[-2:]):
        texto += "; tensión alta sostenida en los 2 últimos turnos"
    return texto


def motivo_escalar(s: Sesion, pregunta: str):
    if s.ticket:
        return None
    if PIDE_HUMANO.search(norm(pregunta)):
        return "solicitud_usuario"
    if len(s.emociones) >= 2 and all(tension(x) >= 0.6 for x in s.emociones[-2:]):
        return "frustracion"
    if s.fallos >= 2:
        return "sin_respuesta"
    return None


def crear_ticket(s: Sesion, args: dict, emocion: str):
    t = {"id": f"ESC-{random.randint(100000, 999999)}", "motivo": args.get("motivo", "solicitud_usuario"),
         "resumen": args.get("resumen", ""), "pendiente": args.get("pendiente", ""), "emocion": emocion,
         "contexto": dict(s.entidades), "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
         "transcripcion": [{"rol": m["role"], "texto": m["content"]} for m in s.historial[-6:]]}
    s.ticket = t
    ESCALAMIENTOS.append(t)
    return t


# ---------------- contexto ----------------

def actualizar_entidades(s: Sesion, resultado: dict):
    filtros = resultado.get("filtros_aplicados") or {}
    for k in ("departamento", "municipio", "naturaleza", "nivel_atencion", "grupo_capacidad", "descripcion_capacidad"):
        if filtros.get(k):
            s.entidades[k] = filtros[k]
    if resultado.get("ips"):
        s.entidades["ips"] = resultado["ips"]


def construir_mensajes(s: Sesion, pregunta: str, escalar):
    activo = ", ".join(f"{k}={v}" for k, v in s.entidades.items()) or "ninguno"
    estado = (f"ESTADO DE LA CONVERSACIÓN\n- CONTEXTO ACTIVO: {activo}\n"
              f"- ESTADO EMOCIONAL del usuario: {estado_emocional(s)}\n"
              f"- Turnos seguidos sin respuesta válida: {s.fallos}")
    if s.ticket:
        estado += f"\n- El caso ya fue escalado ({s.ticket['id']}). No vuelvas a escalar ni repitas el número."
    elif escalar:
        estado += (f"\n- ESCALAR (motivo: {escalar}): llama escalar_a_humano ahora con un resumen de lo que el "
                   "usuario necesita; luego dile con empatía que un analista continuará y da el número de caso.")
    if es_documento(s):
        base = f"{SISTEMA_DOC}\n\n{s.fuente['doc'].ficha()}"
    else:
        base = f"{SISTEMA}\n\n{datos_ips.PROMPT_DATOS}"
        if s.fuente:
            base += (f"\n- FUENTE ACTIVA: el archivo {s.fuente['nombre']} que cargó el usuario, con las mismas columnas "
                     "del dataset; responde con él, no con la API.")
    sistema = f"{base}\n\n{estado}"
    return [{"role": "system", "content": sistema}, *s.historial[-6:], {"role": "user", "content": pregunta}]


def respuesta_inmediata(s: Sesion, pregunta: str):
    """Saludo, gracias, despedida o '¿cómo estás?': plantilla corta, sin llamar al modelo."""
    p = plano(pregunta)
    if not p or not INMEDIATA.fullmatch(p):
        return None
    cats = [c for c, rx in CATEGORIAS if rx.search(p)]
    if not cats:
        return None
    c = cats[0]
    n = s.sociales.get(c, 0)
    s.sociales[c] = n + 1
    texto = PLANTILLAS[c][n % len(PLANTILLAS[c])]
    if c == "despedida" and "gracias" in cats:
        texto = "Con gusto, hasta luego."
    elif c == "como" and "saludo" in cats:
        texto = "Hola, " + texto[0].lower() + texto[1:]
    elif c == "saludo":
        hora = next((v for k, v in SALUDO_HORA.items() if k in p), None)
        texto = f"{hora}, ¿en qué puedo ayudarte?" if hora else texto
    return texto


def ruta(pregunta, escalar, s: Sesion | None = None):
    """Enrutador: charla social -> modelo rápido sin herramientas; lo demás -> modelo con las herramientas de la fuente."""
    if not escalar and SOCIAL.fullmatch(plano(pregunta)):
        return "social", llm.MODELO_RAPIDO, None
    return "herramientas", llm.MODELO, herramientas(s) + [TOOL_ESCALAR]


def estado_resultado(res):
    if res.get("error"):
        return "error"
    if res.get("sospechoso"):
        return "sospechoso"
    return "ok"


def resumen_resultado(res, limite=220):
    if res.get("error"):
        return f"{res.get('error')} · sugerencias: {', '.join(map(str, (res.get('sugerencias') or [])[:3]))}"
    if isinstance(res.get("resultados"), list):  # fragmentos del documento
        return " · ".join(f"{r['ubicacion'] or 'texto'}: {r['texto'][:50]}…" for r in res["resultados"][:3])[:limite]
    clave = next((k for k in ("resultado", "ips", "valores", "id") if k in res), None)
    txt = json.dumps(res.get(clave) if clave else res, ensure_ascii=False, default=str)
    if res.get("sospechoso"):
        txt += f" · sospechoso: {res.get('recomendacion', '')}"
    return txt[:limite]


# ---------------- ciclo de decisión ----------------

def _fin(t0, s: Sesion, modelo, primer_token=None, consultas=0, exitos=0):
    fin = time.perf_counter()
    return {"t": "fin", "x": {"ms_total": int((fin - t0) * 1000),
                              "ms_primer_token": int(((primer_token or fin) - t0) * 1000),
                              "modelo": modelo, "consultas": consultas, "exitos": exitos, "fallos_seguidos": s.fallos}}


async def responder(sid: str, pregunta: str, modelo: str | None = None):
    """Genera eventos: decision, consultando, paso, escalamiento, delta, silencio, fin.
    No espera el análisis emocional del turno: usa la última emoción ya disponible."""
    s = sesion(sid)
    t0 = time.perf_counter()

    if pide_silencio(pregunta):
        yield {"t": "silencio"}
        yield _fin(t0, s, "ninguno")
        return

    inmediata = respuesta_inmediata(s, pregunta)
    if inmediata:
        yield {"t": "decision", "x": {"ruta": "instantánea", "modelo": "plantilla", "escalar": None,
                                      "emocion": estado_emocional(s), "contexto": dict(s.entidades)}}
        yield {"t": "delta", "x": inmediata}
        s.historial += [{"role": "user", "content": pregunta}, {"role": "assistant", "content": inmediata}]
        s.historial[:] = s.historial[-12:]
        yield _fin(t0, s, "plantilla", primer_token=time.perf_counter())
        return

    escalar = motivo_escalar(s, pregunta)
    tipo_ruta, modelo_usado, tools = ruta(pregunta, escalar, s)
    if modelo and tools:
        modelo_usado = modelo
    mensajes = construir_mensajes(s, pregunta, escalar)
    yield {"t": "decision", "x": {"ruta": tipo_ruta, "modelo": modelo_usado, "escalar": escalar,
                                  "emocion": estado_emocional(s), "contexto": dict(s.entidades),
                                  "fuente": s.fuente["nombre"] if s.fuente else "API datos.gov.co",
                                  "fuente_tipo": s.fuente["tipo"] if s.fuente else "api"}}

    texto, primer_token, consultas, exitos = "", None, 0, 0
    for paso in range(1, MAX_PASOS + 2):
        forzar_cierre = paso > MAX_PASOS
        llamadas = None
        try:
            async for tipo, x in llm.stream(mensajes, None if forzar_cierre else tools, modelo=modelo_usado):
                if tipo == "texto":
                    primer_token = primer_token or time.perf_counter()
                    texto += x
                    yield {"t": "delta", "x": re.sub(r"[*#_`>|]+", "", x)}
                elif tipo == "tools":
                    llamadas = x
        except Exception as e:
            # El proveedor rechazó la llamada (p. ej. parámetros inválidos): el agente se corrige y reintenta.
            msg = str(e)
            if "tool" not in msg.lower() or forzar_cierre or texto:
                raise
            yield {"t": "paso", "x": {"n": paso, "herramienta": "(llamada rechazada)", "args": {}, "estado": "error",
                                      "resultado": "Parámetros inválidos: el agente reformula la consulta"}}
            mensajes.append({"role": "system", "content": "Tu llamada anterior a la herramienta tenía parámetros "
                             "inválidos. Vuelve a intentarlo usando solo los parámetros del esquema, con valores de texto simples."})
            continue
        if not llamadas:
            break
        # Aviso inmediato: la interfaz dice una frase de relleno mientras se consulta.
        yield {"t": "consultando", "x": {"herramientas": [c["nombre"] for c in llamadas]}}
        mensajes.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": c["id"], "type": "function", "function": {"name": c["nombre"], "arguments": c["args"] or "{}"}}
            for c in llamadas]})
        for c in llamadas:
            try:
                args = json.loads(c["args"] or "{}")
            except json.JSONDecodeError:
                args = {}
            if c["nombre"] == "escalar_a_humano":
                res = crear_ticket(s, args, estado_emocional(s))
                yield {"t": "escalamiento", "x": res}
                estado = "ok"
            else:
                res = await asyncio.to_thread(ejecutar_herramienta, s, c["nombre"], args)
                consultas += 1
                estado = estado_resultado(res)
                exitos += estado == "ok"
                actualizar_entidades(s, res)
                if estado == "error":
                    res = {**res, "_instruccion": "Corrige: reintenta con la sugerencia más parecida o usa listar_valores."}
                elif estado == "sospechoso":
                    res = {**res, "_instruccion": "Aplica la recomendacion y decláralo en la respuesta."}
            yield {"t": "paso", "x": {"n": paso, "herramienta": c["nombre"], "args": args, "estado": estado,
                                      "resultado": resumen_resultado(res)}}
            mensajes.append({"role": "tool", "tool_call_id": c["id"],
                             "content": json.dumps(res, ensure_ascii=False, default=str)[:6000 if es_documento(s) else 3500]})
        if paso == MAX_PASOS:
            # Cierre forzado: se pasan los resultados como texto plano para que el modelo no intente otra herramienta.
            resultados = [m["content"] for m in mensajes if m["role"] == "tool"][-4:]
            mensajes = [mensajes[0], {"role": "user", "content": (
                f"PREGUNTA: {pregunta}\n\nRESULTADOS DE LAS CONSULTAS YA HECHAS:\n" + "\n".join(resultados) +
                "\n\nYa no hay más consultas. Responde ahora en voz, con honestidad, usando solo estos resultados.")}]

    if consultas:
        s.fallos = 0 if exitos else s.fallos + 1
    if tiene_contenido(texto):  # "..." o vacío no entra al historial
        s.historial += [{"role": "user", "content": pregunta}, {"role": "assistant", "content": texto[:800]}]
        s.historial[:] = s.historial[-12:]
    yield _fin(t0, s, llm.ULTIMO_MODELO["nombre"], primer_token, consultas, exitos)
