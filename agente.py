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

MAX_PASOS = 3
MAX_TOKENS = 220    # tope de la respuesta hablada (llamada sin herramientas)
MAX_TOKENS_HERR = 600  # llamada con herramientas: razonamiento + argumentos; con 220 la llamada salía cortada
HISTORIAL = 4       # mensajes recientes que van completos al modelo (2 intercambios); lo anterior va resumido
CACHE_TTL = 30 * 60  # respuestas repetidas: se reutilizan 30 min sin llamar al modelo
NEGATIVAS = ("frustracion", "enojo", "confusion")
MAX_ARCHIVOS = 4   # sesiones con archivo propio en memoria; al pasar el límite, la más antigua vuelve a la API

HABLA = """Voz: máximo 2 frases cortas (más solo si piden detalle), sin markdown ni listas, cifras con dígitos.
Con frustración o confusión: reconoce breve, simplifica, ofrece un camino. Con enojo: no discutas.
Si hay ESCALAR: llama escalar_a_humano y da el número de caso una vez. Fragmento incompleto: pide en pocas palabras que complete."""

SISTEMA = f"""Eres el agente de voz de datos abiertos de salud de Colombia. Única fuente: el dataset de IPS de datos.gov.co, solo vía herramientas.
Reglas:
1. Toda cifra o dato sale de una herramienta; nunca inventes números.
2. Con "error"/"sugerencias", reintenta tú con la sugerencia más parecida. Con "sospechoso", aplica la "recomendacion" y dilo.
3. Sin datos tras intentar: "Eso no aparece en el dataset" y ofrece algo que sí.
4. Ambigua: elige lo razonable y dilo ("conté sedes, no prestadores"); pregunta solo si las lecturas difieren mucho.
5. Lo que no está (médicos, EPS, precios, calidad, citas): dilo sin herramientas.
6. Usa el CONTEXTO ACTIVO para "¿y en Pereira?".
{HABLA}"""

SISTEMA_DOC = f"""Eres un agente de voz que responde sobre un documento que cargó el usuario. Única fuente: ese documento, solo vía herramientas.
Reglas:
1. Busca con buscar_en_documento (palabras clave); si no hay, reintenta una vez con sinónimos o los términos sugeridos.
2. Responde solo con los fragmentos; si traen "pagina", menciónala ("según la página 4"); nunca hables de fragmentos ni filas ni uses conocimiento general.
3. Contar, sumar, promediar o comparar en tablas: consultar_tabla.
4. Si no lo dice: "Eso no aparece en el documento" y ofrece algo que sí.
{HABLA}"""

TOOL_ESCALAR = {
    "type": "function",
    "function": {
        "name": "escalar_a_humano",
        "description": "Pasa el caso a un analista humano con el contexto.",
        "parameters": {
            "type": "object",
            "properties": {
                "motivo": {"type": "string", "enum": ["frustracion", "sin_respuesta", "solicitud_usuario"]},
                "resumen": {"type": "string"},
                "pendiente": {"type": "string"},
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


def limpiar_args(v):
    """El modelo a veces escribe mal el nombre de un campo ("descripcion_capacidad ", "grupo_capacidad**: "):
    se normaliza para que el filtro no se pierda en silencio."""
    if isinstance(v, dict):
        return {re.sub(r"[^a-z0-9_]", "", str(k).lower()): limpiar_args(x) for k, x in v.items()}
    return v


def ejecutar_herramienta(s: Sesion, nombre, args):
    args = limpiar_args(args)
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
    estado = f"ESTADO\n- CONTEXTO ACTIVO: {activo}\n- EMOCIÓN del usuario: {estado_emocional(s)}"
    if s.fallos:
        estado += f"\n- Turnos seguidos sin respuesta válida: {s.fallos}"
    # lo anterior a los últimos 2 intercambios va en una línea: solo las preguntas, recortadas
    previas = [m["content"][:60] for m in s.historial[:-HISTORIAL] if m["role"] == "user"][-3:]
    if previas:
        estado += "\n- Antes preguntó: " + " | ".join(previas)
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
    recientes = [{"role": m["role"], "content": m["content"][:300]} for m in s.historial[-HISTORIAL:]]
    return [{"role": "system", "content": sistema}, *recientes, {"role": "user", "content": pregunta}]


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


FUERA_ALCANCE = re.compile(r"\b(medic[oa]s?|doctor(es)?|especialista|especialidad|eps|afiliad\w*|precio|cuesta|costo|valor de la consulta|"
                           r"citas?|calidad|ocupacion|enfermer[oa]s?)\b")
BUSCA_IPS = re.compile(r"\b(telefono|direccion|gerente|nit|contacto|ficha|donde queda|ubicad[oa])\b")
CUENTA = re.compile(r"\b(cuant\w*|total|cuales? (son|es) los?|mas|menos|top|ranking|compar\w*|promedio|suma)\b")


def herramientas_relevantes(pregunta, s: Sesion | None):
    """Solo los esquemas que la pregunta puede usar: cada esquema enviado cuesta tokens en cada llamada."""
    todas = herramientas(s)
    if es_documento(s):
        return todas
    p, por_nombre = plano(pregunta), {t["function"]["name"]: t for t in todas}
    if FUERA_ALCANCE.search(p) and not BUSCA_IPS.search(p):
        nombres = ["info_dataset"]
    elif BUSCA_IPS.search(p) or (not CUENTA.search(p) and re.search(r"\b(hospital|clinica|ese|fundacion|ips)\b", p)):
        nombres = ["buscar_ips", "listar_valores"]
    elif CUENTA.search(p):
        nombres = ["consultar_estadistica", "listar_valores"]
    else:
        return todas
    return [por_nombre[n] for n in nombres if n in por_nombre]


def pide_datos(pregunta, s: Sesion | None):
    """Pregunta que necesita una cifra o un dato de la fuente (no social, no fuera de alcance)."""
    p = plano(pregunta)
    if es_documento(s):
        return bool(INTERROGATIVA.search(pregunta.lower()) or "?" in pregunta)
    return bool((CUENTA.search(p) or BUSCA_IPS.search(p) or re.match(r"y (en|el|la|los|las) ", p)) and not FUERA_ALCANCE.search(p))


def ruta(pregunta, escalar, s: Sesion | None = None):
    """Enrutador: charla social -> modelo rápido sin herramientas; lo demás -> modelo con las herramientas relevantes."""
    if not escalar and SOCIAL.fullmatch(plano(pregunta)):
        return "social", llm.MODELO_RAPIDO, None
    return "herramientas", llm.MODELO, herramientas_relevantes(pregunta, s) + [TOOL_ESCALAR]


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


def _limpio(v, n=10):
    """Sin nulos ni fecha de corte (ya está en la ficha), listas de máximo n elementos y decimales redondeados."""
    if isinstance(v, dict):
        return {k: _limpio(x, n) for k, x in v.items() if x not in (None, "", [], {}) and k not in ("fecha_corte", "filtros_aplicados")}
    if isinstance(v, list):
        return [_limpio(x, n) for x in v[:n]]
    if isinstance(v, float):
        return int(v) if v.is_integer() else round(v, 2)
    return v


def compactar(res):
    """Resultado de herramienta para el modelo: JSON sin espacios, máximo 10 filas."""
    return json.dumps(_limpio(res), ensure_ascii=False, default=str, separators=(",", ":"))


# ---------------- caché de respuestas ----------------

CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_LUGARES = {"id": None, "nombres": ()}


def _lugares():
    """Departamentos y municipios del dataset activo (normalizados), para saber si la pregunta nombra un lugar."""
    df = datos_ips._df()
    if _LUGARES["id"] != id(df):
        nombres = set(df["departamento_n"].unique()) | set(df["municipio_n"].unique())
        _LUGARES.update(id=id(df), nombres=tuple(sorted((n for n in nombres if n and len(n) > 2), key=len, reverse=True)))
    return _LUGARES["nombres"]


def clave_cache(s: Sesion, pregunta: str):
    """Pregunta normalizada + fuente + contexto activo (solo si la pregunta no nombra su propio lugar: '¿y en Pereira?')."""
    p = plano(pregunta)
    fuente = s.fuente["nombre"] if s.fuente else "api"
    if es_documento(s):
        return (fuente, p, "")
    with datos_de(s):
        nombra = any(f" {n} " in f" {p} " for n in _lugares())
        fuente += f"#{id(datos_ips._df())}"   # datos nuevos (recarga de la API, otro archivo): caché nueva
    ctx = "" if nombra else "|".join(f"{k}={s.entidades[k]}" for k in ("departamento", "municipio") if s.entidades.get(k))
    return (fuente, p, ctx)


def leer_cache(clave):
    r = CACHE.get(clave)
    if r and time.time() - r["t"] < CACHE_TTL:
        return r
    CACHE.pop(clave, None)
    return None


def guardar_cache(clave, texto, entidades):
    CACHE[clave] = {"t": time.time(), "texto": texto, "entidades": dict(entidades)}
    while len(CACHE) > 300:
        CACHE.popitem(last=False)


# ---------------- comandos sin modelo: repetir y "no te entendí" ----------------

REPITE = re.compile(r"(?:\b(?:repite(?:me)?|repitelo|repitemelo|otra vez|de nuevo|puedes repetir|me repites|por favor|lo|eso)\b\s*)+")
NO_ENTENDI = re.compile(r"(?:\b(?:no te entendi|no entendi|no te escuche|no escuche|no te oi|que dijiste|como dijiste|perdon|disculpa|"
                        r"no te entiendo|no se escucho|por favor)\b\s*)+")


def comando_repetir(s: Sesion, pregunta: str):
    p = plano(pregunta)
    ultima = next((m["content"] for m in reversed(s.historial) if m["role"] == "assistant"), None)
    if not p or not ultima:
        return None
    if REPITE.fullmatch(p) and re.search(r"repit|otra vez|de nuevo", p):
        return ultima
    if NO_ENTENDI.fullmatch(p) and not re.fullmatch(r"(por favor|perdon|disculpa)( (por favor|perdon|disculpa))*", p):
        return "Te lo repito: " + ultima
    return None


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
    cuenta = {}   # tokens de esta pregunta (todas las llamadas del ciclo)
    peticion = {"tipo": "pregunta", "cuenta": cuenta, "max_tokens": MAX_TOKENS}
    llm.PETICION.set(peticion)

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

    def sin_modelo(texto, ruta_, modelo_):
        """Respuesta sin llamar al modelo (repetir, caché): mismo flujo de eventos que una respuesta normal."""
        yield {"t": "decision", "x": {"ruta": ruta_, "modelo": modelo_, "escalar": None,
                                      "emocion": estado_emocional(s), "contexto": dict(s.entidades)}}
        yield {"t": "delta", "x": texto}
        s.historial += [{"role": "user", "content": pregunta}, {"role": "assistant", "content": texto[:800]}]
        s.historial[:] = s.historial[-12:]
        llm.registrar_peticion("pregunta", pregunta, cuenta)
        fin = _fin(t0, s, modelo_, primer_token=time.perf_counter())
        fin["x"]["tokens"] = {"prompt": 0, "completion": 0}
        yield fin

    repetir = comando_repetir(s, pregunta)
    if repetir:
        for ev in sin_modelo(repetir, "instantánea", "plantilla"):
            yield ev
        return

    escalar = motivo_escalar(s, pregunta)
    clave = clave_cache(s, pregunta)
    guardada = None if (escalar or s.ticket) else leer_cache(clave)
    if guardada:
        s.entidades.update(guardada["entidades"])
        for ev in sin_modelo(guardada["texto"], "caché", "caché"):
            yield ev
        return
    tipo_ruta, modelo_usado, tools = ruta(pregunta, escalar, s)
    if modelo and tools:
        modelo_usado = modelo
    mensajes = construir_mensajes(s, pregunta, escalar)
    yield {"t": "decision", "x": {"ruta": tipo_ruta, "modelo": modelo_usado, "escalar": escalar,
                                  "emocion": estado_emocional(s), "contexto": dict(s.entidades),
                                  "fuente": s.fuente["nombre"] if s.fuente else "API datos.gov.co",
                                  "fuente_tipo": s.fuente["tipo"] if s.fuente else "api"}}

    texto, primer_token, consultas, exitos = "", None, 0, 0
    # pregunta de datos: la primera llamada DEBE usar una herramienta (nunca una cifra de memoria)
    exige = bool(tools) and pide_datos(pregunta, s)
    herr_paso = tools
    for paso in range(1, MAX_PASOS + 2):
        forzar_cierre = paso > MAX_PASOS
        peticion["tool_choice"] = "required" if exige and paso == 1 else None
        peticion["max_tokens"] = MAX_TOKENS if forzar_cierre or not herr_paso else MAX_TOKENS_HERR
        llamadas = None
        try:
            async for tipo, x in llm.stream(mensajes, None if forzar_cierre else herr_paso, modelo=modelo_usado):
                if tipo == "texto":
                    primer_token = primer_token or time.perf_counter()
                    texto += x
                    yield {"t": "delta", "x": re.sub(r"[*#_`>|]+", "", x)}
                elif tipo == "tools":
                    llamadas = x
        except Exception as e:
            # El proveedor rechazó la llamada (p. ej. parámetros inválidos): el agente se corrige y reintenta.
            msg = str(e)
            # "tool_use_failed" / "Failed to call a function": la llamada salió mal formada; se reintenta el paso
            if not re.search(r"tool|function", msg, re.I) or forzar_cierre or texto:
                raise
            exige = False   # el reintento no obliga a usar herramienta
            yield {"t": "paso", "x": {"n": paso, "herramienta": "(llamada rechazada)", "args": {}, "estado": "error",
                                      "resultado": "Parámetros inválidos: el agente reformula la consulta"}}
            mensajes.append({"role": "system", "content": "Tu llamada anterior a la herramienta tenía parámetros "
                             "inválidos. Vuelve a intentarlo usando solo los parámetros del esquema, con valores de texto simples."})
            continue
        if not llamadas:
            break
        estados = []
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
            estados.append(estado)
            mensajes.append({"role": "tool", "tool_call_id": c["id"],
                             "content": compactar(res)[:6000 if es_documento(s) else 2500]})
        # todo salió bien: la siguiente llamada solo redacta, sin reenviar los esquemas de herramientas
        herr_paso = None if estados and all(e == "ok" for e in estados) else tools
        if paso == MAX_PASOS:
            # Cierre forzado: se pasan los resultados como texto plano para que el modelo no intente otra herramienta.
            resultados = [m["content"] for m in mensajes if m["role"] == "tool"][-4:]
            mensajes = [mensajes[0], {"role": "user", "content": (
                f"PREGUNTA: {pregunta}\n\nRESULTADOS DE LAS CONSULTAS YA HECHAS:\n" + "\n".join(resultados) +
                "\n\nYa no hay más consultas. Responde ahora en voz, con honestidad, usando solo estos resultados.")}]

    if consultas:
        s.fallos = 0 if exitos else s.fallos + 1
    # a la caché solo van respuestas con datos válidos (o fuera de alcance sin consultas), nunca un escalamiento
    if tiene_contenido(texto) and tipo_ruta == "herramientas" and not s.ticket and (exitos or not consultas):
        guardar_cache(clave, texto.strip(), s.entidades)
    if tiene_contenido(texto):  # "..." o vacío no entra al historial
        s.historial += [{"role": "user", "content": pregunta}, {"role": "assistant", "content": texto[:800]}]
        s.historial[:] = s.historial[-12:]
    llm.registrar_peticion("pregunta", pregunta, cuenta)
    fin = _fin(t0, s, llm.ULTIMO_MODELO["nombre"], primer_token, consultas, exitos)
    fin["x"]["tokens"] = {"prompt": cuenta.get("prompt_tokens", 0), "completion": cuenta.get("completion_tokens", 0)}
    yield fin
