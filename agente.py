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

SISTEMA = """Eres el agente de voz de datos abiertos de salud de Colombia. Tu ÚNICA fuente es el dataset "Relación de IPS públicas y privadas según nivel de atención y capacidad instalada" de datos.gov.co, al que accedes SOLO con herramientas.

Cómo decides:
1. Para cualquier cifra, nombre o dato del dataset, llama una herramienta. Nunca digas un número que no venga de un resultado.
2. Si un resultado trae "error" o "sugerencias", corrígete tú mismo: vuelve a llamar con la sugerencia más parecida, sin preguntarle al usuario si la corrección es obvia.
3. Si trae "sospechoso": true, aplica la "recomendacion" y di brevemente qué ajustaste.
4. Si tras intentarlo no hay datos, dilo con honestidad ("Eso no aparece en el dataset") y ofrece algo que sí puedes responder.
5. Si la pregunta es ambigua y las interpretaciones dan respuestas muy distintas, pregunta en una frase corta; si no, elige la más razonable y dilo (por ejemplo "conté sedes, no prestadores").
6. Lo que el dataset no contiene (médicos, EPS, precios, calidad, citas, datos posteriores a la fecha de corte): responde que no está, sin llamar herramientas.
7. Usa el CONTEXTO ACTIVO para entender referencias como "¿y en Pereira?" o "esa IPS".

Cómo hablas:
- Es voz: 1 a 3 frases cortas, máximo 60 palabras, español natural, sin markdown, listas ni emojis. Escribe las cifras con dígitos.
- Adapta el tono al ESTADO EMOCIONAL: con frustración o confusión, reconoce en pocas palabras, simplifica y ofrece un camino concreto; con enojo, no discutas ni te justifiques.
- Si aparece la instrucción ESCALAR, llama escalar_a_humano y luego informa el número de caso una sola vez."""

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
SOCIAL = re.compile(r"^(hola|buenas|buenos dias|buenas tardes|gracias|muchas gracias|chao|adios|hasta luego|ok|listo|perfecto|quien eres|que eres)\b")


def norm(t):
    t = unicodedata.normalize("NFKD", str(t).lower())
    return "".join(c for c in t if not unicodedata.combining(c)).strip(" ¿?¡!.,")


@dataclass
class Sesion:
    historial: list = field(default_factory=list)      # [{"role", "content"}]
    emociones: list = field(default_factory=list)      # análisis de intervenciones humanas
    pendientes: set = field(default_factory=set)       # análisis en curso
    fallos: int = 0                                    # turnos seguidos sin respuesta válida
    entidades: dict = field(default_factory=dict)      # contexto activo
    ticket: dict | None = None


SESIONES: "OrderedDict[str, Sesion]" = OrderedDict()
ESCALAMIENTOS: list = []


def sesion(sid) -> Sesion:
    if sid not in SESIONES:
        SESIONES[sid] = Sesion()
        while len(SESIONES) > 200:
            SESIONES.popitem(last=False)
    SESIONES.move_to_end(sid)
    return SESIONES[sid]


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
    sistema = f"{SISTEMA}\n\n{datos_ips.PROMPT_DATOS}\n\n{estado}"
    return [{"role": "system", "content": sistema}, *s.historial[-6:], {"role": "user", "content": pregunta}]


def ruta(pregunta, escalar):
    """Enrutador: charla social -> modelo rápido sin herramientas; lo demás -> modelo con herramientas."""
    p = norm(pregunta)
    if not escalar and SOCIAL.match(p) and len(p.split()) <= 6:
        return "social", llm.MODELO_RAPIDO, None
    return "herramientas", llm.MODELO, datos_ips.TOOLS + [TOOL_ESCALAR]


def estado_resultado(res):
    if res.get("error"):
        return "error"
    if res.get("sospechoso"):
        return "sospechoso"
    return "ok"


def resumen_resultado(res, limite=220):
    if res.get("error"):
        return f"{res.get('error')} · sugerencias: {', '.join(map(str, (res.get('sugerencias') or [])[:3]))}"
    clave = next((k for k in ("resultado", "ips", "valores", "id") if k in res), None)
    txt = json.dumps(res.get(clave) if clave else res, ensure_ascii=False, default=str)
    if res.get("sospechoso"):
        txt += f" · sospechoso: {res.get('recomendacion', '')}"
    return txt[:limite]


# ---------------- ciclo de decisión ----------------

async def responder(sid: str, pregunta: str, modelo: str | None = None):
    """Genera eventos: decision, paso, escalamiento, delta, fin."""
    s = sesion(sid)
    t0 = time.perf_counter()
    if s.pendientes:  # espera breve al análisis emocional de este turno
        await asyncio.wait(list(s.pendientes), timeout=0.4)

    escalar = motivo_escalar(s, pregunta)
    tipo_ruta, modelo_usado, tools = ruta(pregunta, escalar)
    if modelo and tools:
        modelo_usado = modelo
    mensajes = construir_mensajes(s, pregunta, escalar)
    yield {"t": "decision", "x": {"ruta": tipo_ruta, "modelo": modelo_usado, "escalar": escalar,
                                  "emocion": estado_emocional(s), "contexto": dict(s.entidades)}}

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
                else:
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
                res = await asyncio.to_thread(datos_ips.ejecutar, c["nombre"], args)
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
                             "content": json.dumps(res, ensure_ascii=False, default=str)[:3500]})
        if paso == MAX_PASOS:
            # Cierre forzado: se pasan los resultados como texto plano para que el modelo no intente otra herramienta.
            resultados = [m["content"] for m in mensajes if m["role"] == "tool"][-4:]
            mensajes = [mensajes[0], {"role": "user", "content": (
                f"PREGUNTA: {pregunta}\n\nRESULTADOS DE LAS CONSULTAS YA HECHAS:\n" + "\n".join(resultados) +
                "\n\nYa no hay más consultas. Responde ahora en voz, con honestidad, usando solo estos resultados.")}]

    if consultas:
        s.fallos = 0 if exitos else s.fallos + 1
    s.historial += [{"role": "user", "content": pregunta}, {"role": "assistant", "content": texto[:800]}]
    s.historial[:] = s.historial[-12:]
    fin = time.perf_counter()
    yield {"t": "fin", "x": {"ms_total": int((fin - t0) * 1000),
                             "ms_primer_token": int(((primer_token or fin) - t0) * 1000),
                             "modelo": llm.ULTIMO_MODELO["nombre"], "consultas": consultas, "exitos": exitos,
                             "fallos_seguidos": s.fallos}}
