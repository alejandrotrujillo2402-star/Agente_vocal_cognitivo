"""Análisis de emoción local por reglas (léxico en español), para no gastar tokens del LLM en cada intervención.
Devuelve el mismo formato que el análisis con LLM más "confianza": baja cuando no hay ninguna señal clara.
"""
import re
import unicodedata

EMOCIONES = ["alegria", "confianza", "interes", "sorpresa", "confusion", "frustracion", "enojo", "tristeza", "miedo"]

# Raíces o frases, sin tildes y en minúscula. Una raíz corta ("molest") cubre molesto, molesta, molestia...
LEXICO = {
    "enojo": ["rabia", "furios", "molest", "harto", "harta", "odio", "maldit", "estupid", "idiota", "inutil", "basura",
              "porqueria", "fastidi", "carajo", "mierda", "joder", "pesim", "indignad", "absurd", "ridicul", "verguenza",
              "me tiene mamad", "que jartera", "berraco", "emputad", "callate ya"],
    "frustracion": ["no sirve", "no funciona", "no me respond", "no respondes", "no me entiend", "no entiendes", "otra vez",
                    "ya te pregunte", "ya te dije", "te lo dije", "no me ayuda", "no ayuda", "siempre lo mismo", "no puede ser",
                    "cansad", "ya van", "perdiendo el tiempo", "perder el tiempo", "no me sirve", "equivoca", "mal hecho",
                    "nada que ver", "sigues sin", "por que no", "de nuevo lo mismo", "no es lo que", "eso no fue lo que"],
    "tristeza": ["triste", "lamentabl", "que pena", "desanim", "deprim", "llorar", "dolor", "murio", "fallecio", "perdi a",
                 "decepcion", "desilusion", "solo me queda"],
    "alegria": ["genial", "excelente", "perfecto", "buenisim", "me encanta", "feliz", "que bien", "chever", "bacan", "super bien",
                "maravill", "muy bien", "gracias", "que bueno", "fantastic", "estupendo", "jaja"],
    "miedo": ["miedo", "asustad", "preocup", "nervios", "ansios", "urgente", "emergencia", "grave", "peligro", "temor",
              "angusti", "auxilio", "ayuda por favor"],
    "sorpresa": ["wow", "increible", "en serio", "no puedo creer", "de verdad", "sorprend", "impresionante", "no sabia", "uy "],
    "confusion": ["no entiendo", "no entendi", "confund", "que significa", "como asi", "no comprendo", "a que te refieres",
                  "no me queda claro", "no se que", "explicame", "que quieres decir"],
    "confianza": ["confio", "de acuerdo", "entendido", "tiene sentido", "me sirve", "me quedo claro", "claro que si"],
}
NEGACION = re.compile(r"\b(no|nunca|tampoco|ni|sin)\s+(estoy|estaba|me|es|era|siento|tan|muy)?\s*$")
INTENSIFICA = re.compile(r"\b(muy|demasiad\w*|tan|super|totalmente|nada|bastante|re|hiper)\b")
INTERROG = re.compile(r"\b(que|cuant\w*|cual\w*|donde|como|quien\w*|cuando)\b")
POSITIVAS, NEGATIVAS = ("alegria", "confianza"), ("frustracion", "enojo", "tristeza", "miedo")


def plano(t):
    t = unicodedata.normalize("NFKD", str(t).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " " + " ".join(re.sub(r"[^a-z0-9ñ ]+", " ", t).split()) + " "


def palabras(t):
    return len(re.findall(r"[^\W\d_]+", str(t)))


def analizar(texto):
    p, crudo = plano(texto), str(texto)
    e = dict.fromkeys(EMOCIONES, 0.0)
    pistas = 0
    for emo, lista in LEXICO.items():
        for raiz in lista:
            for m in re.finditer(r"(?<![a-z])" + re.escape(raiz), p):
                previo = p[max(0, m.start() - 22):m.start()]
                # "no sirve", "no entiendo" ya son frases negativas; "no estoy molesto" invierte la palabra
                if not raiz.startswith(("no ", "ni ")) and NEGACION.search(previo):
                    continue
                e[emo] += 0.5 + (0.15 if INTENSIFICA.search(previo[-12:]) else 0)
                pistas += 1
    mayus = sum(1 for w in re.findall(r"\b[A-ZÁÉÍÓÚÑ]{3,}\b", crudo))
    excl = crudo.count("!")
    if mayus >= 2 or excl >= 2:   # gritar: más enojo/frustración si ya había señal; si no, sorpresa
        if e["enojo"] or e["frustracion"]:
            e["enojo" if e["enojo"] >= e["frustracion"] else "frustracion"] += 0.2
        else:
            e["sorpresa"] += 0.3
        pistas += 1
    pregunta = "?" in crudo or bool(INTERROG.search(p[:20]))
    if pregunta:
        e["interes"] += 0.35   # una pregunta informativa: neutral con interés moderado
    e = {k: round(min(1.0, v), 2) for k, v in e.items()}

    pos = sum(e[k] for k in POSITIVAS) + 0.2 * e["sorpresa"]
    neg = sum(e[k] for k in NEGATIVAS) + 0.5 * e["confusion"]
    polaridad = round(max(-1.0, min(1.0, pos - neg)), 2)
    sentimiento = "positivo" if polaridad > 0.25 else "negativo" if polaridad < -0.25 else "neutral"
    top = max((k for k in EMOCIONES if k != "interes"), key=e.get)
    dominante = top if e[top] >= 0.4 else "interes" if e["interes"] >= 0.4 else "neutral"
    # confianza: alta con señales claras o con una pregunta llana; baja si es un texto largo sin ninguna pista
    confianza = 0.85 if pistas else 0.8 if pregunta else 0.75 if palabras(texto) <= 6 else 0.35
    return {"sentimiento": sentimiento, "polaridad": polaridad, "emociones": e, "dominante": dominante,
            "confianza": confianza, "origen": "local"}
