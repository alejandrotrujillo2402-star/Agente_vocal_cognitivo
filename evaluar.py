"""Evaluación del agente: corre los casos de datos_ips.preguntas_evaluacion() contra varios modelos.
Mide: cifra exacta, herramienta correcta, honestidad (fuera de alcance), recuperación, escalamiento y latencia.

Uso:  python evaluar.py                      # modelos por defecto
      python evaluar.py openai/gpt-oss-120b  # uno solo
Salida: evaluacion/resultados.md y evaluacion/resultados.json
"""
import asyncio
import json
import re
import statistics
import sys
import time
from pathlib import Path

import agente
import datos_ips
from app import analizar_texto

MODELOS = sys.argv[1:] or ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
HONESTO = re.compile(r"no (aparece|esta|está|tengo|contiene|registra|incluye|dispon|cuento|exist|hay (datos|informaci))|no se registra|solo se registra|sólo se registra|fuera de|no forma parte")
EMPATIA = re.compile(r"entiendo|lamento|disculp|comprendo|siento")


def numeros(texto):
    out = set()
    for n in re.findall(r"\d{1,3}(?:[.\s]\d{3})+|\d+", texto):
        out.add(int(re.sub(r"[.\s]", "", n)))
    return out


async def un_caso(modelo, caso):
    sid = f"eval|{modelo}|{caso['id']}"
    agente.SESIONES.pop(sid, None)
    s = agente.sesion(sid)
    if caso["tipo"] == "molesto":
        agente.registrar_emocion(s, await analizar_texto(caso["pregunta"], "Usuario"))
    texto, pasos, escalo, fin, t0, primero = "", [], False, {}, time.perf_counter(), None
    try:
        async for ev in agente.responder(sid, caso["pregunta"], modelo=modelo):
            if ev["t"] == "delta":
                primero = primero or time.perf_counter()
                texto += ev["x"]
            elif ev["t"] == "paso":
                pasos.append(ev["x"])
            elif ev["t"] == "escalamiento":
                escalo = True
            elif ev["t"] == "fin":
                fin = ev["x"]
    except Exception as e:
        texto = f"[ERROR] {e}"
    esp = caso["esperado"]
    usadas = [p["herramienta"] for p in pasos]
    r = {"id": caso["id"], "tipo": caso["tipo"], "pregunta": caso["pregunta"], "respuesta": texto.strip(),
         "herramientas": usadas, "estados": [p["estado"] for p in pasos],
         "latencia_primer_token": round(((primero or time.perf_counter()) - t0), 2),
         "modelo_real": fin.get("modelo", modelo)}
    if esp.get("cifra") == "no_disponible":
        r["ok"] = bool(HONESTO.search(texto.lower()))
        r["criterio"] = "honestidad"
    elif "cifra" in esp:
        r["ok"] = int(esp["cifra"]) in numeros(texto)
        r["criterio"] = f"cifra {esp['cifra']}"
    elif esp.get("escalar"):
        r["ok"] = escalo
        r["criterio"] = "escala a humano"
    r["herramienta_ok"] = (esp.get("herramienta") in usadas) if esp.get("herramienta") else None
    r["empatia"] = bool(EMPATIA.search(texto.lower())) if caso["tipo"] == "molesto" else None
    r["se_recupero"] = (any(e != "ok" for e in r["estados"]) and r["estados"][-1:] == ["ok"]) if r["estados"] else None
    return r


async def main():
    await datos_ips.cargar()
    casos = datos_ips.preguntas_evaluacion()
    salida, tabla = {}, []
    for m in MODELOS:
        res = []
        for c in casos:
            res.append(await un_caso(m, c))
            r = res[-1]
            print(f"{m:22} {c['id']:3} {'OK' if r['ok'] else 'NO'} {r['latencia_primer_token']:>5}s {'>'.join(e[:3] for e in r['estados']) or '-':18} {r['respuesta'][:80]}")
            await asyncio.sleep(8)   # respeta el límite por minuto del proveedor
        salida[m] = res
        pct = lambda xs: f"{100 * sum(xs) / len(xs):.0f}%" if xs else "—"
        tabla.append([m, pct([r["ok"] for r in res]),
                      pct([r["ok"] for r in res if r["tipo"] in ("resumen", "detalle", "trampa") and r["criterio"].startswith("cifra")]),
                      pct([r["herramienta_ok"] for r in res if r["herramienta_ok"] is not None]),
                      pct([r["ok"] for r in res if r["criterio"] == "honestidad"]),
                      pct([r["se_recupero"] for r in res if r["se_recupero"] is not None]),
                      f"{statistics.median([r['latencia_primer_token'] for r in res]):.2f} s"])
    Path("evaluacion").mkdir(exist_ok=True)
    Path("evaluacion/resultados.json").write_text(json.dumps(salida, ensure_ascii=False, indent=1))
    md = ["| Modelo | Aciertos | Cifra exacta | Herramienta correcta | Honestidad | Se recupera | Latencia p50 |",
          "|---|---|---|---|---|---|---|"] + ["| " + " | ".join(f) + " |" for f in tabla]
    Path("evaluacion/resultados.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    asyncio.run(main())
