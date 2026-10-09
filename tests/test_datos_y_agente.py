"""Pruebas sin red: datos sintéticos con las trampas reales del dataset y LLM falso."""
import json

import pytest
from fastapi.testclient import TestClient

import agente
import app as api
import datos_ips
import llm

E = datos_ips.ejecutar


# ---------- capa de datos ----------

def test_prestadores_vs_sedes():
    assert E("consultar_estadistica", {"metrica": "num_prestadores", "filtros": {"departamento": "Caldas"}})["resultado"] == 3
    assert E("consultar_estadistica", {"metrica": "num_sedes", "filtros": {"municipio": "manizales"}})["resultado"] == 2


def test_sinonimos_uci_y_distritos():
    r = E("consultar_estadistica", {"metrica": "suma_capacidad", "filtros": {"departamento": "valle del cauca", "descripcion_capacidad": "uci adultos"}})
    assert r["resultado"] == 90 and any("Cali" in n or "cali" in n for n in r["notas"])
    r = E("consultar_estadistica", {"metrica": "suma_capacidad", "filtros": {"municipio": "Manizales", "descripcion_capacidad": "UCI adultos"}})
    assert r["resultado"] == 35


def test_valor_mal_escrito_devuelve_sugerencias():
    r = E("consultar_estadistica", {"metrica": "num_sedes", "filtros": {"departamento": "Caldaz"}})
    assert r.get("resultado") == 3 or (r["error"] == "valor_no_encontrado" and "caldas" in r["sugerencias"])
    r = E("consultar_estadistica", {"metrica": "num_sedes", "filtros": {"municipio": "Narnia"}})
    assert r["error"] == "valor_no_encontrado"


def test_nivel_en_privadas_es_sospechoso():
    r = E("consultar_estadistica", {"metrica": "num_prestadores", "filtros": {"naturaleza": "Privada", "nivel_atencion": "3"}})
    assert r["resultado"] == 0 and r["sospechoso"] and "nivel" in r["recomendacion"]


def test_buscar_ips_tolerante():
    r = E("buscar_ips", {"nombre": "santa sofía"})
    assert r["resultado"][0]["capacidad"]["CAMAS Adultos"] == 120
    assert E("buscar_ips", {"nombre": "hospital inexistente xyz"})["error"] == "ips_no_encontrada"


def test_ejecutar_nunca_lanza():
    assert "error" in E("no_existe", {})
    assert "error" in E("consultar_estadistica", {"parametro_raro": 1})


def test_brief_y_casos_de_evaluacion():
    b = datos_ips.brief()
    assert len(b["preguntas"]) == 5 and b["stats"]["prestadores"] == 5
    assert all("pregunta" in c and "esperado" in c for c in datos_ips.preguntas_evaluacion())


# ---------- agente ----------

def guion(*respuestas):
    """LLM falso: cada llamada consume la siguiente respuesta (texto o llamadas a herramientas)."""
    cola = list(respuestas)
    vistos = []

    async def stream(messages, tools=None, modelo=None):
        vistos.append({"messages": [dict(m) for m in messages], "tools": tools, "modelo": modelo})
        r = cola.pop(0)
        if isinstance(r, list):
            yield "tools", [{"id": f"t{i}", "nombre": n, "args": json.dumps(a)} for i, (n, a) in enumerate(r)]
        else:
            yield "texto", r
    return stream, vistos


async def correr(sid, pregunta):
    return [ev async for ev in agente.responder(sid, pregunta)]


@pytest.mark.anyio
async def test_se_corrige_ante_error(monkeypatch):
    stream, vistos = guion([("consultar_estadistica", {"metrica": "num_sedes", "filtros": {"municipio": "Narnia"}})],
                           [("consultar_estadistica", {"metrica": "num_sedes", "filtros": {"municipio": "Manizales"}})],
                           "En Manizales hay 2 sedes.")
    monkeypatch.setattr(llm, "stream", stream)
    ev = await correr("a1", "¿Cuántas sedes hay en Narnia o Manizales?")
    pasos = [e["x"] for e in ev if e["t"] == "paso"]
    assert [p["estado"] for p in pasos] == ["error", "ok"]
    assert "".join(e["x"] for e in ev if e["t"] == "delta") == "En Manizales hay 2 sedes."
    assert agente.sesion("a1").entidades["municipio"] == "manizales"
    assert ev[-1]["x"]["exitos"] == 1 and agente.sesion("a1").fallos == 0


@pytest.mark.anyio
async def test_contexto_activo_y_emocion_entran_al_prompt(monkeypatch):
    s = agente.sesion("a2")
    s.entidades["municipio"] = "manizales"
    agente.registrar_emocion(s, {"sentimiento": "negativo", "dominante": "frustracion", "emociones": {"frustracion": 0.8}})
    stream, vistos = guion("Claro.")
    monkeypatch.setattr(llm, "stream", stream)
    await correr("a2", "¿y cuántas ambulancias?")
    sistema = vistos[0]["messages"][0]["content"]
    assert "municipio=manizales" in sistema and "frustracion" in sistema


@pytest.mark.anyio
async def test_escala_cuando_lo_piden(monkeypatch):
    stream, vistos = guion([("escalar_a_humano", {"motivo": "solicitud_usuario", "resumen": "Quiere camas UCI", "pendiente": "cifra"})],
                           "Un analista continuará, tu caso es ESC.")
    monkeypatch.setattr(llm, "stream", stream)
    ev = await correr("a3", "Quiero hablar con una persona")
    assert "ESCALAR" in vistos[0]["messages"][0]["content"]
    t = [e for e in ev if e["t"] == "escalamiento"][0]["x"]
    assert t["id"].startswith("ESC-") and agente.ESCALAMIENTOS[-1] is t
    stream2, vistos2 = guion("ok")
    monkeypatch.setattr(llm, "stream", stream2)
    await correr("a3", "otra cosa con un asesor")
    assert "ya fue escalado" in vistos2[0]["messages"][0]["content"]


@pytest.mark.anyio
async def test_escala_por_frustracion_sostenida(monkeypatch):
    s = agente.sesion("a4")
    for _ in range(2):
        agente.registrar_emocion(s, {"emociones": {"frustracion": 0.7}})
    assert agente.motivo_escalar(s, "¿cuántas camas?") == "frustracion"


@pytest.mark.anyio
async def test_dos_turnos_fallidos_escalan(monkeypatch):
    for _ in range(2):
        stream, _ = guion([("buscar_ips", {"nombre": "zzzz qqqq"})], "No la encuentro.")
        monkeypatch.setattr(llm, "stream", stream)
        await correr("a5", "datos de la clínica zzzz")
    assert agente.sesion("a5").fallos == 2
    assert agente.motivo_escalar(agente.sesion("a5"), "y entonces?") == "sin_respuesta"


@pytest.mark.anyio
async def test_enrutador_social_sin_herramientas(monkeypatch):
    stream, vistos = guion("¡Hola! Pregúntame por IPS.")
    monkeypatch.setattr(llm, "stream", stream)
    ev = await correr("a6", "Hola, buenos días")
    assert vistos[0]["tools"] is None and vistos[0]["modelo"] == llm.MODELO_RAPIDO
    assert ev[0]["x"]["ruta"] == "social"


@pytest.mark.anyio
async def test_limite_de_pasos_fuerza_respuesta(monkeypatch):
    malas = [[("listar_valores", {"columna": "municipio", "contiene": "x"})]] * agente.MAX_PASOS
    stream, vistos = guion(*malas, "No pude encontrarlo.")
    monkeypatch.setattr(llm, "stream", stream)
    ev = await correr("a7", "algo raro")
    assert vistos[-1]["tools"] is None
    assert ev[-1]["t"] == "fin"


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---------- API ----------

cliente = TestClient(api.app)


def test_endpoints_basicos(monkeypatch):
    assert cliente.get("/").status_code == 200
    assert cliente.get("/api/estado").json()["filas"] == 7
    assert cliente.get("/api/brief").json()["titulo"]
    stream, _ = guion("Hay 2 sedes.")
    monkeypatch.setattr(llm, "stream", stream)
    r = cliente.post("/api/preguntar", json={"sesion": "w1", "pregunta": "¿cuántas sedes?"})
    tipos = [json.loads(l)["t"] for l in r.text.splitlines()]
    assert tipos[0] == "decision" and tipos[-1] == "fin"


def test_analizar_guarda_emocion_en_sesion(monkeypatch):
    async def falso(messages, modelo=None):
        return {"sentimiento": "negativo", "polaridad": -0.8, "emociones": {"enojo": 0.9}, "dominante": "enojo"}
    monkeypatch.setattr(llm, "json_rapido", falso)
    cliente.post("/api/analizar", json={"texto": "esto es pésimo", "hablante": "Hablante 1", "sesion": "w2"})
    assert agente.sesion("w2").emociones[-1]["dominante"] == "enojo"


def test_para_voz_lee_cifras():
    assert api.para_voz("Hay 41.427 registros y 12,5%") == "Hay cuarenta y un mil cuatrocientos veintisiete registros y doce coma cinco por ciento"
