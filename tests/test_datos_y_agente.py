"""Pruebas sin red: datos sintéticos con las trampas reales del dataset y LLM falso."""
import asyncio
import json
import time

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


def test_sumar_capacidad_sin_grupo_obliga_a_corregir():
    r = E("consultar_estadistica", {"metrica": "suma_capacidad", "filtros": {"municipio": "Manizales"}})
    assert r["error"] == "falta_grupo_capacidad" and "CAMAS" in r["sugerencias"]


def test_esquemas_aceptan_null_en_opcionales():
    p = datos_ips.TOOLS[0]["function"]["parameters"]["properties"]
    assert "null" in p["agrupar_por"]["type"] and "null" in p["filtros"]["properties"]["municipio"]["type"]
    assert E("consultar_estadistica", {"metrica": "num_sedes", "agrupar_por": None, "filtros": {"municipio": None}})["resultado"] == 5


def test_nivel_en_privadas_es_sospechoso():
    r = E("consultar_estadistica", {"metrica": "num_prestadores", "filtros": {"naturaleza": "Privada", "nivel_atencion": "3"}})
    assert r["resultado"] == 0 and r["sospechoso"] and "nivel" in r["recomendacion"]


def test_buscar_ips_tolerante():
    r = E("buscar_ips", {"nombre": "santa sofía"})
    assert r["resultado"][0]["capacidad"]["CAMAS Adultos"] == 120
    assert E("buscar_ips", {"nombre": "hospital inexistente xyz"})["error"] == "ips_no_encontrada"


def test_ejecutar_nunca_lanza():
    assert "error" in E("no_existe", {})
    assert E("consultar_estadistica", {"parametro_raro": 1})["resultado"] == 5   # tolera parámetros extra
    assert E("consultar_estadistica", {"metrica": "num_sedes", "top_n": "3", "filtros": {"nivel_atencion": 3}})["resultado"] == 1


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
async def test_se_recupera_si_el_proveedor_rechaza_la_llamada(monkeypatch):
    llamadas = {"n": 0}

    async def stream(messages, tools=None, modelo=None):
        llamadas["n"] += 1
        if llamadas["n"] == 1:
            raise RuntimeError("Tool call validation failed: parameters for tool consultar_estadistica")
        yield "texto", "Hay 5 sedes."
        
    monkeypatch.setattr(llm, "stream", stream)
    ev = await correr("a8", "¿cuántas sedes hay?")
    assert [e["x"]["estado"] for e in ev if e["t"] == "paso"] == ["error"]
    assert "".join(e["x"] for e in ev if e["t"] == "delta") == "Hay 5 sedes."


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
    stream, vistos = guion("Soy el agente de datos de IPS.")
    monkeypatch.setattr(llm, "stream", stream)
    ev = await correr("a6", "¿Quién eres?")
    assert vistos[0]["tools"] is None and vistos[0]["modelo"] == llm.MODELO_RAPIDO
    assert ev[0]["x"]["ruta"] == "social"


@pytest.mark.anyio
async def test_saludo_con_pregunta_usa_herramientas(monkeypatch):
    stream, vistos = guion("Hay 3 sedes.")
    monkeypatch.setattr(llm, "stream", stream)
    ev = await correr("a9", "Hola, ¿cuántas sedes hay en Caldas?")
    assert ev[0]["x"]["ruta"] == "herramientas" and vistos[0]["tools"]


def sin_llm(monkeypatch):
    """Cualquier llamada al modelo hace fallar la prueba."""
    async def stream(*a, **k):
        raise AssertionError("no debía llamar al LLM")
        yield
    monkeypatch.setattr(llm, "stream", stream)


@pytest.mark.anyio
@pytest.mark.parametrize("frase", ["quédate callado", "cállate por favor", "Silencio.", "no hables más"])
async def test_comando_de_silencio_no_llama_al_llm(monkeypatch, frase):
    sin_llm(monkeypatch)
    ev = await correr("s1", frase)
    assert [e["t"] for e in ev] == ["silencio", "fin"]
    assert agente.sesion("s1").historial == []


@pytest.mark.anyio
async def test_espera_con_pregunta_no_es_silencio(monkeypatch):
    stream, _ = guion("Hay 3 sedes.")
    monkeypatch.setattr(llm, "stream", stream)
    ev = await correr("s2", "Espera, ¿cuántas sedes hay en Caldas?")
    assert "silencio" not in [e["t"] for e in ev]


@pytest.mark.anyio
@pytest.mark.parametrize("frase,esperado", [("hola", "Hola, ¿en qué puedo ayudarte?"), ("gracias", "Con gusto."),
                                            ("buenos días", "Buenos días, ¿en qué puedo ayudarte?"),
                                            ("Muchas gracias, hasta luego", "Con gusto, hasta luego."),
                                            ("Hola, ¿cómo estás?", "Hola, muy bien, gracias. ¿En qué puedo ayudarte?")])
async def test_respuesta_social_instantanea_sin_llm(monkeypatch, frase, esperado):
    sin_llm(monkeypatch)
    t0 = time.perf_counter()
    ev = await correr("i-" + frase, frase)
    assert time.perf_counter() - t0 < 0.05
    assert [e["t"] for e in ev] == ["decision", "delta", "fin"]
    assert ev[0]["x"]["ruta"] == "instantánea" and ev[1]["x"] == esperado


@pytest.mark.anyio
async def test_respuesta_social_varia(monkeypatch):
    sin_llm(monkeypatch)
    textos = [(await correr("i-var", "hola"))[1]["x"] for _ in range(3)]
    assert textos[0] == "Hola, ¿en qué puedo ayudarte?" and len(set(textos)) == 3


@pytest.mark.anyio
async def test_responder_no_espera_el_analisis_emocional(monkeypatch):
    async def lento(messages, modelo=None):
        await asyncio.sleep(2)
        return {}
    monkeypatch.setattr(llm, "json_rapido", lento)
    stream, _ = guion("Hay 5 sedes.")
    monkeypatch.setattr(llm, "stream", stream)
    # texto largo y sin señales claras: el clasificador local no basta y va al LLM (lento aquí)
    analisis = asyncio.create_task(api.analizar(api.Analisis(texto="le cuento esto para un trabajo de la universidad que tengo",
                                                             hablante="Hablante 1", sesion="e1")))
    await asyncio.sleep(0)  # el análisis queda en curso
    t0 = time.perf_counter()
    ev = await correr("e1", "¿cuántas sedes hay?")
    assert time.perf_counter() - t0 < 0.1 and not analisis.done()
    assert "".join(e["x"] for e in ev if e["t"] == "delta") == "Hay 5 sedes."
    analisis.cancel()
    await asyncio.gather(analisis, return_exceptions=True)


@pytest.mark.anyio
async def test_respuesta_vacia_no_entra_al_historial(monkeypatch):
    stream, _ = guion("...")
    monkeypatch.setattr(llm, "stream", stream)
    await correr("v1", "dile que no es que")
    assert agente.sesion("v1").historial == []


@pytest.mark.anyio
async def test_avisa_antes_de_consultar(monkeypatch):
    stream, _ = guion([("consultar_estadistica", {"metrica": "num_sedes", "filtros": {"departamento": "Caldas"}})], "Hay 3 sedes.")
    monkeypatch.setattr(llm, "stream", stream)
    tipos = [e["t"] for e in await correr("c1", "¿Cuántas sedes hay en Caldas?")]
    assert tipos.index("consultando") < tipos.index("paso") < tipos.index("delta")


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


def test_health_informa_stt_tts_y_error(monkeypatch):
    monkeypatch.setitem(api.STT, "error", "HTTP 401 (clave de Deepgram rechazada)")
    h = cliente.get("/health").json()
    assert {"stt", "tts", "stt_error"} <= h.keys() and "401" in h["stt_error"]


def test_motivo_deepgram_legible():
    class Resp:
        status_code, body = 400, b'{"err_msg": "No such model/language/tier combination found."}'

    class Fallo(Exception):
        response = Resp()
    assert "model" in api.motivo_deepgram(Fallo())


def test_para_voz_lee_cifras():
    assert api.para_voz("Hay 41.427 registros y 12,5%") == "Hay cuarenta y un mil cuatrocientos veintisiete registros y doce coma cinco por ciento"
    assert api.para_voz("Hay 1 646 camas") == "Hay mil seiscientos cuarenta y seis camas"
