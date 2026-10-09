"""Capa de datos: IPS públicas y privadas (datos.gov.co, s2ru-bqt6).
Versión base con el contrato acordado; la versión de Cursor la reemplaza con el mismo contrato.

Contrato: cargar(), estado(), TOOLS, ejecutar(nombre, args), brief(), PROMPT_DATOS, preguntas_evaluacion()
"""
import asyncio
import difflib
import os
import re
import unicodedata
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd

DATASET = "s2ru-bqt6"
SODA3 = f"https://www.datos.gov.co/api/v3/views/{DATASET}/query.json"
SODA2 = f"https://www.datos.gov.co/resource/{DATASET}.json"
SNAPSHOT = Path(__file__).parent / "data" / "ips_snapshot.parquet"
PAGINA = 1000

DISTRITOS = {"valle del cauca": ["cali", "buenaventura"], "atlantico": ["barranquilla"],
             "bolivar": ["cartagena"], "magdalena": ["santa marta"]}
SINONIMOS = {"uci adultos": ["intensiva adultos", "cuidado intensivo adulto"],
             "uci adulto": ["intensiva adultos", "cuidado intensivo adulto"],
             "uci pediatrica": ["intensiva pediatrica", "cuidado intensivo pediatrico"],
             "uci neonatal": ["intensiva neonatal", "cuidado intensivo neonatal"],
             "intermedia adultos": ["intermedia adultos", "cuidado intermedio adulto"],
             "uci todas": ["intensiva adultos", "cuidado intensivo adulto", "intensiva pediatrica",
                           "cuidado intensivo pediatrico", "intensiva neonatal", "cuidado intensivo neonatal"]}

_D = {"df": None, "fuente": "sin datos", "actualizado": None, "brief": None}
# Fuente de la sesión en curso: un archivo del usuario con las columnas del dataset. None = la API (_D).
_ACTIVA: ContextVar = ContextVar("fuente_activa", default=None)


def _fuente():
    return _ACTIVA.get() or _D


@contextmanager
def con_fuente(fuente):
    """Dentro del bloque, las herramientas, estado() y brief() usan `fuente` en lugar de la API."""
    token = _ACTIVA.set(fuente)
    try:
        yield
    finally:
        _ACTIVA.reset(token)


def fuente_desde_tabla(df, nombre):
    """Tabla con las columnas del dataset -> fuente propia, con la misma normalización que la API."""
    return {"df": _preparar(df), "fuente": f"archivo {nombre}", "actualizado": datetime.now().strftime("%H:%M"), "brief": None}


def norm(t):
    t = unicodedata.normalize("NFKD", str(t).lower())
    return "".join(c for c in t if not unicodedata.combining(c)).strip()


# ---------------- carga ----------------

def _preparar(filas_o_df):
    df = filas_o_df if isinstance(filas_o_df, pd.DataFrame) else pd.DataFrame(filas_o_df)
    for c in ["departamento", "municipio", "nombre_prestador", "naturaleza", "num_nivel_atencion", "nom_sede_ips",
              "nom_grupo_capacidad", "nom_descripcion_capacidad", "c_digo_prestador", "c_digo_sede", "direcci_n",
              "tel_fono", "email", "gerente", "nit_ips", "fecha_corte"]:
        if c not in df:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str).str.strip()
    df["cantidad"] = pd.to_numeric(df.get("num_cantidad_capacidad_instalada", 0), errors="coerce").fillna(0).astype(int)
    for c in ["departamento", "municipio", "naturaleza", "nom_grupo_capacidad", "nom_descripcion_capacidad",
              "nombre_prestador", "nom_sede_ips"]:
        df[c + "_n"] = df[c].map(norm)
    canon = {d: k for k, ds in DISTRITOS.items() for d in ds}
    df["depto_canonico"] = df["departamento_n"].map(lambda d: canon.get(d, d))
    df["sede_id"] = df["c_digo_sede"].where(df["c_digo_sede"] != "", df["nombre_prestador"] + "|" + df["nom_sede_ips"])
    df["prestador_id"] = df["c_digo_prestador"].where(df["c_digo_prestador"] != "", df["nombre_prestador"])
    return df


async def _descargar():
    token = os.getenv("DATOS_GOV_TOKEN", "")
    cab = {"X-App-Token": token} if token else {}
    async with httpx.AsyncClient(timeout=60) as http:
        r = await http.get(SODA2, params={"$select": "count(*) as n"}, headers=cab)
        total = int(r.json()[0]["n"])
        sem = asyncio.Semaphore(6)

        async def pagina(i):
            async with sem:
                if token:
                    try:
                        q = await http.post(SODA3, headers=cab, json={"query": "SELECT *", "page": {"pageNumber": i + 1, "pageSize": PAGINA}})
                        q.raise_for_status()
                        d = q.json()
                        return (d if isinstance(d, list) else d.get("data") or d.get("rows") or []), "api soda3"
                    except Exception:
                        pass
                q = await http.get(SODA2, params={"$limit": PAGINA, "$offset": i * PAGINA, "$order": ":id"}, headers=cab)
                q.raise_for_status()
                return q.json(), "api soda2"

        res = await asyncio.gather(*(pagina(i) for i in range((total + PAGINA - 1) // PAGINA)))
    filas = [f for p, _ in res for f in p]
    if len(filas) != total:
        raise ValueError(f"descargadas {len(filas)} de {total}")
    return filas, res[0][1]


async def _refrescar():
    try:
        filas, modo = await _descargar()
        df = _preparar(filas)
        _D.update(df=df, fuente=modo, actualizado=datetime.now().strftime("%H:%M"), brief=None)
        try:
            SNAPSHOT.parent.mkdir(exist_ok=True)
            df.drop(columns=[c for c in df.columns if c.endswith("_n")]).to_parquet(SNAPSHOT)
        except Exception:
            pass
    except Exception as e:
        _D["error_api"] = str(e)[:200]


async def cargar():
    if SNAPSHOT.exists():
        try:
            _D.update(df=_preparar(pd.read_parquet(SNAPSHOT)), fuente="snapshot",
                      actualizado=datetime.fromtimestamp(SNAPSHOT.stat().st_mtime).strftime("%Y-%m-%d %H:%M"))
            asyncio.create_task(_refrescar())
            return
        except Exception:
            pass
    await _refrescar()


def usar_dataframe(df, fuente="prueba"):
    _D.update(df=_preparar(df), fuente=fuente, actualizado=datetime.now().strftime("%H:%M"), brief=None)


def _df():
    df = _fuente()["df"]
    if df is None:
        raise RuntimeError("datos no cargados")
    return df


def fecha_corte():
    df = _fuente()["df"]
    return (df["fecha_corte"].mode().iat[0] if df is not None and len(df) else "")


_ESTADO = {}


def estado():
    f = _fuente()
    df = f["df"]
    if df is None:
        return {"fuente": f["fuente"], "filas": 0, "error": f.get("error_api")}
    if _ESTADO.get("clave") == id(df):
        return _ESTADO["valor"]
    valor = {"fuente": f["fuente"], "filas": int(len(df)), "prestadores": int(df["prestador_id"].nunique()),
             "sedes": int(df["sede_id"].nunique()), "municipios": int(df[["departamento", "municipio"]].drop_duplicates().shape[0]),
             "fecha_corte": fecha_corte(), "actualizado": f["actualizado"], "error_api": f.get("error_api")}
    _ESTADO["clave"], _ESTADO["valor"] = id(df), valor
    return valor


# ---------------- resolución de valores ----------------

def _sinonimo_uci(v):
    """Interpreta las formas habladas de UCI/intermedia: 'camas de uci adulto', 'cuidados intensivos', 'uci'."""
    v = re.sub(r"\b(camas?|de|del|la|las|unidad(es)?|para)\b", " ", v)
    v = " ".join(v.split())
    intensiva = any(w in v for w in ("uci", "intensiv"))
    intermedia = any(w in v for w in ("ucin", "intermedi"))
    if not (intensiva or intermedia):
        return None
    tipo = "intermedia" if intermedia and not "uci " in v + " " else "intensiva"
    if "neonat" in v:
        return f"uci neonatal" if tipo == "intensiva" else None
    if "pediat" in v or "nin" in v:
        return "uci pediatrica" if tipo == "intensiva" else None
    if "adult" in v:
        return "uci adultos" if tipo == "intensiva" else "intermedia adultos"
    return "uci todas" if tipo == "intensiva" else None


def _resolver(df, col, valor, umbral=0.8):
    v = norm(valor)
    if col == "nom_descripcion_capacidad_n" and v not in SINONIMOS:
        v = _sinonimo_uci(v) or v
    valores = df[col].unique().tolist()
    if col == "nom_descripcion_capacidad_n" and v.startswith("uci"):
        edad = {"uci adultos": "adult", "uci adulto": "adult", "uci pediatrica": "pediat", "uci neonatal": "neonat"}.get(v, "")
        ok = [x for x in valores if "intensiv" in x and edad in x]
        if ok:
            return ok, None
    if v in valores:
        return [v], None
    if v in SINONIMOS:
        ok = [s for s in SINONIMOS[v] if s in valores]
        if ok:
            return ok, None
    parciales = [x for x in valores if v and v in x]
    if 0 < len(parciales) <= 3:
        return parciales, None
    cercanos = difflib.get_close_matches(v, valores, n=5, cutoff=0.5)
    if cercanos and difflib.SequenceMatcher(None, v, cercanos[0]).ratio() >= umbral:
        return [cercanos[0]], None
    return None, (cercanos or parciales[:5])


def _filtrar(df, f, notas):
    aplicados = {}
    if f.get("departamento"):
        d = norm(f["departamento"])
        col = "depto_canonico" if f.get("incluir_distritos", True) and d in DISTRITOS else "departamento_n"
        vals, sug = _resolver(df, col, d)
        if vals is None:
            return None, {"error": "valor_no_encontrado", "campo": "departamento", "sugerencias": sug}
        df = df[df[col].isin(vals)]
        aplicados["departamento"] = ", ".join(vals)
        if col == "depto_canonico":
            notas.append(f"Incluye los distritos {', '.join(DISTRITOS[d])}, que el dataset registra aparte.")
    f = dict(f)
    grupos = {"camas", "consultorios", "salas", "ambulancias", "camillas", "unidad movil", "sillas"}
    if f.get("grupo_capacidad") and norm(f["grupo_capacidad"]) not in grupos and not f.get("descripcion_capacidad"):
        f["descripcion_capacidad"] = f.pop("grupo_capacidad")
        notas.append(f"'{f['descripcion_capacidad']}' es un tipo de capacidad, no un grupo: se filtró por descripción.")
    for campo, col in [("municipio", "municipio_n"), ("naturaleza", "naturaleza_n"),
                       ("grupo_capacidad", "nom_grupo_capacidad_n"), ("descripcion_capacidad", "nom_descripcion_capacidad_n")]:
        if f.get(campo):
            vals, sug = _resolver(df if campo != "naturaleza" else _df(), col, f[campo])
            if vals is None:
                return None, {"error": "valor_no_encontrado", "campo": campo, "sugerencias": sug}
            df = df[df[col].isin(vals)]
            aplicados[campo] = ", ".join(vals)
    if f.get("nivel_atencion"):
        nivel = str(f["nivel_atencion"]).strip()
        df = df[df["num_nivel_atencion"] == nivel]
        aplicados["nivel_atencion"] = nivel
        notas.append("El nivel de atención solo se registra para IPS públicas.")
    return df, aplicados


# ---------------- herramientas ----------------

def consultar_estadistica(metrica="num_sedes", filtros=None, agrupar_por=None, top_n=10, **_):
    df = _df()
    top_n = int(top_n or 10)
    if isinstance(filtros, str):
        import json as _j
        filtros = _j.loads(filtros or "{}")
    notas = []
    sub, aplicados = _filtrar(df, filtros or {}, notas)
    if sub is None:
        return aplicados
    col_grupo = {"departamento": "depto_canonico", "municipio": "municipio", "naturaleza": "naturaleza",
                 "nivel_atencion": "num_nivel_atencion", "grupo_capacidad": "nom_grupo_capacidad",
                 "descripcion_capacidad": "nom_descripcion_capacidad"}.get(agrupar_por or "")

    def medir(d):
        if metrica == "num_prestadores":
            return int(d["prestador_id"].nunique())
        if metrica == "suma_capacidad":
            return int(d["cantidad"].sum())
        return int(d["sede_id"].nunique())

    out = {"metrica": metrica, "filtros_aplicados": aplicados, "fecha_corte": fecha_corte()}
    if metrica == "suma_capacidad" and not (filtros or {}).get("grupo_capacidad") and not (filtros or {}).get("descripcion_capacidad") and agrupar_por not in ("grupo_capacidad", "descripcion_capacidad"):
        # sumar camas + consultorios + ambulancias no tiene sentido: se obliga a corregir
        return {"error": "falta_grupo_capacidad", "campo": "filtros.grupo_capacidad",
                "sugerencias": ["CAMAS", "CONSULTORIOS", "SALAS", "AMBULANCIAS", "CAMILLAS", "UNIDAD MOVIL", "SILLAS"],
                "recomendacion": "Repite la consulta con filtros.grupo_capacidad (por ejemplo CAMAS) o agrupa por grupo_capacidad."}
    if col_grupo:
        serie = sub.groupby(col_grupo).apply(medir, include_groups=False).sort_values(ascending=False)
        out["resultado"] = {str(k): int(v) for k, v in serie.head(top_n).items()}
        out["grupos_totales"] = int(len(serie))
    else:
        out["resultado"] = medir(sub)
    vacio = (out["resultado"] == 0) if not col_grupo else not out["resultado"]
    if vacio:
        if aplicados.get("nivel_atencion") and aplicados.get("naturaleza") and "publica" not in aplicados["naturaleza"]:
            out.update(sospechoso=True, recomendacion="Las IPS privadas y mixtas no tienen nivel registrado: quita el filtro de nivel y explícalo.")
        elif aplicados.get("departamento") and aplicados.get("municipio"):
            out.update(sospechoso=True, recomendacion="Revisa que el municipio pertenezca a ese departamento o quita el departamento.")
        else:
            out.update(sospechoso=True, recomendacion="Ningún registro cumple los filtros; quita el filtro más específico.")
    out["notas"] = notas
    return out


def buscar_ips(nombre, municipio=None, departamento=None, limite=3, **_):
    df = _df()
    n = norm(nombre)
    sub = df
    if municipio:
        sub = sub[sub["municipio_n"] == norm(municipio)]
    if departamento:
        sub = sub[sub["depto_canonico"] == norm(departamento)] if norm(departamento) in DISTRITOS else sub[sub["departamento_n"] == norm(departamento)]
    nombres = pd.concat([sub["nombre_prestador_n"], sub["nom_sede_ips_n"]]).unique().tolist()
    palabras = [p for p in n.split() if len(p) > 2]
    candidatos = [x for x in nombres if all(p in x for p in palabras)] if palabras else []
    if not candidatos:
        candidatos = difflib.get_close_matches(n, nombres, n=limite, cutoff=0.6)
    if not candidatos:
        sug = difflib.get_close_matches(n, df["nombre_prestador_n"].unique().tolist(), n=5, cutoff=0.4)
        return {"error": "ips_no_encontrada", "sugerencias": [df.loc[df["nombre_prestador_n"] == s, "nombre_prestador"].iat[0] for s in sug],
                "fecha_corte": fecha_corte()}
    filas = sub[sub["nombre_prestador_n"].isin(candidatos) | sub["nom_sede_ips_n"].isin(candidatos)]
    salida = []
    for sede, g in list(filas.groupby("sede_id", sort=False))[:limite]:
        p = g.iloc[0]
        cap = g.groupby(["nom_grupo_capacidad", "nom_descripcion_capacidad"])["cantidad"].sum()
        salida.append({"prestador": p["nombre_prestador"], "sede": p["nom_sede_ips"], "municipio": p["municipio"],
                       "departamento": p["departamento"], "naturaleza": p["naturaleza"],
                       "nivel": p["num_nivel_atencion"] or "no registrado", "nit": p["nit_ips"],
                       "direccion": p["direcci_n"], "telefono": p["tel_fono"], "gerente": p["gerente"],
                       "capacidad": {f"{a} {b}": int(v) for (a, b), v in cap.items()}})
    return {"ips": salida[0]["prestador"], "resultado": salida, "coincidencias": int(filas["sede_id"].nunique()),
            "filtros_aplicados": {"municipio": municipio, "departamento": departamento}, "fecha_corte": fecha_corte()}


def listar_valores(columna, contiene=None, limite=15, **_):
    df = _df()
    col = {"departamento": "departamento", "municipio": "municipio", "naturaleza": "naturaleza",
           "grupo_capacidad": "nom_grupo_capacidad", "descripcion_capacidad": "nom_descripcion_capacidad",
           "nivel_atencion": "num_nivel_atencion"}.get(columna)
    if not col:
        return {"error": "columna_no_valida", "sugerencias": ["departamento", "municipio", "naturaleza", "grupo_capacidad", "descripcion_capacidad", "nivel_atencion"]}
    vals = df[col][df[col] != ""].value_counts()
    if contiene:
        c = norm(contiene)
        c = "intensiv" if c in ("uci", "ucis", "cuidados intensivos") else c
        vals = vals[[c in norm(v) for v in vals.index]]
    return {"columna": columna, "valores": list(vals.index[:limite]), "total_distintos": int(len(vals)), "fecha_corte": fecha_corte()}


def info_dataset(**_):
    return {**estado(), "contiene": "IPS (prestadores y sedes) por departamento y municipio, naturaleza, nivel de atención (solo públicas), datos de contacto y capacidad instalada (camas, consultorios, salas, ambulancias, camillas, sillas, unidades móviles).",
            "no_contiene": "médicos, especialidades, EPS, precios, calidad, citas, ocupación ni datos posteriores a la fecha de corte."}


FUNCIONES = {"consultar_estadistica": consultar_estadistica, "buscar_ips": buscar_ips,
             "listar_valores": listar_valores, "info_dataset": info_dataset}


def ejecutar(nombre, args):
    try:
        f = FUNCIONES.get(nombre)
        if not f:
            return {"error": "herramienta_desconocida", "sugerencias": list(FUNCIONES)}
        return f(**(args or {}))
    except TypeError as e:
        return {"error": "argumentos_invalidos", "detalle": str(e)[:200], "sugerencias": []}
    except Exception as e:
        return {"error": "fallo_interno", "detalle": str(e)[:200], "sugerencias": []}


_FILTROS = {"type": "object", "properties": {
    "departamento": {"type": "string"}, "municipio": {"type": "string"},
    "naturaleza": {"type": "string", "description": "Pública, Privada o Mixta"},
    "nivel_atencion": {"type": ["string", "integer"], "description": "1, 2 o 3 (solo públicas)"},
    "grupo_capacidad": {"type": "string", "description": "Solo uno de: CAMAS, CONSULTORIOS, SALAS, AMBULANCIAS, CAMILLAS, UNIDAD MOVIL, SILLAS. La UCI va en descripcion_capacidad"},
    "descripcion_capacidad": {"type": "string", "description": "Detalle dentro del grupo. Ej: uci adultos, uci neonatal, Adultos, Pediátrica, Básica, Medicalizada"},
    "incluir_distritos": {"type": ["boolean", "string"]}}}

TOOLS = [
    {"type": "function", "function": {"name": "consultar_estadistica",
     "description": "Cifras exactas: número de prestadores, de sedes o suma de capacidad instalada (camas, ambulancias...: indica SIEMPRE filtros.grupo_capacidad al sumar), con filtros y ranking opcional.",
     "parameters": {"type": "object", "properties": {
         "metrica": {"type": "string", "enum": ["num_prestadores", "num_sedes", "suma_capacidad"]},
         "filtros": _FILTROS,
         "agrupar_por": {"type": "string", "enum": ["departamento", "municipio", "naturaleza", "nivel_atencion", "grupo_capacidad", "descripcion_capacidad"]},
         "top_n": {"type": ["integer", "string"]}}, "required": ["metrica"]}}},
    {"type": "function", "function": {"name": "buscar_ips",
     "description": "Ficha de una IPS por nombre (tolera errores): sede, contacto, gerente, nivel y capacidad.",
     "parameters": {"type": "object", "properties": {"nombre": {"type": "string"}, "municipio": {"type": "string"},
                                                       "departamento": {"type": "string"}}, "required": ["nombre"]}}},
    {"type": "function", "function": {"name": "listar_valores",
     "description": "Valores válidos de una columna, para resolver nombres dudosos.",
     "parameters": {"type": "object", "properties": {
         "columna": {"type": "string", "enum": ["departamento", "municipio", "naturaleza", "grupo_capacidad", "descripcion_capacidad", "nivel_atencion"]},
         "contiene": {"type": "string"}}, "required": ["columna"]}}},
    {"type": "function", "function": {"name": "info_dataset",
     "description": "Qué contiene y qué no contiene el dataset, totales y fecha de corte.",
     "parameters": {"type": "object", "properties": {}}}},
]

def _permitir_null(esquema):
    props = esquema.get("properties", {})
    req = set(esquema.get("required", []))
    for k, v in props.items():
        if v.get("type") == "object":
            _permitir_null(v)
        if k not in req:
            t = v.get("type")
            tipos = t if isinstance(t, list) else [t]
            if "null" not in tipos:
                v["type"] = tipos + ["null"]
            v.pop("enum", None) if k != "metrica" else None


for _t in TOOLS:
    _permitir_null(_t["function"]["parameters"])

PROMPT_DATOS = """FICHA DEL DATASET
- Cada fila es una sede con un tipo de capacidad. Prestador (la institución, con NIT) es distinto de sede (cada punto de atención). "¿Cuántas IPS?" se responde con prestadores y se aclara; nunca se cuentan filas.
- Naturaleza: Pública, Privada o Mixta. El nivel de atención (1, 2, 3) SOLO existe para públicas.
- Cali, Buenaventura, Barranquilla, Cartagena y Santa Marta aparecen como departamentos aparte; por defecto se suman a su departamento y se dice.
- Capacidad: CAMAS, CONSULTORIOS, SALAS, AMBULANCIAS, CAMILLAS, UNIDAD MOVIL, SILLAS; la UCI aparece con varios nombres (usa "uci adultos").
- Fecha de corte del REPS: noviembre de 2022. No hay datos más recientes.
- NO contiene: médicos, especialidades, EPS, precios, calidad, citas ni ocupación."""


def miles(n):
    return f"{int(n):,}".replace(",", ".")


def brief():
    f = _fuente()
    if f.get("brief"):
        return f["brief"]
    df = _df()
    e = estado()
    nat = df.drop_duplicates("prestador_id")["naturaleza"].value_counts()
    top = df.groupby("depto_canonico")["sede_id"].nunique().sort_values(ascending=False)
    camas = int(df.loc[df["nom_grupo_capacidad_n"] == "camas", "cantidad"].sum())
    uci = int(df.loc[df["nom_descripcion_capacidad_n"].isin(SINONIMOS["uci adultos"]), "cantidad"].sum())
    camas_sede = df[df["nom_grupo_capacidad_n"] == "camas"].groupby(["sede_id", "nombre_prestador", "municipio"])["cantidad"].sum()
    mayor = camas_sede.idxmax() if len(camas_sede) else ("", "", "")
    depto_top = top.index[0].title() if len(top) else ""
    b = {"titulo": "IPS públicas y privadas de Colombia",
         "tipo": "Datos abiertos · datos.gov.co",
         "resumen": (f"Registro oficial de {miles(e['prestadores'])} prestadores de salud con {miles(e['sedes'])} sedes en {miles(e['municipios'])} municipios: "
                     f"naturaleza, nivel de atención y capacidad instalada ({miles(camas)} camas, {miles(uci)} de UCI adultos). {e['fecha_corte']}."),
         "temas": ["Prestadores y sedes", "Pública vs privada", "Nivel de atención", "Camas y UCI", "Ambulancias", "Contacto de IPS"],
         "preguntas": [f"¿Cuántas sedes de IPS hay en {depto_top}?",
                       "¿Cuáles son los 5 municipios con más ambulancias medicalizadas?",
                       f"¿Qué capacidad tiene {mayor[1].title()} en {mayor[2].title()}?" if mayor[1] else "¿Qué capacidad tiene el hospital más grande de Caldas?",
                       "¿Hay más prestadores públicos o privados en Caldas?",
                       "¿Cuántos médicos trabajan en Manizales?"],
         "presentacion_voz": (f"Tengo cargado el registro oficial de IPS de Colombia: {e['prestadores']} prestadores y {e['sedes']} sedes, "
                              "con su capacidad instalada. Pregúntame por camas, ambulancias, una IPS o un municipio."),
         "stats": {**e, "naturaleza": {k: int(v) for k, v in nat.items()}, "camas": camas, "uci_adultos": uci}}
    f["brief"] = b
    return b


def preguntas_evaluacion():
    """Casos con respuesta esperada calculada desde el DataFrame (para evaluar.py)."""
    df = _df()
    c = lambda **f: consultar_estadistica(**f).get("resultado")
    casos = [
        {"id": "r1", "tipo": "resumen", "pregunta": "¿Cuántas sedes de IPS hay en Caldas?",
         "esperado": {"herramienta": "consultar_estadistica", "cifra": c(metrica="num_sedes", filtros={"departamento": "Caldas"})}},
        {"id": "r2", "tipo": "resumen", "pregunta": "¿Cuántos prestadores públicos hay en Antioquia?",
         "esperado": {"herramienta": "consultar_estadistica", "cifra": c(metrica="num_prestadores", filtros={"departamento": "Antioquia", "naturaleza": "Pública"})}},
        {"id": "r3", "tipo": "resumen", "pregunta": "¿Cuántas camas hay en total en Manizales?",
         "esperado": {"herramienta": "consultar_estadistica", "cifra": c(metrica="suma_capacidad", filtros={"municipio": "Manizales", "grupo_capacidad": "CAMAS"})}},
        {"id": "f1", "tipo": "fuera", "pregunta": "¿Cuántos médicos trabajan en Manizales?", "esperado": {"cifra": "no_disponible"}},
        {"id": "f2", "tipo": "fuera", "pregunta": "¿Qué EPS tiene más afiliados en Caldas?", "esperado": {"cifra": "no_disponible"}},
        {"id": "f3", "tipo": "fuera", "pregunta": "¿Cuánto cuesta una consulta en el Hospital de Caldas?", "esperado": {"cifra": "no_disponible"}},
        {"id": "t1", "tipo": "trampa", "pregunta": "¿Cuántas IPS privadas de nivel 3 hay en Colombia?", "esperado": {"cifra": "no_disponible"}},
        {"id": "t2", "tipo": "trampa", "pregunta": "¿Cuántas camas de UCI adultos hay en el Valle del Cauca?",
         "esperado": {"herramienta": "consultar_estadistica", "cifra": c(metrica="suma_capacidad", filtros={"departamento": "Valle del Cauca", "descripcion_capacidad": "uci adultos"})}},
        {"id": "m1", "tipo": "molesto", "pregunta": "Esto no sirve para nada, ya te pregunté y no me respondes. Quiero hablar con una persona.",
         "esperado": {"escalar": True}},
    ]
    return [k for k in casos if k["esperado"].get("cifra", 0) is not None]
