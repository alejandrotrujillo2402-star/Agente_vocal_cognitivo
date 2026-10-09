import pandas as pd
import pytest

import datos_ips


def fila(depto, mun, prestador, sede, nat, nivel, grupo, desc, n, cod_p, cod_s):
    return {"departamento": depto, "municipio": mun, "nombre_prestador": prestador, "nom_sede_ips": sede,
            "naturaleza": nat, "num_nivel_atencion": nivel, "nom_grupo_capacidad": grupo,
            "nom_descripcion_capacidad": desc, "num_cantidad_capacidad_instalada": str(n),
            "c_digo_prestador": cod_p, "c_digo_sede": cod_s, "fecha_corte": "Fecha corte REPS: Nov 5 2022"}


FILAS = [
    fila("Caldas", "Manizales", "HOSPITAL SANTA SOFIA", "SEDE PRINCIPAL", "Pública", "3", "CAMAS", "Adultos", 120, "P1", "S1"),
    fila("Caldas", "Manizales", "HOSPITAL SANTA SOFIA", "SEDE PRINCIPAL", "Pública", "3", "CAMAS", "Intensiva Adultos", 20, "P1", "S1"),
    fila("Caldas", "Manizales", "HOSPITAL SANTA SOFIA", "SEDE PRINCIPAL", "Pública", "3", "AMBULANCIAS", "Medicalizada", 2, "P1", "S1"),
    fila("Caldas", "Manizales", "CLINICA AVIDANTI", "AVIDANTI MANIZALES", "Privada", "", "CAMAS", "Cuidado Intensivo Adulto", 15, "P2", "S2"),
    fila("Caldas", "Villamaria", "ESE VILLAMARIA", "CENTRO", "Pública", "1", "CONSULTORIOS", "Consulta Externa", 6, "P3", "S3"),
    fila("Valle del cauca", "Palmira", "HOSPITAL PALMIRA", "PRINCIPAL", "Pública", "2", "CAMAS", "Intensiva Adultos", 10, "P4", "S4"),
    fila("Cali", "Cali", "FUNDACION VALLE DEL LILI", "SEDE SUR", "Privada", "", "CAMAS", "Intensiva Adultos", 80, "P5", "S5"),
]


@pytest.fixture(autouse=True)
def datos():
    datos_ips.usar_dataframe(pd.DataFrame(FILAS))
