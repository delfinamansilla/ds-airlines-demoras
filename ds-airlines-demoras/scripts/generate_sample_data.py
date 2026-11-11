"""Genera datos SINTÉTICOS con el mismo esquema que los CSV del TP.

Sirve para probar el repo (tests, notebooks) sin tener los datos reales.
Los resultados sobre estos datos NO tienen valor analítico: están inventados.

Uso:
    python scripts/generate_sample_data.py [carpeta_destino]   # default: data/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)

AEROPUERTOS = ["EZE", "BOG", "MDE", "GRU", "PTY", "LIM", "UIO", "SCL", "MVD", "CCS"]
DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
CLIMAS = ["Despejado", "Nublado", "Lluvia", "Niebla", "Tormenta"]
AVIONES = ["Airbus A320", "Airbus A320neo", "Airbus A330", "Boeing 737", "Boeing 787", "ATR 72"]
OCUPACIONES = ["Estudiante", "Empleado", "Ingeniero", "Comerciante", "Docente"]
PROVINCIAS = ["CABA", "Bolívar", "Chiriquí", "Pernambuco", "Nariño"]


def _coma(x: pd.Series, dec: int = 1) -> pd.Series:
    return x.round(dec).astype(str).str.replace(".", ",", regex=False)


def make_vuelos(n: int, con_demora: bool, start_id: int = 1) -> pd.DataFrame:
    df = pd.DataFrame({
        "id_vuelo": np.arange(start_id, start_id + n),
        "aeropuerto_origen": RNG.choice(AEROPUERTOS, n),
        "aeropuerto_destino": RNG.choice(AEROPUERTOS, n),
        "hora_salida_programada": [f"{h:02d}:{m:02d}" for h, m in
                                   zip(RNG.integers(5, 23, n), RNG.choice([0, 15, 30, 45], n))],
        "dia_semana": RNG.choice(DIAS, n),
        "distancia_vuelo": RNG.gamma(1.2, 1500, n).astype(int).clip(35, 10050),
        "condiciones_climaticas": RNG.choice(CLIMAS, n, p=[.4, .25, .2, .1, .05]),
        "congestion_aerea": RNG.choice(["Baja", "Media", "Alta"], n, p=[.4, .4, .2]),
        "tipo_avion": RNG.choice(AVIONES, n),
        "ocupacion_vuelo": RNG.normal(71, 16, n).clip(9, 99.8),
        "temporada_alta": RNG.choice(["VERDADERO", "FALSO"], n),
        "puerta_embarque": RNG.integers(1, 40, n),
        "visibilidad": RNG.uniform(0.2, 30, n),
    })
    df["tiempo_estimado_vuelo"] = (df["distancia_vuelo"] / 11 + 20).astype(int)
    if con_demora:
        logit = (-1.3 + 1.1 * (df["congestion_aerea"] == "Alta") + 1.6 * (df["condiciones_climaticas"] == "Tormenta")
                 + 0.45 * (df["temporada_alta"] == "VERDADERO") - 0.04 * (df["visibilidad"] - 15))
        df["demora"] = (RNG.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df["ocupacion_vuelo"] = _coma(df["ocupacion_vuelo"])
    df["visibilidad"] = _coma(df["visibilidad"])
    return df


def make_clientes(n: int) -> pd.DataFrame:
    vuelos = RNG.poisson(15, n) + 1
    premium = RNG.random(n) < 0.12
    vuelos = np.where(premium, vuelos * 3, vuelos)
    millas = np.where(RNG.random(n) < (0.2 + 0.4 * premium), (vuelos * RNG.gamma(2, 400, n)).astype(int), 0)
    gasto = vuelos * RNG.gamma(2, 250, n) * np.where(premium, 2, 1)
    return pd.DataFrame({
        "id_cliente": RNG.permutation(np.arange(10000, 10000 + n)),
        "sexo": RNG.choice(["M", "F"], n),
        "provincia": RNG.choice(PROVINCIAS, n),
        "edad": RNG.normal(38, 11, n).clip(18, 80).astype(int),
        "ocupacion": RNG.choice(OCUPACIONES, n),
        "cant_vuelos": vuelos,
        "clase_preferida": RNG.choice(["Económica", "Ejecutiva", "Primera"], n, p=[.7, .25, .05]),
        "gasto_acumulado": gasto.round(2),
        "gasto_acumulado_extra": (gasto * RNG.uniform(.05, .2, n)).round(2),
        "programaMillas": np.where(millas > 0, "Sí", "No"),
        "cantidad_millas": millas,
        "ingreso_mensual": (RNG.gamma(2.5, 900, n) * np.where(premium, 2.5, 1)).astype(int),
        "anticipacion_compra_promedio": RNG.integers(1, 90, n),
        "canal_compra": RNG.choice(["Web", "App", "Agencia"], n),
    })


def main(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    kw = dict(sep=";", encoding="latin1", index=False)
    vuelos = make_vuelos(3000, True)
    nuevos = make_vuelos(500, False, start_id=100001)
    clientes = make_clientes(2500)
    clientes_dec = clientes.copy()
    for c in ("gasto_acumulado", "gasto_acumulado_extra"):
        clientes_dec[c] = _coma(clientes_dec[c], 2)
    relacion = pd.DataFrame({
        "id_vuelo": RNG.choice(nuevos["id_vuelo"], 4000),
        "id_pasajero": RNG.choice(clientes["id_cliente"], 4000),
    })
    vuelos.to_csv(dest / "Vuelos.csv", **kw)
    nuevos.to_csv(dest / "casosNuevos.csv", **kw)
    clientes_dec.to_csv(dest / "Clientes.csv", **kw)
    clientes.sample(frac=.9, random_state=0).to_csv(dest / "Clientes_Demorados_SinDuplicados.csv", **kw)
    relacion.to_csv(dest / "casosNuevos_Clientes.csv", **kw)
    print(f"Datos sintéticos generados en {dest.resolve()}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data")
