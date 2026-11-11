"""Carga y limpieza de datos.

Reemplaza el bloque que se repetía en todos los notebooks (montar Drive,
``read_csv`` con sep/encoding, coma decimal -> punto, hora "HH:MM" -> int).
Las funciones de limpieza son idempotentes: se pueden aplicar dos veces.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd

from . import config


def read_csv(name: str, data_dir: str | Path | None = None) -> pd.DataFrame:
    """Lee uno de los CSV definidos en ``config.FILES`` (sep=';', latin1)."""
    base = Path(data_dir) if data_dir else config.get_data_dir()
    path = base / config.FILES[name]
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Poné los CSV en esa carpeta o definí "
            "VUELOS_DATA_DIR (ver data/README.md)."
        )
    return pd.read_csv(path, sep=config.CSV_SEP, encoding=config.CSV_ENCODING)


def es_texto(serie: pd.Series) -> bool:
    """True si la columna es texto (dtype ``object`` en pandas 2, ``str`` en pandas 3)."""
    return serie.dtype == object or pd.api.types.is_string_dtype(serie)


def decimal_coma_a_float(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convierte '52,8' -> 52.8 sólo en las columnas que todavía son texto."""
    for col in cols:
        if col in df.columns and es_texto(df[col]):
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False).astype(float)
    return df


def normalizar_si_no(valor) -> bool:
    """True para 'Sí', 'si', 'SI', ' sí ', etc. (ignora tildes y mayúsculas)."""
    texto = unicodedata.normalize("NFKD", str(valor)).encode("ascii", "ignore").decode()
    return texto.strip().lower() == "si"


def clean_vuelos(df: pd.DataFrame) -> pd.DataFrame:
    """Coma decimal -> float y 'HH:MM' -> hora entera. Devuelve una copia."""
    df = df.copy()
    decimal_coma_a_float(df, config.VUELOS_DECIMAL_COMA)
    col = "hora_salida_programada"
    if col in df.columns and es_texto(df[col]):
        df[col] = df[col].str.split(":").str[0].astype(int)
    return df


def load_vuelos(data_dir=None) -> pd.DataFrame:
    return clean_vuelos(read_csv("vuelos", data_dir))


def load_casos_nuevos(data_dir=None) -> pd.DataFrame:
    return clean_vuelos(read_csv("casos_nuevos", data_dir))


def clean_clientes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    decimal_coma_a_float(df, config.CLIENTES_DECIMAL_COMA)
    return df


def load_clientes(data_dir=None) -> pd.DataFrame:
    return clean_clientes(read_csv("clientes", data_dir))


def load_clientes_demorados(data_dir=None) -> pd.DataFrame:
    """Clientes con vuelos demorados, sin duplicados (input de K-Means y PCA)."""
    return clean_clientes(read_csv("clientes_demorados", data_dir))


def load_relacion(data_dir=None) -> pd.DataFrame:
    """Relación vuelo <-> cliente de los casos nuevos (usa 'id_cliente')."""
    df = read_csv("relacion", data_dir)
    if "id_pasajero" in df.columns and "id_cliente" not in df.columns:
        df = df.rename(columns={"id_pasajero": "id_cliente"})
    return df
