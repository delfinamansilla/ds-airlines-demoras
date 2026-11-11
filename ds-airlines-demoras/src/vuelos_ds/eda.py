"""Estadísticos del análisis exploratorio (sin gráficos; ver ``plots.py``)."""
from __future__ import annotations

import pandas as pd

from . import config


def frecuencias(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Frecuencia absoluta y relativa (%) de una variable categórica."""
    abs_ = df[col].value_counts()
    rel = (df[col].value_counts(normalize=True) * 100).round(2)
    return pd.DataFrame({"frecuencia": abs_, "porcentaje": rel})


def tasa_demora_por(df: pd.DataFrame, col: str, target: str = config.TARGET) -> pd.Series:
    """% de vuelos demorados por categoría, ordenado de mayor a menor."""
    return (df.groupby(col)[target].mean() * 100).sort_values(ascending=False)


def resumen_por_target(df: pd.DataFrame, cols: list[str], target: str = config.TARGET) -> dict:
    """Media/mediana/desvío de cada variable según demora + dif. % de medias."""
    out = {}
    for col in cols:
        res = df.groupby(target)[col].agg(["mean", "median", "std"]).round(2)
        dif = (res.loc[1, "mean"] - res.loc[0, "mean"]) / res.loc[0, "mean"] * 100
        out[col] = (res, round(dif, 2))
    return out


def outliers_iqr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Cantidad y % de outliers por el método IQR (1.5 * IQR)."""
    filas = []
    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
        filas.append({"variable": col, "outliers": int(mask.sum()),
                      "porcentaje": round(mask.mean() * 100, 2)})
    return pd.DataFrame(filas).set_index("variable")


def marcar_outliers_distancia(df: pd.DataFrame) -> pd.Series:
    """Booleana: True si la distancia supera el límite superior del IQR."""
    q1, q3 = df["distancia_vuelo"].quantile([0.25, 0.75])
    return df["distancia_vuelo"] > q3 + 1.5 * (q3 - q1)


def agregar_rango_distancia(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'rango_distancia' (Corto/Medio/Largo) por terciles."""
    df = df.copy()
    df["rango_distancia"] = pd.qcut(df["distancia_vuelo"], q=3, labels=["Corto", "Medio", "Largo"])
    return df


def correlaciones_altas(df: pd.DataFrame, umbral: float = 0.45) -> pd.DataFrame:
    """Pares de variables numéricas con |correlación| > umbral (sin duplicar pares)."""
    corr = df.select_dtypes("number").corr()
    pares = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if abs(corr.loc[a, b]) > umbral:
                pares.append({"var_1": a, "var_2": b, "correlacion": round(corr.loc[a, b], 3)})
    return pd.DataFrame(pares, columns=["var_1", "var_2", "correlacion"])
