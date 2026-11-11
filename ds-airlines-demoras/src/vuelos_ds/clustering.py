"""Segmentación de clientes con K-Means."""
from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config
from .data import normalizar_si_no


def prepare_kmeans_features(clientes: pd.DataFrame) -> tuple[pd.DataFrame, "np.ndarray"]:
    """Deja sólo variables útiles, codifica y estandariza.

    - Descarta id/sexo/provincia/canal.
    - 'programaMillas' -> 0/1.
    - One-hot de ocupación y clase preferida.
    - StandardScaler (K-Means es sensible a la escala).
    Devuelve (DataFrame sin escalar, matriz escalada); ambos con el mismo orden de filas.
    """
    df = clientes.drop(columns=[c for c in config.KMEANS_COLS_DESCARTADAS if c in clientes.columns]).copy()
    if "programaMillas" in df.columns:
        df["programaMillas"] = df["programaMillas"].map(normalizar_si_no).astype(int)
    df = pd.get_dummies(df, columns=[c for c in config.KMEANS_ONEHOT if c in df.columns])
    df = df.dropna()
    return df, StandardScaler().fit_transform(df)


def elbow(X, ks=range(1, 10)) -> list[float]:
    """Inercia (WCSS) para cada K."""
    return [KMeans(n_clusters=k, n_init=10, random_state=0).fit(X).inertia_ for k in ks]


def silhouettes(X, ks=range(2, 6), sample_size: int | None = None) -> dict[int, float]:
    """Coeficiente de silueta por K (``sample_size`` acelera con datasets grandes)."""
    out = {}
    for k in ks:
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
        out[k] = silhouette_score(X, labels, sample_size=sample_size, random_state=0)
    return out


def fit_kmeans(X, k: int) -> KMeans:
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)


def interpretar_silueta(score: float) -> str:
    if score > 0.5:
        return "Estructura sólida, clústeres bien definidos."
    if score > 0.25:
        return "Estructura razonable, aunque existen algunos solapamientos."
    return "Estructura débil, los clústeres están muy cerca unos de otros."


def perfil_clusters(df_con_cluster: pd.DataFrame, cols: list[str] | None = None,
                    cluster: str = "Cluster") -> pd.DataFrame:
    """Promedio de cada variable por cluster."""
    cols = [c for c in (cols or config.KMEANS_PERFIL) if c in df_con_cluster.columns]
    return df_con_cluster.groupby(cluster)[cols].mean()


def poder_discriminante(df_con_cluster: pd.DataFrame, cols: list[str] | None = None,
                        cluster: str = "Cluster") -> pd.Series:
    """Variabilidad (%) de las medias entre clusters: desvío / promedio * 100."""
    medias = perfil_clusters(df_con_cluster, cols, cluster)
    return (medias.std() / medias.mean() * 100).sort_values(ascending=False)
