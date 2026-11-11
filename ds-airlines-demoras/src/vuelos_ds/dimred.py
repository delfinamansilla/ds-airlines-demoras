"""Análisis de Componentes Principales (PCA)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import config


def fit_pca(df: pd.DataFrame, features: list[str] | None = None):
    """Estandariza las variables y ajusta un PCA completo.

    Devuelve (pca, X_escalado, X_en_componentes).
    """
    features = features or config.CLIENTES_NUMERICAS
    X = StandardScaler().fit_transform(df[features].values)
    pca = PCA().fit(X)
    return pca, X, pca.transform(X)


def tabla_varianza(pca: PCA) -> pd.DataFrame:
    """Autovalor, % de varianza y % acumulado por componente."""
    ratio = pca.explained_variance_ratio_
    return pd.DataFrame({
        "Componente": [f"PC{i + 1}" for i in range(len(ratio))],
        "Autovalor": pca.explained_variance_,
        "% varianza": ratio * 100,
        "% acumulada": np.cumsum(ratio) * 100,
    }).set_index("Componente")


def loadings(pca: PCA, features: list[str] | None = None) -> pd.DataFrame:
    features = features or config.CLIENTES_NUMERICAS
    return pd.DataFrame(pca.components_.T, index=features,
                        columns=[f"PC{i + 1}" for i in range(len(features))])


def kaiser(pca: PCA) -> int:
    """Cantidad de componentes con autovalor > 1 (criterio de Kaiser)."""
    return int((pca.explained_variance_ > 1).sum())


def bartlett(X: np.ndarray) -> tuple[float, float]:
    """Test de esfericidad de Bartlett. H0: la matriz de correlación es la identidad.

    chi2 = -(n - 1 - (2p + 5)/6) * ln|R|, con p(p-1)/2 grados de libertad.
    (Misma fórmula que ``factor_analyzer.calculate_bartlett_sphericity``,
    sin depender de esa librería.)
    """
    n, p = X.shape
    corr = np.corrcoef(X, rowvar=False)
    chi2 = -(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(corr))
    pvalue = stats.chi2.sf(chi2, p * (p - 1) / 2)
    return float(chi2), float(pvalue)
