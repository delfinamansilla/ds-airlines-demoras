"""Gráficos reutilizables.

Antes había ~25 celdas con el mismo patrón (countplot + barplot de % de demora,
boxplots en loop, scatter de componentes...). Cada patrón es ahora una función.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from . import config
from .eda import tasa_demora_por


def set_style() -> None:
    sns.set_theme(style="whitegrid")


def _ax(ax=None, figsize=(10, 5)):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _rotate(ax, rotation=45):
    ax.tick_params(axis="x", rotation=rotation)


def plot_counts(df, col, hue=None, ax=None, order=True, rotation=45, palette="viridis"):
    """Barras de frecuencia de ``col`` (opcionalmente partidas por ``hue``)."""
    ax = _ax(ax, (12, 5))
    orden = df[col].value_counts().index if (order and hue is None) else None
    if hue is None:
        sns.countplot(data=df, x=col, hue=col, order=orden, palette=palette, legend=False, ax=ax)
    else:
        sns.countplot(data=df, x=col, hue=hue, order=orden, palette=palette, ax=ax)
    ax.set(title=f"Distribución de {col}" if hue is None else f"{col} según {hue}",
           xlabel=col, ylabel="Cantidad")
    _rotate(ax, rotation)
    return ax


def plot_numeric_overview(df, col):
    """Boxplot (arriba) + histograma con KDE (abajo) de una variable numérica."""
    fig, (ax_box, ax_hist) = plt.subplots(
        2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=(10, 6))
    sns.boxplot(x=df[col], ax=ax_box, color="skyblue")
    ax_box.set(title=f"Distribución de {col}", xlabel="")
    sns.histplot(df[col], ax=ax_hist, kde=True, color="darkblue")
    ax_hist.set(xlabel=col, ylabel="Frecuencia")
    fig.tight_layout()
    return fig


def plot_target_balance(df, target=config.TARGET):
    """Conteo con % encima de las barras + torta de la variable objetivo."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(data=df, x=target, hue=target, palette="magma", legend=False, ax=ax1)
    for p in ax1.patches:
        ax1.annotate(f"{100 * p.get_height() / len(df):.1f}%",
                     (p.get_x() + p.get_width() / 2, p.get_height()),
                     ha="center", va="bottom")
    ax1.set(title="No demorados (0) vs demorados (1)", xlabel="Demora", ylabel="Cantidad")
    df[target].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax2)
    ax2.set(title="Proporción de vuelos demorados", ylabel="")
    fig.tight_layout()
    return fig


def plot_rate_by(df, col, target=config.TARGET, ax=None, palette="Purples", rotation=45):
    """Barras con el % de vuelos demorados por categoría de ``col``."""
    ax = _ax(ax, (12, 6))
    tasa = tasa_demora_por(df, col, target)
    sns.barplot(x=tasa.index.astype(str), y=tasa.values, hue=tasa.index.astype(str),
                palette=palette, legend=False, ax=ax)
    ax.set(title=f"% de demora según {col}", xlabel=col, ylabel="% de vuelos demorados")
    _rotate(ax, rotation)
    return ax


def plot_counts_and_rate(df, col, target=config.TARGET, palette="Purples"):
    """Par de gráficos: conteo por demora + % de demora, para una categórica."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    plot_counts(df, col, hue=target, ax=ax1)
    plot_rate_by(df, col, target, ax=ax2, palette=palette)
    fig.tight_layout()
    return fig


def plot_numeric_vs_target(df, col, target=config.TARGET):
    """Boxplot + histograma de ``col`` separado por demora."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df, x=target, y=col, hue=target, legend=False, ax=ax1)
    ax1.set_title(f"{col} según demora")
    sns.histplot(data=df, x=col, hue=target, kde=True, bins=30, ax=ax2)
    ax2.set_title(f"Distribución de {col} según demora")
    fig.tight_layout()
    return fig


def plot_matrix(df, cols, kind="corr", ax=None):
    """Heatmap de correlación (``kind='corr'``) o covarianza (``kind='cov'``)."""
    ax = _ax(ax, (10, 7))
    m = df[cols].corr() if kind == "corr" else df[cols].cov()
    sns.heatmap(m, annot=True, fmt=".2f", cmap="coolwarm" if kind == "corr" else "viridis", ax=ax)
    ax.set_title("Matriz de correlación" if kind == "corr" else "Matriz de covarianza")
    return ax


def plot_heatmap_table(tabla, title, fmt=".2%", cmap="YlOrRd", annot=True, ax=None):
    ax = _ax(ax, (10, 6))
    sns.heatmap(tabla, annot=annot, fmt=fmt, cmap=cmap, ax=ax)
    ax.set_title(title)
    return ax


def plot_confusion(model, X_test, y_test, name, ax=None):
    ax = _ax(ax, (5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["No Demora", "Demora"], cmap="Blues", ax=ax)
    ax.set_title(f"Matriz de confusión - {name}")
    return ax


def plot_elbow(ks, wcss, ax=None):
    ax = _ax(ax, (8, 4))
    ax.plot(list(ks), wcss, marker="o")
    ax.set(title="Método del codo", xlabel="Número de clusters", ylabel="WCSS")
    ax.grid(True)
    return ax


def plot_silhouette(scores: dict, ax=None):
    ax = _ax(ax, (8, 4))
    ax.plot(list(scores), list(scores.values()), marker="o", color="green")
    ax.set(title="Coeficiente de silueta por K", xlabel="K", ylabel="Silueta")
    ax.grid(True)
    return ax


def plot_cluster_scatter(df, x, y, cluster="Cluster", ax=None, title=None):
    ax = _ax(ax, (10, 5))
    sns.scatterplot(data=df, x=x, y=y, hue=cluster, palette="viridis", ax=ax)
    ax.set_title(title or f"Segmentos: {x.replace('_', ' ')} vs {y.replace('_', ' ')}")
    ax.grid(True, alpha=.3)
    return ax


def plot_bar_series(serie: pd.Series, title, ylabel, ax=None, palette="magma"):
    ax = _ax(ax, (10, 5))
    sns.barplot(x=serie.index.astype(str), y=serie.values, hue=serie.index.astype(str),
                palette=palette, legend=False, ax=ax)
    ax.set(title=title, ylabel=ylabel)
    _rotate(ax)
    return ax


def plot_scree(pca, ax=None):
    """Varianza explicada (barras) y acumulada (línea) por componente."""
    ax = _ax(ax, (9, 5))
    ratio = pca.explained_variance_ratio_ * 100
    n = np.arange(1, len(ratio) + 1)
    ax.bar(n, ratio, alpha=.7, label="Individual")
    ax.plot(n, np.cumsum(ratio), "o--", color="C1", label="Acumulada")
    ax.set(title="Varianza explicada por componente principal",
           xlabel="Componente principal", ylabel="% de varianza")
    ax.legend()
    ax.grid(True)
    return ax


def plot_pc_pairs(x_pca, pairs, alpha=0.075, s=6, clip=None):
    """Scatter de pares de componentes (índices base 0), en una grilla.

    ``clip=(xmax, ymin)`` replica el recorte de ejes que usaba el notebook original
    sólo para PC1 vs PC2 (oculta outliers extremos); por defecto no se recorta.
    """
    n = len(pairs)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    for ax, (i, j) in zip(axes.ravel(), pairs):
        ax.scatter(x_pca[:, i], x_pca[:, j], alpha=alpha, s=s)
        ax.set(xlabel=f"PC{i + 1}", ylabel=f"PC{j + 1}", title=f"PC{i + 1} vs PC{j + 1}")
        ax.grid(True)
        if clip and (i, j) == (0, 1):
            ax.set_xlim(right=clip[0])
            ax.set_ylim(bottom=clip[1])
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig
