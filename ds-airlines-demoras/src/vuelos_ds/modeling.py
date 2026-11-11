"""Modelado de demoras (Árbol de Decisión y KNN) y armado del entregable final."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config
from .data import es_texto, normalizar_si_no


@dataclass
class FlightPreprocessor:
    """Codifica vuelos igual en entrenamiento y en casos nuevos.

    Se ajusta UNA vez con los datos de entrenamiento y se reutiliza para los
    casos nuevos. Las categorías se ordenan alfabéticamente (igual que el
    ``LabelEncoder`` del notebook original), así que los códigos son los mismos.
    Una categoría nunca vista en entrenamiento se codifica como -1.
    """

    categoricas: list[str] = field(default_factory=lambda: list(config.VUELOS_CATEGORICAS))
    descartar: list[str] = field(default_factory=lambda: list(config.VUELOS_COLS_DESCARTADAS))
    target: str = config.TARGET

    def _base(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=[c for c in self.descartar if c in df.columns]).copy()
        for col in config.VUELOS_BOOLEANAS:
            if col in df.columns and es_texto(df[col]):
                df[col] = df[col].map({"VERDADERO": 1, "FALSO": 0})
            elif col in df.columns and df[col].dtype == bool:
                df[col] = df[col].astype(int)
        return df

    def fit(self, df: pd.DataFrame) -> "FlightPreprocessor":
        base = self._base(df).dropna()
        self.encoder_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.encoder_.fit(base[self.categoricas])
        self.feature_cols_ = [c for c in base.columns if c != self.target]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        base = self._base(df).dropna()
        base[self.categoricas] = self.encoder_.transform(base[self.categoricas]).astype(int)
        return base[self.feature_cols_ + ([self.target] if self.target in base.columns else [])]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def split_train_test(df_prep: pd.DataFrame, target: str = config.TARGET):
    """70/30 estratificado y reproducible."""
    X, y = df_prep.drop(columns=target), df_prep[target]
    return train_test_split(X, y, test_size=config.TEST_SIZE,
                            random_state=config.RANDOM_STATE, stratify=y)


def train_tree(X_train, y_train, max_depth: int = config.ARBOL_MAX_DEPTH,
               criterion: str = config.ARBOL_CRITERION, class_weight=None) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion,
                                   class_weight=class_weight, random_state=config.RANDOM_STATE)
    return model.fit(X_train, y_train)


def train_knn(X_train, y_train, n_neighbors: int = 5):
    """KNN con estandarización incluida en el pipeline (no hay que escalar a mano)."""
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=n_neighbors)).fit(
        X_train, y_train)


def matriz_y_costo(y_true, y_pred, costo_fp: float = config.COSTO_FP,
                   costo_fn: float = config.COSTO_FN) -> dict:
    """Matriz de confusión (tn, fp, fn, tp) y costo total en USD."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "costo": int(fp * costo_fp + fn * costo_fn)}


def costo_baseline(y_true, costo_fn: float = config.COSTO_FN) -> int:
    """Costo de NO predecir nunca una demora: todas las demoras reales son falsos negativos."""
    return int((np.asarray(y_true) == 1).sum() * costo_fn)


def evaluate(model, X_test, y_test, name: str, verbose: bool = True) -> dict:
    """Imprime el classification_report y devuelve métricas + matriz de confusión + costo."""
    pred = model.predict(X_test)
    rep = classification_report(y_test, pred, output_dict=True, zero_division=0)
    if verbose:
        print(f"\n--- Evaluación: {name} ---")
        print(classification_report(y_test, pred, zero_division=0))
    return {"modelo": name, "accuracy": rep["accuracy"],
            "precision_demora": rep["1"]["precision"],
            "recall_demora": rep["1"]["recall"], "f1_demora": rep["1"]["f1-score"],
            **matriz_y_costo(y_test, pred)}


def buscar_arbol(X_train, y_train, X_test, y_test, profundidades=range(3, 9),
                 criterios=("gini", "entropy"), balanceos=(None, "balanced")) -> pd.DataFrame:
    """Grilla de árboles (profundidad x criterio x balanceo) con métricas y costo en USD.

    Igual que en el informe, se evalúa sobre el conjunto de prueba. Ojo: elegir
    hiperparámetros con el mismo conjunto con el que se reporta el resultado final
    es optimista; lo ideal es validación cruzada o un conjunto de validación aparte.
    """
    filas = []
    for crit in criterios:
        for bal in balanceos:
            for depth in profundidades:
                m = train_tree(X_train, y_train, depth, crit, bal)
                filas.append({"criterio": crit, "balanceo": bal or "no", "profundidad": depth,
                              **{k: v for k, v in evaluate(m, X_test, y_test, "", verbose=False).items()
                                 if k not in ("modelo", "f1_demora")}})
    return pd.DataFrame(filas)


def buscar_knn(X_train, y_train, X_test, y_test, ks=(1, 3, 5, 7, 9, 11, 15)) -> pd.DataFrame:
    """Accuracy, recall y costo de KNN para distintos valores de K."""
    filas = []
    for k in ks:
        m = evaluate(train_knn(X_train, y_train, k), X_test, y_test, "", verbose=False)
        filas.append({"k": k, **{c: m[c] for c in ("accuracy", "precision_demora", "recall_demora", "costo")}})
    return pd.DataFrame(filas)


def asignar_beneficio(demora_predicha: pd.Series, programa_millas: pd.Series) -> pd.Series:
    """Sala VIP / Voucher de comida / Ninguno según predicción y programa de millas."""
    es_socio = programa_millas.map(normalizar_si_no)
    return pd.Series(
        np.select([demora_predicha & es_socio, demora_predicha],
                  ["Sala VIP", "Voucher comida"], default="Ninguno"),
        index=demora_predicha.index)


def predecir_casos_nuevos(model, pre: FlightPreprocessor, casos_nuevos: pd.DataFrame) -> pd.DataFrame:
    """Devuelve los casos nuevos (sin nulos) con la columna 'Predicho_demora_vuelo' (SI/NO)."""
    X = pre.transform(casos_nuevos)
    out = casos_nuevos.loc[X.index].copy()
    out["Predicho_demora_vuelo"] = np.where(model.predict(X) == 1, "SI", "NO")
    return out


def armar_entregable(casos_predichos: pd.DataFrame, relacion: pd.DataFrame,
                     clientes: pd.DataFrame) -> pd.DataFrame:
    """Une predicción + pasajeros + programa de millas y asigna el beneficio."""
    df = relacion.merge(casos_predichos, on="id_vuelo", how="inner")
    df = df.merge(clientes[["id_cliente", "programaMillas"]], on="id_cliente", how="left")
    df["Tipo beneficio"] = asignar_beneficio(df["Predicho_demora_vuelo"] == "SI",
                                             df["programaMillas"])
    cols = [c for c in config.ENTREGABLE_COLS if c in df.columns]
    return df[cols]
