"""Tests de humo del pipeline con datos sintéticos (no requieren los CSV reales)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vuelos_ds import clustering, config, data, dimred, eda, modeling

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    spec = importlib.util.spec_from_file_location("gen", ROOT / "scripts" / "generate_sample_data.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    gen.main(d)
    return d


def test_clean_vuelos_es_idempotente(data_dir):
    df = data.load_vuelos(data_dir)
    assert df["visibilidad"].dtype == float and df["hora_salida_programada"].dtype == int
    pd.testing.assert_frame_equal(df, data.clean_vuelos(df))


def test_normalizar_si_no():
    assert all(data.normalizar_si_no(v) for v in ["Sí", "si", " SÍ ", "SI"])
    assert not any(data.normalizar_si_no(v) for v in ["No", "no", None, np.nan])


def test_clientes_coma_decimal(data_dir):
    cl = data.load_clientes(data_dir)
    assert cl["gasto_acumulado"].dtype == float


def test_preprocessor_codifica_igual_en_train_y_nuevos(data_dir):
    vuelos = data.load_vuelos(data_dir)
    nuevos = data.load_casos_nuevos(data_dir)
    pre = modeling.FlightPreprocessor().fit(vuelos)
    Xn = pre.transform(nuevos)
    assert list(Xn.columns) == pre.feature_cols_
    assert "tiempo_estimado_vuelo" not in pre.feature_cols_
    raro = nuevos.copy()
    raro.loc[raro.index[0], "tipo_avion"] = "Avion Nuevo"
    assert pre.transform(raro).iloc[0]["tipo_avion"] == -1


def test_modelos_y_entregable(data_dir):
    vuelos = data.load_vuelos(data_dir)
    nuevos = data.load_casos_nuevos(data_dir)
    pre = modeling.FlightPreprocessor()
    X_tr, X_te, y_tr, y_te = modeling.split_train_test(pre.fit_transform(vuelos))
    assert len(X_tr) + len(X_te) == len(vuelos)
    arbol, knn = modeling.train_tree(X_tr, y_tr), modeling.train_knn(X_tr, y_tr)
    m = modeling.evaluate(arbol, X_te, y_te, "Árbol", verbose=False)
    assert 0.5 < m["accuracy"] <= 1 and modeling.evaluate(knn, X_te, y_te, "KNN", verbose=False)
    pred = modeling.predecir_casos_nuevos(arbol, pre, nuevos)
    final = modeling.armar_entregable(pred, data.load_relacion(data_dir), data.load_clientes(data_dir))
    assert set(final["Tipo beneficio"]) <= {"Sala VIP", "Voucher comida", "Ninguno"}
    assert (final["Tipo beneficio"] == "Sala VIP").any()
    assert (final.loc[final["Predicho_demora_vuelo"] == "NO", "Tipo beneficio"] == "Ninguno").all()


def test_costos_y_baseline():
    y = pd.Series([1, 1, 0, 0, 0])
    pred = np.array([1, 0, 1, 0, 0])
    r = modeling.matriz_y_costo(y, pred)
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (1, 1, 1, 2)
    assert r["costo"] == 1 * 80 + 1 * 50
    assert modeling.costo_baseline(y) == 2 * 50


def test_busquedas(data_dir):
    vuelos = data.load_vuelos(data_dir)
    X_tr, X_te, y_tr, y_te = modeling.split_train_test(modeling.FlightPreprocessor().fit_transform(vuelos))
    grilla = modeling.buscar_arbol(X_tr, y_tr, X_te, y_te, profundidades=[3, 5])
    assert len(grilla) == 2 * 2 * 2 and {"costo", "recall_demora", "profundidad"} <= set(grilla.columns)
    assert list(modeling.buscar_knn(X_tr, y_tr, X_te, y_te, ks=[3, 5])["k"]) == [3, 5]


def test_asignar_beneficio_reglas():
    pred = pd.Series([True, True, False])
    socio = pd.Series(["Sí", "No", "Sí"])
    assert modeling.asignar_beneficio(pred, socio).tolist() == ["Sala VIP", "Voucher comida", "Ninguno"]


def test_kmeans(data_dir):
    cl = data.load_clientes_demorados(data_dir)
    df, X = clustering.prepare_kmeans_features(cl)
    assert X.shape[0] == len(df) and np.allclose(X.mean(axis=0), 0, atol=1e-8)
    assert set(clustering.silhouettes(X, range(2, 4), sample_size=500)) == {2, 3}
    df["Cluster"] = clustering.fit_kmeans(X, 3).labels_
    assert clustering.perfil_clusters(df).shape[0] == 3
    assert not clustering.poder_discriminante(df).isna().any()


def test_pca_y_bartlett(data_dir):
    cl = data.load_clientes_demorados(data_dir)
    pca, X, Xp = dimred.fit_pca(cl)
    assert Xp.shape[1] == len(config.CLIENTES_NUMERICAS)
    assert dimred.tabla_varianza(pca)["% acumulada"].iloc[-1] == pytest.approx(100)
    chi2, p = dimred.bartlett(X)
    assert chi2 > 0 and 0 <= p <= 1


def test_eda_helpers(data_dir):
    v = data.load_vuelos(data_dir)
    assert eda.tasa_demora_por(v, "congestion_aerea").between(0, 100).all()
    assert set(eda.outliers_iqr(v, ["distancia_vuelo"]).columns) == {"outliers", "porcentaje"}
