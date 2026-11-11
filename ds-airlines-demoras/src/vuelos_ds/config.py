"""Configuración central: rutas, nombres de archivos y listas de columnas.

Todo lo que antes estaba copiado en cada notebook (ruta de Drive, separador,
encoding, listas de variables) vive acá, en un solo lugar.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COLAB_DATA_DIR = Path("/content/drive/MyDrive/TpCienciaDatos")


def get_data_dir() -> Path:
    """Carpeta con los CSV. Prioridad: variable de entorno > Drive (Colab) > ./data."""
    env = os.environ.get("VUELOS_DATA_DIR")
    if env:
        return Path(env)
    if COLAB_DATA_DIR.exists():
        return COLAB_DATA_DIR
    return REPO_ROOT / "data"


def get_output_dir() -> Path:
    """Carpeta donde se guardan los entregables (Excel, etc.)."""
    out = Path(os.environ.get("VUELOS_OUTPUT_DIR", REPO_ROOT / "outputs"))
    out.mkdir(parents=True, exist_ok=True)
    return out


CSV_SEP = ";"
CSV_ENCODING = "latin1"

FILES = {
    "vuelos": "Vuelos.csv",
    "clientes": "Clientes.csv",
    "casos_nuevos": "casosNuevos.csv",
    "relacion": "casosNuevos_Clientes.csv",
    "clientes_demorados": "Clientes_Demorados_SinDuplicados.csv",
}

RANDOM_STATE = 42
TEST_SIZE = 0.3

TARGET = "demora"

VUELOS_CATEGORICAS = [
    "aeropuerto_origen",
    "aeropuerto_destino",
    "dia_semana",
    "condiciones_climaticas",
    "congestion_aerea",
    "tipo_avion",
]
VUELOS_BOOLEANAS = ["temporada_alta"]
VUELOS_NUMERICAS = [
    "hora_salida_programada",
    "distancia_vuelo",
    "ocupacion_vuelo",
    "visibilidad",
    "tiempo_estimado_vuelo",
]
VUELOS_DECIMAL_COMA = ["ocupacion_vuelo", "visibilidad"]
VUELOS_COLS_DESCARTADAS = ["id_vuelo", "puerta_embarque", "tiempo_estimado_vuelo"]

COSTO_FP = 80
COSTO_FN = 50

ARBOL_MAX_DEPTH = 5
ARBOL_CRITERION = "entropy"

CLIENTES_NUMERICAS = [
    "edad",
    "cant_vuelos",
    "gasto_acumulado",
    "gasto_acumulado_extra",
    "cantidad_millas",
    "ingreso_mensual",
    "anticipacion_compra_promedio",
]
CLIENTES_DECIMAL_COMA = ["gasto_acumulado", "gasto_acumulado_extra"]

KMEANS_COLS_DESCARTADAS = ["id_cliente", "sexo", "provincia", "canal_compra"]
KMEANS_ONEHOT = ["ocupacion", "clase_preferida"]
KMEANS_PERFIL = CLIENTES_NUMERICAS + ["programaMillas"]

ENTREGABLE_COLS = [
    "id_vuelo",
    "id_cliente",
    "aeropuerto_origen",
    "aeropuerto_destino",
    "hora_salida_programada",
    "dia_semana",
    "distancia_vuelo",
    "condiciones_climaticas",
    "congestion_aerea",
    "tipo_avion",
    "ocupacion_vuelo",
    "temporada_alta",
    "puerta_embarque",
    "visibilidad",
    "tiempo_estimado_vuelo",
    "Predicho_demora_vuelo",
    "Tipo beneficio",
]
