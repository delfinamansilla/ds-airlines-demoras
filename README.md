# TP Ciencia de Datos: Demoras aéreas (DS Airlines)

**Cátedra:** Ciencia de Datos (K502) · UTN · Grupo 01
**Profesoras:** Silvia Scime, Martina Di Carlo
**Trabajo:** TPI Etapa 2, "Demoras Aéreas" (entrega 05/06/2026)

**Integrantes**
- Arfuso, María de los Ángeles
- Diodati, Regina
- Mansilla, Delfina
- Maurutto, Francesca
- Uribarri, Sebastian

## Sobre este repositorio

El trabajo se desarrolló originalmente en **notebooks de Google Colab** (cada integrante trabajó en el suyo) y en **IBM SPSS**. Este repositorio es una **recopilación y reorganización de la parte hecha en Python**: reúne los notebooks en un solo proyecto, pasa el código repetido a módulos compartidos y corrige algunos errores, sin cambiar la metodología del informe. Los notebooks originales de Colab se conservan sin modificar en [`archive/`](archive/).

El análisis completo (todos los modelos, justificaciones y conclusiones) está en el informe del TP. Acá sólo está el código en Python.

## El problema

DS Airlines quiere anticipar las demoras de sus vuelos y compensar a los pasajeros afectados al menor costo posible:

- **Etapa 1:** predecir si un vuelo se va a demorar y asignar un beneficio: **Sala VIP** a los socios del programa de millas y **voucher de comida** al resto. Los errores tienen costo: falso positivo = 80 USD (beneficio entregado sin necesidad), falso negativo = 50 USD (demora no anticipada).
- **Etapa 2:** segmentar a los pasajeros de vuelos con demora prevista (clustering) y reducir la dimensionalidad de sus variables (PCA).

## Herramientas utilizadas

| Análisis | Herramienta | ¿Está en este repo? |
|---|---|---|
| EDA de vuelos | Python (Colab) | Sí |
| Árbol de Decisión | Python (Colab); replicado en IBM SPSS Modeler | Sí (versión Python) |
| KNN | Python (Colab) | Sí |
| Regresión Logística | IBM SPSS Statistics | No |
| Análisis Discriminante (LDA) | Python (Colab) e IBM SPSS Statistics | No |
| K-Means | Python (Colab) | Sí |
| PCA | Python (Colab) | Sí |
| Clustering bietápico | IBM SPSS Statistics | No |
| Clustering jerárquico | IBM SPSS Statistics | No |

## Estructura

```
├── notebooks/
│   ├── 01_eda_vuelos.ipynb          # análisis exploratorio de Vuelos.csv
│   ├── 02_modelado_demoras.ipynb    # árbol vs KNN, evaluación con costos y entregable (Excel)
│   ├── 03_kmeans_clientes.ipynb     # segmentación de clientes con K-Means
│   └── 04_pca_clientes.ipynb        # componentes principales
├── src/vuelos_ds/                   # código compartido por los notebooks
│   ├── config.py                    # rutas, nombres de archivos, columnas y costos
│   ├── data.py                      # carga y limpieza de CSV
│   ├── eda.py                       # estadísticos del EDA
│   ├── plots.py                     # gráficos reutilizables
│   ├── modeling.py                  # preprocesamiento, modelos, evaluación y entregable
│   ├── clustering.py                # K-Means
│   └── dimred.py                    # PCA y test de Bartlett
├── tests/                           # tests con datos sintéticos
├── scripts/generate_sample_data.py  # genera CSV sintéticos para probar el repo
├── data/                            # acá van los CSV (no se versionan)
└── archive/                         # notebooks originales de Colab, sin modificar
```

## Cómo correrlo

```bash
pip install -e ".[dev]"          # instala el paquete y las dependencias
# copiar los CSV en data/ (ver data/README.md), o definir VUELOS_DATA_DIR
jupyter lab notebooks/
```

**Sin los datos reales:** `python scripts/generate_sample_data.py` crea CSV sintéticos en `data/`. Sirven para comprobar que el código corre, pero están inventados y sus resultados no significan nada. Para verificar el código: `pytest`.

**En Colab:**
```python
!git clone https://github.com/delfinamansilla/ds-airlines-demoras.git
!pip install -q -e ./ds-airlines-demoras
```
Si los CSV están en `MyDrive/TpCienciaDatos/` y Drive está montado, se detectan solos.

## Datos

Los CSV son material de la cátedra y **no se incluyen** en el repo. Los archivos esperados y su esquema están en [`data/README.md`](data/README.md).

> El clustering y el PCA se hicieron sobre los pasajeros de vuelos que la **versión inicial** del árbol de la Etapa 1 predijo como demorados (ver sección 10 del informe). Ese archivo (`Clientes_Demorados_SinDuplicados.csv`) se armó fuera de estos notebooks.

## Resultados principales

*(Corrida con los datos reales; detalle en las conclusiones de cada notebook y en el informe.)*

| Análisis | Resultado |
|---|---|
| Demoras | 31,85 % de vuelos demorados. Clima (tormenta 62 %) y congestión aérea (alta 47 %) son los factores más fuertes. |
| Árbol de Decisión (entropía, prof. 5) | Accuracy 0,71, precisión 0,59, recall 0,27. Costo: 73.490 USD. Es el modelo elegido en el informe. |
| KNN (k = 5) | Accuracy 0,66, precisión 0,44, recall 0,27. Costo: 91.880 USD. |
| K-Means | K = 3 sobre 22.391 clientes: un segmento premium (~54 vuelos, ~48 mil millas) y dos de menor uso. Silueta 0,17 (estructura débil). |
| PCA | 4 componentes explican el 83,6 % de la varianza. PC1 (44 %) = gasto, vuelos y millas. |
| Segmentación elegida en el informe | Clustering bietápico (SPSS, 10 variables): clúster Elite (14,7 %) y Estándar (85,3 %). |

## Limitaciones

- Con los costos del informe (FP = 80, FN = 50 USD), **no predecir nunca una demora costaría 71.650 USD**, menos que cualquiera de los modelos. Marcar un vuelo como demorado sólo conviene si la precisión supera ~61,5 %; el árbol tiene 59,4 %.
- La configuración del árbol se eligió evaluando sobre el conjunto de prueba, lo que es optimista. Lo ideal sería validación cruzada.
- El recall es bajo (27 %): el modelo deja pasar casi tres de cada cuatro demoras.
- La silueta de K-Means (0,17) indica clusters que se solapan.

## Qué cambió respecto de los notebooks originales

**Código repetido, ahora en un solo lugar**
- Montaje de Drive, `read_csv` con `sep`/`encoding` y listas de columnas, copiados en cada notebook → `config.py` y `data.py`.
- Conversión de coma decimal y de `HH:MM` (5 copias) → `data.clean_vuelos`.
- El par de gráficos "conteo + % de demora" repetido para 5 variables → `plots.plot_counts_and_rate`.
- 7 celdas de scatter de componentes (una duplicada) → `plots.plot_pc_pairs`.
- Las 7 variables numéricas de clientes, definidas por separado en K-Means y PCA → `config.CLIENTES_NUMERICAS`.

**Alineado con la versión final del informe**
- Árbol con entropía y profundidad 5, con grilla de búsqueda y costo en USD (`modeling.buscar_arbol`).
- `tiempo_estimado_vuelo` se descarta del modelado (correlación 0,9987 con `distancia_vuelo`).
- KNN con tabla de K y evaluación por costo (`modeling.buscar_knn`).

**Correcciones**
- **Codificación de casos nuevos:** antes se reajustaba el `LabelEncoder` con los casos nuevos. Ahora el preprocesador se ajusta una vez con el entrenamiento (`modeling.FlightPreprocessor`).
- **Beneficio "Sala VIP":** la comparación era `programaMillas == "SI"`, pero el CSV trae `"Sí"` con tilde. Ahora se normalizan tildes y mayúsculas, con un test que lo cubre.
- **Scatter PC3 vs PC4:** salía con los ejes rotulados como "Componente 2" y "4".
- **Test de Bartlett:** se calcula con scipy, sin depender de `factor_analyzer`.
- Se eliminaron celdas vacías, código viejo comentado y títulos numerados dos veces.
