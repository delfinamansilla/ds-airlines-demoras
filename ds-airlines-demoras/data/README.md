# Datos

Los CSV **no se versionan** (son material de la cátedra). Copialos en esta carpeta, o apuntá a
otra con la variable de entorno `VUELOS_DATA_DIR`. En Colab, si existe
`/content/drive/MyDrive/TpCienciaDatos`, se usa automáticamente.

Todos van con separador `;` y encoding `latin1`.

| Archivo | Contenido | Usado en |
|---|---|---|
| `Vuelos.csv` | 15.000 vuelos históricos con la columna objetivo `demora` (0/1) | notebooks 01 y 02 |
| `casosNuevos.csv` | 5.000 vuelos sin `demora`, para predecir | notebook 02 |
| `casosNuevos_Clientes.csv` | relación `id_vuelo` ↔ pasajero (`id_pasajero` o `id_cliente`) | notebook 02 |
| `Clientes.csv` | clientes (`programaMillas`, gastos, etc.) | notebooks 02 y 04 |
| `Clientes_Demorados_SinDuplicados.csv` | clientes con vuelos demorados, sin duplicados | notebooks 03 y 04 |

> `Clientes_Demorados_SinDuplicados.csv` se armó fuera de los notebooks originales. Si el
> script que lo genera existe, conviene agregarlo a `scripts/`.

## Probar sin los datos reales

```bash
python scripts/generate_sample_data.py          # crea CSV sintéticos en data/
```

Los datos sintéticos tienen el mismo esquema pero están **inventados**: sirven para verificar
que el código corre, no para sacar conclusiones.
