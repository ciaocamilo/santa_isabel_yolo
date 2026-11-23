# Escenas de ejemplo

Dos adquisiciones reales Sentinel-2 L2A del Nevado Santa Isabel para la demostración de dos fechas. Son lo único
que la aplicación necesita empaquetar para funcionar en Render sin la caché del generador.

| Carpeta | Adquisición (UTC) | Split en v0003 | Papel |
|---|---|---|---|
| `20240815_T18NVL_20240815T152651_20240815T152653` | 2024-08-15 15:31:49 | validación | fecha inicial |
| `20260810_T18NVL_20260810T152649_20260810T152745` | 2026-08-10 15:31:46 | prueba | fecha final |

Criterio de elección (`cuaderno.elegir_par_demo`): mismo mes del calendario en años distintos, ninguna escena de
entrenamiento, mayor intervalo y mayor observabilidad. `demo.json` registra el par.

Cada carpeta contiene:

- `rgb/<clave>.png`: PNG RGB original del generador (B4/B3/B2, reflectancia 0–0,60 → 0–255), sin modificar.
- `calidad.tif`: 0 = exterior o sin observación, 1 = observado y no visible, 2 = observado y visible (SCL 4, 5, 6, 11).
- `soporte_png.tif`: píxeles con datos válidos en el PNG.
- `metadata.json`: fecha, asset de Earth Engine, malla (EPSG:32618, 10 m, 545 × 598), receta, hashes y origen.

Se copian con `python tools/preparar_ejemplos.py 2024-08 2026-08` desde la caché del generador mensual (solo
lectura); la herramienta comprueba que los hashes coinciden con los del productor.

**Fuente y crédito.** Contiene datos modificados de Copernicus Sentinel-2 (2024, 2026) procesados mediante Google
Earth Engine (`COPERNICUS/S2_SR_HARMONIZED`). Los datos Sentinel son de uso libre, completo y abierto con
atribución.

Las etiquetas de estas fechas en el dataset son pseudoetiquetas automáticas (NDSI) sin revisión humana. La escena
inicial perteneció a validación (se usó para elegir checkpoint y umbral, no para ajustar pesos); la final es de
prueba.
