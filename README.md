# Nieve y hielo visible en el Nevado Santa Isabel · YOLO Segmentation + Streamlit

Un usuario carga una escena Sentinel-2 del Nevado Santa Isabel y obtiene el contorno de la nieve/hielo visible y
su superficie estimada en hectáreas. Con dos fechas obtiene un mapa que localiza dónde desapareció, apareció o
persistió esa cobertura, con la pérdida en ha y como porcentaje del área inicial observable.

Dos puntos de entrada comparten el mismo código (`src/santa_isabel`):

- `notebooks/01_entrenar_evaluar.ipynb`: audita el dataset, entrena en la GPU, evalúa y publica el paquete.
- `app.py`: aplicación Streamlit en español que consume ese paquete, en local o en Render.

> **Alcance.** La clase reúne nieve e hielo visibles. Una reducción entre dos fechas puede incluir nieve
> estacional, sombras y errores del modelo: se informa como **pérdida observada de cobertura**. Dos fechas no
> miden balance de masa ni permiten atribuir una tendencia glaciar. Las etiquetas del dataset son pseudoetiquetas
> automáticas (reglas NDSI) sin revisión humana: las métricas son concordancia con ellas.

## Estado (bitácora)

| Nivel | Estado |
|---|---|
| Probado con fixture | Sí: `pytest` (hitos A–D y servicio), sin GEE, pesos ni dataset. |
| Entrenado en RTX | Sí: NVIDIA GeForce RTX 3050 Laptop (4 GB), corrida `rgb_v0003_20261122_2200`. |
| Evaluado con datos reales | Sí: dataset `v0003` (28 escenas), prueba temporal aplicada una vez. Referencia automática. |
| Verificado en Linux CPU limpio | Sí: clon limpio en Ubuntu 24.04 (WSL) con `requirements-app.txt`. |
| Verificado en Render | **Pendiente**: no se dispuso de una cuenta. El repositorio queda listo (`render.yaml`). |
| Experimento espectral (NDSI/NDVI/B4) | Pendiente: se priorizó la corrida RGB completa y su demostración. |

## Estructura

```
santa_isabel_yolo/
  app.py                         aplicación Streamlit
  notebooks/01_entrenar_evaluar.ipynb
  src/santa_isabel/
    escenas.py                   leer_escena(archivos, perfil) -> Escena · receta radiométrica del productor
    dataset.py                   auditoría de la versión, YAML locales, caché del generador
    inferencia.py                preparar_entrada(escena, contrato) · predecir(imagen, paquete) -> Prediccion
    areas.py                     medir(escena, prediccion) -> Medicion · comparar(anterior, posterior, config) -> Cambio
    evaluacion.py                métricas por píxel, ROC/AUC por conteos, errores de superficie
    paquete.py                   cargar_paquete(ruta) -> Modelo · publicar_paquete
    flujo.py                     analizar_escena, comparar_fechas, evaluación de splits, protocolo de test
    entrenamiento.py, cuaderno.py, visual.py, config.py, entorno.py
  configs/experimento.toml       configuración (dataset v0003 por defecto)
  tests/                         fixtures deterministas y casos de aceptación
  examples/                      dos fechas reales con máscara de calidad, fuente y crédito
  models/                        paquete publicado (contrato, pesos, métricas, ROC, hashes) y puntero ACTIVO
  reports/<run_id>/              métricas CSV/JSON, figuras, capturas e informe (se genera en local, fuera de Git)
  tools/                         utilidades (entorno, notebook, ejemplos, verificación, informe, capturas)
  requirements-app.txt · environment-train.yml · render.yaml · entorno/
  data/datasets/v0003/           dataset extraído (uso local, fuera de Git)
```

## 1. Instalación

### Aplicación (CPU, sin torch ni Ultralytics)

```
python -m venv .venv            # Python 3.11
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-app.txt
streamlit run app.py
```

### Entrenamiento (`yolo_gee`, GPU NVIDIA)

El entorno parte de una réplica del entorno conda `gee` (Python 3.11.15, conda-forge), que se conserva intacto.
`environment-train.yml` documenta la receta exacta usada y una receta portable:

```
conda env create -f environment-train.yml
conda activate yolo_gee
python tools/configurar_entorno.py     # capa secuencial de MKL y kernel «Python (yolo_gee)»
python -m pip check
```

Versiones resueltas en el equipo de entrenamiento (Windows 11, driver NVIDIA 621.14): Python 3.11.15,
`torch 2.15.0+cu132`, `torchvision 0.29.2+cu132` (índice `https://download.pytorch.org/whl/cu132`, el que indica
el selector oficial de PyTorch para la versión estable), Ultralytics 8.4.216, ONNX 1.23.2, ONNX Runtime 1.32.0,
Rasterio 1.4.4, NumPy 2.4.6. Se verifican `import ee` (1.7.35) y `import geemap` (0.37.1).

**Kernel.** En VS Code o Jupyter seleccione **Python (yolo_gee)**. La primera celda imprime el intérprete, torch,
CUDA, GPU y VRAM, ejecuta una operación en `cuda:0` y se detiene con un diagnóstico si CUDA no está disponible.

## 2. Dataset

Extraiga el ZIP del generador mensual (SDD 03) en `data/datasets/`; el ZIP ya contiene la carpeta `v0003/`:

```
data/datasets/v0003/{manifiesto.csv, configuracion.toml, resumen.json}
data/datasets/v0003/{rgb,espectral}/{data.yaml, images/, labels/}
```

La versión por defecto está en `configs/experimento.toml` (`[dataset].version`) y la raíz se resuelve desde la
carpeta del proyecto, no desde el directorio de trabajo. Los `data.yaml` traen un `path:` absoluto de otro equipo:
el notebook genera copias locales en `runs/<run_id>/data_yaml/` cambiando solo esa línea. En el manifiesto, el
marcador literal `<variante>` de la columna `etiqueta` se sustituye por `rgb` o `espectral`.

**Caché del generador (opcional).** El ZIP no trae máscaras de calidad ni GeoTIFF. Con la caché por escena del
generador se evalúa por píxel observable y se copian los ejemplos. Indique su ruta en `configs/local.toml` (no se
versiona) o en `SANTA_ISABEL_CACHE`:

```toml
[dataset]
raiz_cache = "C:/ruta/al/generador/salida/cache_escenas"
```

Sin caché el notebook funciona igual, pero las métricas se rotulan «concordancia con pseudoetiquetas sobre toda
la imagen, sin control de observabilidad».

## 3. Notebook

```
python tools/construir_notebook.py        # solo si cambia el generador del cuaderno
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=yolo_gee ^
        --ExecutePreprocessor.timeout=-1 notebooks/01_entrenar_evaluar.ipynb
```

o ábralo y ejecute todas las celdas con el kernel **Python (yolo_gee)**. Tarda unos 12 minutos en una RTX 3050.
Bloques: configuración y dispositivo → auditoría y exploración → referencias → entrenamiento → selección con
validación → prueba temporal (una sola vez) → exportación, paquete e inferencia de ida y vuelta.

Salidas: `runs/<run_id>/` (corridas intermedias) y `reports/<run_id>/` (evidencias), ambas fuera de Git;
`models/<id>/` (paquete) y `examples/` (dos escenas reales), que sí se versionan.

## 4. Aplicación

```
conda activate yolo_gee      # o el entorno de requirements-app.txt
streamlit run app.py
```

- **Analizar una imagen**: cargue los archivos o pulse «Cargar ejemplo del Santa Isabel» y después **Analizar**.
- **Comparar dos fechas**: «Cargar ejemplo del Santa Isabel (dos fechas)» y **Comparar**.
- **Modelo y evaluación**: identificador del paquete, dataset, métricas y ROC del paquete activo.

| Carga | Resultado |
|---|---|
| Paquete de escena del generador (ZIP o `metadata.json` + PNG + `calidad.tif` + `soporte_png.tif`) | Hectáreas sobre píxeles observables; apta para comparar. |
| GeoTIFF RGB de 8 bits o multibanda de reflectancia | Hectáreas; sin QA se rotula «área aparente, sin control de nubes» y no se cuantifica la pérdida entre fechas. Si faltan nombres de banda, nivel o fecha, la app los pide. |
| PNG/JPG RGB de 8 bits | Inferencia visual sin hectáreas. Un PNG original del generador recupera malla y QA por su hash. |

Límites de carga publicados y probados: **60 MB** y **4 Mpx** por imagen, ZIP de hasta 40 archivos y 200 MB
expandidos. Las imágenes mayores que 640 px se analizan por teselas a escala nativa y se unen por máximo de
puntuación. Solo se aceptan archivos de escena, nunca pesos.

Variables: `SANTA_ISABEL_PAQUETE` (paquete a cargar), `SANTA_ISABEL_BACKEND` (`onnx` o `ultralytics`),
`SANTA_ISABEL_CACHE`, `SANTA_ISABEL_RAIZ`.

## 5. Pruebas

```
python -m pytest                       # 69 pruebas; las que necesitan datos o paquete reales se omiten si faltan
python tools/verificar_paquete.py      # reabre el paquete en un proceso limpio y analiza los ejemplos en CPU
python tools/prueba_servicio.py        # arranque con PORT, health check, dos sesiones, subida real y demo
```

| Hito | Qué cubre | Con qué datos |
|---|---|---|
| A · Desarrollo paralelo | Lectura, máscara, áreas, comparador; la app abre sin modelo y explica qué necesita | Fixtures (la predicción simulada solo existe en `tests/`) |
| B · Integración | PNG + metadatos frente a GeoTIFF de la misma fecha; bandas incompatibles, fecha ausente, hash erróneo; sin fuga de fechas | Fixtures + v0003 real + caché del generador (28/28 PNG idénticos en ambas variantes) |
| C · Geometría | Ejemplo numérico de aceptación y sus casos QA, fecha invertida, denominador cero, sin intersección, celdas desplazadas, distinto CRS, afín rotada, solapes, letterbox no cuadrado | Fixtures |
| D · Evaluación | AUC 0,75 del caso unitario, una sola clase, empates, píxeles ignorados, protocolo de test | Fixtures |
| E · Entrenamiento real | Lote y corrida en RTX, checkpoint, evaluación y paquete reabierto en proceso limpio | Datos reales |
| F · Servicio | `PORT` configurable, dos sesiones aisladas, subida real, demo de dos fechas | Datos reales, Windows y Linux |

## 6. Resultados medidos (corrida `rgb_v0003_20261122_2200`)

Dataset exacto: `v0003`, huella de contenido `ef44e9dcf906fd08`, ejecución `20261123T012015Z_440a11`; 18/4/6
escenas y 273/7/41 instancias en train/val/test; todas positivas; referencia `automatico_ndsi` sin revisión humana.

| Corrida | Épocas | Tiempo | Pico de memoria GPU |
|---|---|---|---|
| Integración (1 época) | 1 | 21 s | 1 752 MiB asignados |
| Inicial del SDD (50 épocas, paciencia 10, lote 2) | se detuvo en la 12 | 58 s | 2 001 MiB |
| Ajustada (`nbs` = lote, sin parada temprana) | 150 | 574 s (9,6 min) | 1 998 MiB |

La configuración inicial no llega a entrenar: con lote 2 Ultralytics acumula gradientes hasta un lote nominal de
64 (un paso del optimizador cada ~3,5 épocas) y la parada temprana se guía por el mAP de instancia de una
validación de 7 instancias. La corrida ajustada usa un paso por lote y compara checkpoints por IoU de cobertura.

Selección (solo validación): `3_ajustada/epoch100.pt`, `imgsz` 1024, umbral operativo 0,366.

| Conjunto · soporte | Precisión | Recall | F1 | IoU | AUC | MAE área |
|---|---:|---:|---:|---:|---:|---:|
| Validación · píxel observable | 0,860 | 0,872 | 0,866 | 0,764 | 0,969 | 0,48 ha |
| **Prueba temporal · píxel observable** | **0,889** | **0,390** | **0,542** | **0,371** | **0,908** | **5,03 ha** |
| Prueba temporal · observable sin ambiguos | 0,890 | 0,390 | 0,542 | 0,372 | 0,908 | 5,04 ha |
| Prueba temporal · toda la imagen, sin control de observabilidad | 0,888 | 0,390 | 0,542 | 0,371 | 0,908 | 5,03 ha |

Lectura honesta de estas cifras:

- La caída de validación (F1 0,87) a prueba (F1 0,54) muestra que elegir entre 19 checkpoints con 4 escenas de
  validación fue optimista: el IoU de validación osciló entre 0,01 y 0,76 según el checkpoint.
- En prueba, tres escenas salen bien (IoU 0,74–0,88) y dos quedan sin ninguna detección sobre el umbral; los
  parches menores de 25 px casi no se recuperan (1 de 25).
- El comparador NDSI ≥ 0,4 obtiene F1 0,82 contra las mismas pseudoetiquetas, pero es una comparación circular
  (las etiquetas nacen de ese índice) y no demuestra superioridad de ningún método.
- El error medio de superficie en prueba (5,0 ha por escena) es del mismo orden que el cambio medido en la
  demostración: esa diferencia entre dos fechas no se distingue del error del modelo.

Demostración de dos fechas (2024-08-15 → 2026-08-10, soporte común 3 021,87 ha = 93,2 % del AOI): inicial
comparable 9,78 ha, final 4,89 ha, persistencia 4,87 ha, pérdida bruta 4,91 ha (50,2 %), ganancia 0,02 ha, cambio
neto −4,89 ha (−50,0 %). El notebook, la app, ONNX en CPU y PyTorch en CPU dan las mismas cifras.

### Mediciones del servicio

| Medición | Windows 11 · Python 3.11.15 | Linux (Ubuntu 24.04 en WSL) · Python 3.12.3 |
|---|---:|---:|
| Instalación de `requirements-app.txt` | — | 54 s (29 s con caché de pip) |
| torch / Ultralytics instalados | no se cargan | no |
| Carga del modelo ONNX | 0,30 s | 0,23 s |
| Inferencia por escena (545 × 598) | 0,26–0,33 s | 0,56–0,64 s |
| Memoria pico, proceso de verificación (solo ONNX) | 454 MiB | 460 MiB |
| Arranque en frío de Streamlit hasta `/_stcore/health` | 4,9 s | 0,8 s |
| Latencia «Analizar» una escena / «Comparar» dos escenas | 0,6 s / 1,0 s | 0,7 s / 1,6 s |
| Memoria pico con dos sesiones (app + modelo) | 560 MiB | 598 MiB |

PyTorch en CPU (backend de referencia) usa 866 MiB de pico y da el mismo resultado. Las métricas de la corrida
viajan con el paquete (`models/si-rgb-yolo26n-rgb_v0003_20261122_2200/metricas.json`,
`metricas_test_por_escena.csv`, `roc_test.png` y `roc_val.png`). Las mediciones de servicio de esta tabla proceden
de `tools/verificar_paquete.py` y `tools/prueba_servicio.py`, cuya salida se guardó en `reports/` (carpeta local,
no incluida en el repositorio); se reproducen ejecutando esas dos herramientas. **En Render no se ha medido
nada**: repita `python tools/prueba_servicio.py` en el servicio y registre sus valores. Si el plan no dispone de
unos 600 MiB libres, restrinja el tamaño de carga (`[limites]`) o suba de plan y documente la medición real.

## 7. Publicación en Render

1. Suba este repositorio a GitHub (`git remote add origin …`, `git push -u origin main`). `data/`, `runs/`,
   `reports/`, entornos, cachés y `configs/local.toml` están excluidos por `.gitignore`.
2. En Render: **New → Blueprint** y seleccione el repositorio; `render.yaml` define el servicio web.
   - Build: instala `requirements-app.txt` y ejecuta `python tools/verificar_paquete.py --backends onnx`, que
     verifica los hashes del paquete y hace una inferencia real de los dos ejemplos.
   - Start: `streamlit run app.py --server.address 0.0.0.0 --server.port "$PORT" --server.headless true`.
   - Health check: `/_stcore/health`. Python: `PYTHON_VERSION=3.11.15`.
3. Tras el arranque abra la URL, pulse «Cargar ejemplo del Santa Isabel (dos fechas)» y «Comparar»; debe mostrar
   las cifras del apartado 6. Registre arranque en frío, latencias y memoria del plan usado.

El modelo se carga una vez por proceso (`st.cache_resource`), las inferencias se serializan con un candado, cada
sesión tiene su directorio temporal y los temporales antiguos se limpian. Al reiniciar, el servicio se reconstruye
desde los activos versionados; no descarga pesos en cada ejecución.

## 8. Reproducir la demostración

```
conda activate yolo_gee
streamlit run app.py
```

1. Portada: ubicar el Santa Isabel y plantear la pregunta.
2. «Analizar una imagen» → «Cargar ejemplo del Santa Isabel» → **Analizar**: original, contorno, sector no
   observable (gris) y hectáreas.
3. «Comparar dos fechas» → «Cargar ejemplo del Santa Isabel (dos fechas)» → **Comparar**: señalar el parche
   perdido (rojo), leer L y Δ y explicar su diferencia.
4. «Modelo y evaluación»: F1, ROC/AUC y referencia automática.
5. Mostrar una limitación (escenas de prueba sin detección, parches pequeños) y cerrar con lo que permiten dos
   fechas.

Evidencias de la consigna: el notebook escribe en `reports/<run_id>/` las tablas CSV y las figuras; ahí se guardan
también las `capturas/` de la app y el informe. Esa carpeta es local (está en `.gitignore`): en un clon no existe
hasta que se ejecuta el notebook.

```
python tools/capturar_app.py http://127.0.0.1:8501 reports/<run_id>/capturas   # requiere playwright y Edge
python tools/generar_informe.py                                                # informe.md, .html y .pdf
```

`informe.md` se llena solo con archivos de la corrida real (necesita `reports/<run_id>/`); los apartados
«[Completar]» son del autor.

## 9. Diagnóstico

| Síntoma | Causa y solución |
|---|---|
| La primera celda se detiene: «CUDA no está disponible» | Kernel distinto de «Python (yolo_gee)», wheel `+cpu` o driver. Reinstale con el índice `cu132` y revise `nvidia-smi`. |
| `OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll` | NumPy de conda-forge (MKL + OpenMP de LLVM) junto al wheel pip de torch (OpenMP de Intel). `python tools/configurar_entorno.py` fija `MKL_THREADING_LAYER=SEQUENTIAL` en el entorno. No use `KMP_DUPLICATE_LIB_OK`. |
| `The EPSG code is unknown … proj.db … another PROJ installation` | Una `PROJ_LIB` global (p. ej. PostgreSQL/PostGIS) apunta a otro PROJ. `santa_isabel` fija la del entorno al importarse: impórtelo antes que `rasterio`. |
| `CUDA out of memory` | Reduzca primero `[entrenamiento].lote`; con lote 2 e `imgsz` 1024 el pico medido es 2,0 GB. Si reduce `imgsz`, revise la tabla de tamaño mínimo de parches del notebook. |
| El entrenamiento «termina» en pocas épocas sin aprender | Es la configuración inicial: use `[entrenamiento.ajustado]` (`nbs` = lote, `paciencia` 0). |
| `conda create --clone gee` falla con `SpecNotFoundInPackageCache … x264` | Error de conda con versiones con época. Use `mamba create --file entorno/gee-conda-explicit-win64.txt`. |
| La app dice que no hay modelo publicado | Falta `models/ACTIVO` o la carpeta del paquete: ejecute el notebook hasta la exportación. |
| «Hash erróneo» al cargar el paquete | Un archivo del paquete cambió (p. ej. conversión de finales de línea al clonar sin `.gitattributes`). Restaure el paquete publicado. |

## 10. Licencias y fuentes

- Código de este repositorio: usa Ultralytics YOLO (AGPL-3.0); los pesos ajustados derivan de `yolo26n-seg.pt`.
- Datos: contiene datos modificados de Copernicus Sentinel-2 procesados mediante Google Earth Engine.
- Especificaciones de origen (no incluidas en el repositorio): `SDD_03_Dataset_Mensual_Automatico_YOLO.md`
  (contrato del productor) y `SDD_04_Santa_Isabel_YOLO_Streamlit.md` (este proyecto).

## Autor

Ing. Camilo Andrés Castañeda Galindo · Estudiante de Maestría en Inteligencia Artificial · 2026
