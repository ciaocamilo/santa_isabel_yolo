# Paquetes del modelo

Cada carpeta `models/<id>/` es un **paquete inmutable** publicado por la celda de exportación de
`notebooks/01_entrenar_evaluar.ipynb`. El archivo `models/ACTIVO` contiene el identificador del paquete que
carga la aplicación (también se puede fijar con la variable `SANTA_ISABEL_PAQUETE`).

```
models/<id>/
  modelo.json                    contrato (ver abajo)
  pesos/best.pt                  pesos ajustados · backend de referencia (Ultralytics + PyTorch)
  pesos/best.onnx                exportación para el servicio en CPU (ONNX Runtime, NMS embebido)
  metricas.json                  selección en validación, prueba temporal, protocolo y bitácora
  metricas_test_por_escena.csv   métricas por fecha de la prueba temporal
  roc_test.png, roc_val.png      curvas ROC con el punto operativo
  SHA256SUMS                     SHA-256 de todos los archivos
```

## Contrato (`modelo.json`, esquema 1.0)

| Campo | Contenido |
|---|---|
| `tarea`, `clase` | `segment`, `{0: nieve_hielo_visible}` |
| `variante`, `receta` | `rgb` o `espectral`; versión y codificación radiométrica del productor |
| `entrada` | `imgsz`, alineación (letterbox centrado, relleno 114), orden de canales por backend, teselas |
| `posproceso` | umbral de propuestas, **umbral operativo**, IoU del NMS, tope de detecciones, definición de la puntuación |
| `backends` | archivo, versiones y SHA-256 de cada backend; `backend_referencia` y `backend_servicio` |
| `dataset` | versión, huella de contenido, hash del manifiesto, escenas por split; selección en `val`, prueba en `test` |
| `referencia` | origen de las etiquetas (`automatico_ndsi`), revisión humana y aviso de alcance |
| `seleccion`, `entrenamiento`, `versiones` | criterio y candidatos, corrida, dispositivo, pico de memoria, versiones |
| `archivos` | SHA-256 de cada archivo del paquete |

`cargar_paquete(ruta)` valida esquema, tarea, clase, variante, umbrales, hashes y backend antes de inferir. Si
algo falla, el error indica cómo resolverlo y la aplicación lo muestra; nunca presenta resultados de nieve con
pesos genéricos.

Un paquete no se edita. Para cambiar pesos, resolución o umbral se ejecuta de nuevo el notebook, que publica otro
identificador y actualiza `ACTIVO`.

## Tamaño y publicación

Los pesos del modelo nano ocupan unos 6 MB (`.pt`) y 11 MB (`.onnx`): se versionan en Git junto al código. Si un
modelo mayor superara el límite del repositorio, publíquelo como activo de un GitHub Release con URL fija,
descárguelo en la construcción y deje que `cargar_paquete` verifique su hash antes de activarlo.
`.gitattributes` impide que Git convierta finales de línea dentro de `models/` y `examples/`, porque cambiaría
los hashes.

## Licencias

- Pesos ajustados: derivados de `yolo26n-seg.pt` de Ultralytics (AGPL-3.0). Su redistribución y el uso del
  servicio quedan sujetos a esa licencia, salvo licencia comercial de Ultralytics.
- Datos de entrenamiento y ejemplos: datos modificados de Copernicus Sentinel-2 procesados con Google Earth
  Engine (uso libre con atribución).
