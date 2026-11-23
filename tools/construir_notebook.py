"""Genera notebooks/01_entrenar_evaluar.ipynb (sin salidas). Ejecutarlo después con el kernel yolo_gee:

    python tools/construir_notebook.py
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=yolo_gee \
        --ExecutePreprocessor.timeout=-1 notebooks/01_entrenar_evaluar.ipynb
"""
from pathlib import Path

import nbformat as nbf

RAIZ = Path(__file__).resolve().parents[1]
celdas = []


def md(texto):
    celdas.append(nbf.v4.new_markdown_cell(texto.strip("\n")))


def code(texto):
    celdas.append(nbf.v4.new_code_cell(texto.strip("\n")))


md("""
# Nieve y hielo visible en el Nevado Santa Isabel · entrenamiento, evaluación y paquete

**Pregunta.** ¿Cuánta cobertura de nieve y hielo visible hay en una escena Sentinel-2 del Santa Isabel y dónde
cambió entre dos fechas?

**Objetivo del cuaderno.** Afinar un modelo de segmentación de instancias (YOLO26n-seg) con el dataset mensual
del generador (SDD 03), elegir checkpoint, resolución y umbral **solo con validación**, aplicar esa configuración
congelada **una vez** a la prueba temporal y publicar un paquete inmutable que la aplicación Streamlit consume con
las mismas funciones de lectura, inferencia y medición (`src/santa_isabel`).

**Alcance.** La clase reúne nieve e hielo visibles. Las etiquetas son pseudoetiquetas automáticas por reglas NDSI,
sin revisión humana: las métricas miden *concordancia con esas pseudoetiquetas*. Dos fechas no miden balance de
masa ni permiten atribuir una tendencia glaciar; en la interfaz y en el informe se habla de **pérdida observada de
cobertura**.

Kernel: **Python (yolo_gee)**. Cada bloque guarda sus tablas y figuras en `reports/<run_id>/`.
""")

md("## 1 · Configuración, dispositivo y estado de preparación")
code("""
import santa_isabel  # primero: fija PROJ/GDAL y la capa de hilos de MKL antes de importar numpy

import json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from santa_isabel import areas, cargar_config, cuaderno, dataset, flujo, paquete, visual
from santa_isabel import entrenamiento as ent
from santa_isabel.escenas import leer_escena

pd.set_option("display.width", 220, "display.max_columns", 40, "display.float_format", "{:.4f}".format)

VARIANTE = "rgb"                                  # primero RGB; el experimento espectral es posterior
PRUEBA_RAPIDA = os.environ.get("SI_SMOKE") == "1"  # ensayo de integración del cuaderno (no publica en models/)
cfg = cargar_config()                              # resuelve la raíz del proyecto, no el directorio de Jupyter
RAIZ = Path(cfg["_raiz"])
RUN_ID = os.environ.get("SI_RUN_ID") or (("smoke_" if PRUEBA_RAPIDA else "") +
                                         f"{VARIANTE}_{cfg['dataset']['version']}_{datetime.now():%Y%m%d_%H%M}")
CORRIDA = RAIZ / "runs" / RUN_ID
REPORTES = CORRIDA / "reports" if PRUEBA_RAPIDA else RAIZ / "reports" / RUN_ID
MODELOS = CORRIDA / "models" if PRUEBA_RAPIDA else RAIZ / "models"
EJEMPLOS = CORRIDA / "examples" if PRUEBA_RAPIDA else RAIZ / "examples"
for carpeta in (CORRIDA, REPORTES, MODELOS):
    carpeta.mkdir(parents=True, exist_ok=True)

gpu = ent.diagnostico_gpu(exigir=True)   # se detiene con diagnóstico de kernel, wheel y driver si no hay CUDA
print("Intérprete :", sys.executable)
print(f"torch {gpu['torch']} · torchvision {gpu['torchvision']} · ultralytics {gpu['ultralytics']} · CUDA "
      f"{gpu['cuda_compilada']} · cuDNN {gpu['cudnn']}")
print(f"GPU        : {gpu['gpu']} · VRAM {gpu['vram_total_mib']} MiB (libre {gpu['vram_libre_mib']} MiB) · "
      f"operación tensorial en {gpu['operacion_cuda']} sincronizada: {gpu['operacion_ok']}")
e, inf = cfg["entrenamiento"], cfg["inferencia"]
print(f"Dataset    : {cfg['dataset']['version']} · variante {VARIANTE} · {dataset.carpeta_version(cfg)}")
print(f"Caché QA   : {dataset.raiz_cache(cfg) or 'no configurada (métricas sobre toda la imagen)'}")
print(f"Semilla {cfg['proyecto']['semilla']} · device={e['device']} · workers={e['workers']} · lote {e['lote']} · "
      f"imgsz {e['imgsz']} · modelo base {e['modelo_base']}")
print(f"Corrida    : {RUN_ID}\\n  runs    → {CORRIDA}\\n  reports → {REPORTES}\\n  models  → {MODELOS}")
SOPORTE = "observable" if dataset.raiz_cache(cfg) else "imagen"
print(f"Soporte de evaluación principal: {flujo.SOPORTES[SOPORTE]}")
""")

md("""
## 2 · Auditoría y exploración

Antes de entrenar se valida el contrato con el generador: archivos, encabezados, claves, fechas UTC, perfil
L1C/L2A, hashes, clase única, pares imagen–etiqueta en ambas variantes y ausencia de fuga de fechas entre splits.
Un archivo corrupto, un TXT ausente o una fecha duplicada **bloquean** el entrenamiento. Los splits son los del
generador (92/19/21 meses planificados); abajo se ven las escenas útiles reales.
""")
code("""
aud = dataset.auditar_dataset(cfg)
tabla_splits = pd.DataFrame(aud.tabla_splits())
display(tabla_splits)
for aviso in aud.avisos:
    print("AVISO  ·", aviso)
if not aud.ok:
    raise RuntimeError("La auditoría bloquea el entrenamiento:\\n- " + "\\n- ".join(aud.bloqueos))
identidad = aud.identidad()
print(f"\\nAuditoría superada: {len(aud.filas)} escenas, {len(aud.instancias)} instancias, "
      f"{aud.hashes_verificados} hashes verificados contra el manifiesto.")
print("Límites temporales (cortes del generador):", aud.resumen["cortes"]["corte_val"], "→ val ·",
      aud.resumen["cortes"]["corte_test"], "→ test · huella de contenido", identidad["huella_contenido"])
cuaderno.guardar_csv(aud.tabla_splits(), REPORTES / "tabla_splits.csv")
cuaderno.guardar_json({"identidad": identidad, "avisos": aud.avisos, "meses": aud.meses,
                       "equivalencias_manifiesto": dataset.EQUIVALENCIAS}, REPORTES / "auditoria.json");
""")

md("""
**Tamaño de los parches.** La red ve la imagen reescalada a `imgsz` y, al entrenar, compara máscaras reducidas por
`mask_ratio`. Un parche que ocupa menos de una celda de esa máscara apenas aporta señal: la tabla cuenta cuántos
hay para cada resolución candidata (1 px = 10 m × 10 m = 0,01 ha).
""")
code("""
cuaderno.figura_histogramas(aud, REPORTES / "fig_histogramas.png"); plt.show()
tabla_tam = pd.DataFrame(cuaderno.tabla_tamano_minimo(aud, cfg, inf["imgsz_candidatos"], e["mask_ratio"]))
display(tabla_tam)
tabla_tam.to_csv(REPORTES / "tabla_tamano_minimo.csv", index=False, encoding="utf-8-sig")
""")

md("""
## 3 · Referencias

Los polígonos YOLO (bordes de píxel, normalizados) se rasterizan a la malla del PNG con la regla del productor:
un píxel es positivo si su centro cae dentro del polígono. Con la caché del generador se cruza cada escena con su
máscara de calidad (`calidad.tif = 2`: observado y visible) y su soporte. Se prepara también el comparador
**NDSI ≥ umbral** con la receta del productor.

> **Circularidad.** La pseudoetiqueta nace del NDSI. Evaluar el umbral NDSI contra ella es circular y no demuestra
> superioridad de ningún método. Además, la configuración efectiva del productor (`ambiguos_bloquean = false`) deja
> como **fondo** los componentes NDSI ambiguos (p. ej. en contacto con nubes): la columna `ambiguos_ha` muestra
> cuánta superficie queda en esa situación. Por eso se reporta también una evaluación que excluye esos píxeles.
""")
code("""
tabla_ref = cuaderno.tabla_referencias(aud, cfg)
display(pd.DataFrame(tabla_ref).drop(columns=["origen", "revision_humana"]))
cuaderno.guardar_csv(tabla_ref, REPORTES / "tabla_referencias.csv")
ejemplos_etiqueta = cuaderno.elegir_ejemplos(tabla_ref)
cuaderno.figura_ejemplos(aud, cfg, {"positivo": ejemplos_etiqueta["positivo"], "difícil": ejemplos_etiqueta["dificil"]},
                         REPORTES / "fig_ejemplos_etiquetas.png"); plt.show()
print("Ejemplo negativo: no existe en", aud.version, "(las", len(aud.filas), "escenas son positivas).")
print("Delineación humana independiente: ninguna escena del dataset la tiene (revision_humana = False en todas).",
      "La revisión de la escena piloto 2024-01-23 del selector fue asistida por NDSI: sirve para integración, no",
      "como prueba independiente.")
""")

md("""
## 4 · Entrenamiento en la RTX

Se afina `yolo26n-seg.pt` preentrenado con una clase. Cada parche es una instancia; la cobertura es la **unión** de
las máscaras. Los `data.yaml` del generador traen un `path:` absoluto de otro equipo: se generan copias locales en
la carpeta de corrida cambiando solo esa línea. Sin mosaic, mixup ni jitter HSV (alterarían el significado
radiométrico); solo aumentos geométricos, que transforman igual imagen y polígonos.

Tres pasos: (a) un lote real y una época de integración con el pico de memoria; (b) la **configuración inicial
propuesta** (hasta 50 épocas, paciencia 10); (c) la **corrida ajustada** que resulta de medir (b).
""")
code("""
yamls = dataset.yaml_locales(cfg, CORRIDA)
print("YAML local:", yamls[VARIANTE]); print(yamls[VARIANTE].read_text(encoding="utf-8"))
display(pd.DataFrame([ent.comprobar_lote(cfg, yamls[VARIANTE], s) for s in dataset.SPLITS]))
aj = e["ajustado"]
reg_int = ent.entrenar(cfg, yamls[VARIANTE], CORRIDA, "1_integracion", epocas=e["epocas_integracion"], silencioso=True)
reg_ini = ent.entrenar(cfg, yamls[VARIANTE], CORRIDA, "2_inicial", epocas=2 if PRUEBA_RAPIDA else e["epocas"],
                       silencioso=True)
reg_aj = ent.entrenar(cfg, yamls[VARIANTE], CORRIDA, "3_ajustada", epocas=3 if PRUEBA_RAPIDA else aj["epocas"],
                      paciencia=aj["paciencia"], silencioso=True,
                      extra={"nbs": aj["nbs"], "save_period": 1 if PRUEBA_RAPIDA else aj["guardar_cada"]})
registros = pd.DataFrame([{k: r[k] for k in ("nombre", "dispositivo", "gpu", "epocas_pedidas", "epocas_ejecutadas",
                                             "mejor_epoca_fitness", "segundos", "pico_asignado_mib",
                                             "pico_reservado_mib")} | {"nbs": r["argumentos"].get("nbs"),
                                                                       "paciencia": r["argumentos"].get("patience")}
                          for r in (reg_int, reg_ini, reg_aj)])
display(registros)
registros.to_csv(REPORTES / "tabla_entrenamiento.csv", index=False, encoding="utf-8-sig")
cuaderno.guardar_json({"integracion": reg_int, "inicial": reg_ini, "ajustada": reg_aj}, REPORTES / "entrenamiento.json")
cuaderno.figura_curvas({"inicial (nbs 64, paciencia 10)": reg_ini["carpeta"],
                        f"ajustada (nbs {aj['nbs']}, sin parada temprana)": reg_aj["carpeta"]},
                       REPORTES / "fig_curvas_entrenamiento.png"); plt.show()
""")
md("""
**Lectura.** Con lote 2, Ultralytics acumula gradientes hasta un lote nominal `nbs = 64`: 32 lotes por paso del
optimizador, es decir un paso cada ~3,5 épocas con 9 lotes por época. Además, la parada temprana usa el mAP de
instancia de validación (4 escenas, 7 instancias), que es ruido. Por eso la corrida inicial se detiene pronto sin
haber aprendido. La corrida ajustada usa `nbs = lote` (un paso por lote), desactiva la parada temprana y guarda
checkpoints periódicos. `best.pt` responde al *fitness* interno de Ultralytics (mAP de cajas y máscaras en
validación); la selección final de este cuaderno usa el IoU de cobertura, en el bloque siguiente.
""")

md("""
## 5 · Selección con validación

Solo se usa validación. Para cada checkpoint se hace **una inferencia por imagen** con umbral de propuestas bajo
(0,01) y se guarda la confianza y la máscara de cada instancia. La puntuación de un píxel es el **máximo** de las
confianzas de las instancias que lo cubren (0 fuera de propuestas): un ranking con empates, no una probabilidad
calibrada. Se elige el checkpoint con mayor **IoU de cobertura** (desempate: menor MAE de área), después la
resolución de inferencia y por último el **umbral operativo que maximiza F1** (empates → umbral mayor). La
configuración se congela en un protocolo que impide evaluar test antes o más de una vez.
""")
code("""
protocolo = flujo.Protocolo()
pesos = ent.checkpoints_candidatos({"2_inicial": reg_ini["carpeta"], "3_ajustada": reg_aj["carpeta"]},
                                   CORRIDA / "candidatos")
candidatos = []
for c in pesos:
    modelo = paquete.modelo_en_memoria(c["ruta"], cfg, VARIANTE, dispositivo=0)
    el = flujo.inferir_split(cfg, aud, modelo, "val", protocolo=protocolo, proposito="seleccion de checkpoint")
    u = flujo.elegir_umbral(el, SOPORTE)
    candidatos.append({"checkpoint": c["checkpoint"], "ruta": str(c["ruta"]), "imgsz": e["imgsz"],
                       **flujo.resumen_seleccion(el, u["umbral"], SOPORTE),
                       "propuestas": sum(x["instancias_propuestas"] for x in el),
                       "tope_alcanzado": any(x["tope_alcanzado"] for x in el)})
k = flujo.mejor_candidato(candidatos)
tabla_ckpt = pd.DataFrame(candidatos).drop(columns="ruta")
tabla_ckpt["elegido"] = ["◀" if i == k else "" for i in range(len(candidatos))]
display(tabla_ckpt)
tabla_ckpt.to_csv(REPORTES / "seleccion_checkpoints.csv", index=False, encoding="utf-8-sig")
PESOS = Path(candidatos[k]["ruta"])
ious = pd.to_numeric(tabla_ckpt["iou"], errors="coerce").dropna()
IOU_MIN, IOU_MAX = (float(ious.min()), float(ious.max())) if len(ious) else (None, None)
print("Checkpoint elegido por IoU de cobertura en validación:", candidatos[k]["checkpoint"])
print(f"IoU entre los {len(candidatos)} checkpoints: mínimo {IOU_MIN} · máximo {IOU_MAX} "
      f"(con {len(el)} escenas de validación la selección es inestable)")
""")
code("""
resoluciones = []
for imgsz in inf["imgsz_candidatos"]:
    modelo = paquete.modelo_en_memoria(PESOS, cfg, VARIANTE, imgsz=imgsz, dispositivo=0)
    el = flujo.inferir_split(cfg, aud, modelo, "val", protocolo=protocolo, proposito="seleccion de resolucion")
    u = flujo.elegir_umbral(el, SOPORTE)
    resoluciones.append({"imgsz": imgsz, **flujo.resumen_seleccion(el, u["umbral"], SOPORTE),
                         "recall_parches_menores_25px": (lambda r: sum(x["recuperados"] for x in r[:2]) /
                                                         max(sum(x["parches"] for x in r[:2]), 1))(
                             cuaderno.recall_por_tamano(el, u["umbral"]))})
kr = flujo.mejor_candidato(resoluciones)
display(pd.DataFrame(resoluciones))
pd.DataFrame(resoluciones).to_csv(REPORTES / "seleccion_resolucion.csv", index=False, encoding="utf-8-sig")
IMGSZ = resoluciones[kr]["imgsz"]

modelo = paquete.modelo_en_memoria(PESOS, cfg, VARIANTE, imgsz=IMGSZ, dispositivo=0)
el_val = flujo.inferir_split(cfg, aud, modelo, "val", protocolo=protocolo, proposito="seleccion de umbral")
umbral = flujo.elegir_umbral(el_val, SOPORTE)
UMBRAL = umbral["umbral"]
SELECCION = {"split": "val", "soporte": SOPORTE, "checkpoint": candidatos[k]["checkpoint"], "imgsz": int(IMGSZ),
             "umbral": float(UMBRAL), "umbral_f1": umbral.get("umbral_f1"),
             "margen_numerico": umbral.get("margen_numerico"), "f1_val": umbral.get("f1"),
             "empates_f1": umbral.get("empates"),
             "criterio": "checkpoint y resolución por mayor IoU de cobertura en validación (desempate: menor MAE de "
                         "área); umbral operativo por máximo F1 en validación (empates → umbral mayor), menos un "
                         "margen numérico que no cambia la máscara de validación",
             "candidatos_checkpoint": len(candidatos), "iou_val_min": IOU_MIN, "iou_val_max": IOU_MAX}
protocolo.congelar(SELECCION)
modelo.contrato["posproceso"]["umbral_operativo"] = float(UMBRAL)
f1_val = "N/D" if umbral.get("f1") is None else f"{umbral['f1']:.4f}"
print(f"Configuración congelada: {SELECCION['checkpoint']} · imgsz {IMGSZ} · umbral operativo {UMBRAL:.4f} "
      f"(F1 val {f1_val}; {umbral['motivo']})")
""")
md("""
**Validación con la configuración elegida.** Tres soportes: píxel observable (principal), observable sin los
componentes ambiguos, y toda la imagen sin control de observabilidad. La figura muestra aciertos y fallos; la tabla
final, cuántos parches de la referencia se recuperan según su tamaño.
""")
code("""
res_val = {s: flujo.metricas_split(el_val, UMBRAL, s) for s in flujo.SOPORTES
           if all(flujo.soporte_valido(x, s) is not None for x in el_val)}
COLUMNAS = ["precision", "recall", "f1", "iou", "exactitud", "auc", "mae_ha", "sesgo_ha", "tp", "fp", "fn", "tn"]
display(pd.DataFrame({s: {c: r["agregado"][c] for c in COLUMNAS} for s, r in res_val.items()}).T)
cuaderno.figura_predicciones(el_val, UMBRAL, REPORTES / "fig_validacion_predicciones.png",
                             f"Validación · {SELECCION['checkpoint']} · umbral {UMBRAL:.3f}", SOPORTE); plt.show()
tabla_parches_val = pd.DataFrame(cuaderno.recall_por_tamano(el_val, UMBRAL))
display(tabla_parches_val)
cuaderno.guardar_csv(res_val[SOPORTE]["por_escena"], REPORTES / "metricas_val_por_escena.csv");
""")

md("""
## 6 · Prueba temporal (una sola vez)

La configuración congelada se aplica **una vez** a las escenas de test (fechas posteriores a todo el entrenamiento
y la validación, mismo nevado: mide transferencia entre fechas del lugar). Clasificación binaria por píxel:
precisión = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN), IoU = TP/(TP+FP+FN). La **ROC** enfrenta FPR y
TPR al recorrer el umbral sobre la puntuación previa al umbral operativo; el **AUC** es el área bajo esa curva.
""")
code("""
el_test = flujo.inferir_split(cfg, aud, modelo, "test", protocolo=protocolo, proposito="prueba temporal (unica)")
res_test = {s: flujo.metricas_split(el_test, UMBRAL, s) for s in flujo.SOPORTES
            if all(flujo.soporte_valido(x, s) is not None for x in el_test)}
tabla_test = pd.DataFrame({s: {c: r["agregado"][c] for c in COLUMNAS} for s, r in res_test.items()}).T
display(tabla_test)
por_escena = pd.DataFrame(res_test[SOPORTE]["por_escena"])
columnas_escena = ["mes", "perfil", "referencia", "precision", "recall", "f1", "iou", "auc", "area_ref_ha",
                   "area_pred_ha", "error_ha", "instancias_propuestas", "tope_alcanzado", "nd"]
display(por_escena[[c for c in columnas_escena if c in por_escena]])
for s, r in res_test.items():
    cuaderno.guardar_csv(r["por_escena"], REPORTES / f"metricas_test_por_escena_{s}.csv")
tabla_test.to_csv(REPORTES / "metricas_test_agregadas.csv", encoding="utf-8-sig")
print("Referencia:", ", ".join(identidad["origen_referencia"]), "· revisión humana:", identidad["revision_humana"])
print("Protocolo:", json.dumps({k: v for k, v in protocolo.resumen().items() if k != "eventos"}, ensure_ascii=False))
""")
code("""
m_test, m_val = res_test[SOPORTE]["agregado"], res_val[SOPORTE]["agregado"]
curvas = {"prueba temporal": {"roc": res_test[SOPORTE]["roc"], "pr": res_test[SOPORTE]["pr"],
                              "punto": cuaderno.punto_operativo(m_test)},
          "validación": {"roc": res_val[SOPORTE]["roc"], "pr": res_val[SOPORTE]["pr"],
                         "punto": cuaderno.punto_operativo(m_val)}}
cuaderno.figura_roc(curvas, REPORTES / "fig_roc.png", f"ROC por píxel · {SOPORTE} · umbral {UMBRAL:.3f}"); plt.show()
cuaderno.figura_roc({"prueba temporal": curvas["prueba temporal"]}, REPORTES / "roc_test.png", "ROC · prueba temporal")
plt.close()
cuaderno.figura_roc({"validación": curvas["validación"]}, REPORTES / "roc_val.png", "ROC · validación"); plt.close()
cuaderno.figura_pr({"prueba temporal": {"pr": res_test[SOPORTE]["pr"], "punto": (m_test["recall"], m_test["precision"])},
                    "validación": {"pr": res_val[SOPORTE]["pr"], "punto": (m_val["recall"], m_val["precision"])}},
                   REPORTES / "fig_pr.png"); plt.show()
cuaderno.figura_matriz(m_test, REPORTES / "fig_matriz_test.png",
                       f"Prueba temporal · píxeles {SOPORTE} · umbral {UMBRAL:.3f}"); plt.show()
roc = res_test[SOPORTE]["roc"]
if roc["definida"]:
    pd.DataFrame({"fpr": roc["fpr"], "tpr": roc["tpr"], "umbral": roc["umbrales"]}).to_csv(
        REPORTES / "roc_test_puntos.csv", index=False)
print(f"AUC prueba: {roc['auc']} · soportes: {roc['positivos']} px de nieve, {roc['negativos']} px de fondo")
""")
code("""
cuaderno.figura_predicciones(el_test, UMBRAL, REPORTES / "fig_test_predicciones.png",
                             f"Prueba temporal · umbral {UMBRAL:.3f} · FP y FN sobre píxeles {SOPORTE}", SOPORTE)
plt.show()
tabla_parches = pd.DataFrame(cuaderno.recall_por_tamano(el_test, UMBRAL))
display(tabla_parches)
tabla_parches.to_csv(REPORTES / "recall_parches_test.csv", index=False, encoding="utf-8-sig")
ndsi = {s: flujo.comparador_ndsi(el_test, s) for s in res_test if s != "imagen"} if SOPORTE == "observable" else {}
if ndsi:
    comp = pd.DataFrame({f"YOLO · {s}": {c: res_test[s]["agregado"][c] for c in COLUMNAS[:8]} for s in ndsi} |
                        {f"NDSI ≥ {el_test[0]['umbral_ndsi']} · {s} (circular)":
                         {c: r["agregado"][c] for c in COLUMNAS[:8]} for s, r in ndsi.items()}).T
    display(comp)
    comp.to_csv(REPORTES / "comparador_ndsi_test.csv", encoding="utf-8-sig")
map_test = ent.validar_instancias(cfg, PESOS, yamls[VARIANTE], "test", CORRIDA, "val_test", imgsz=IMGSZ)
map_val = ent.validar_instancias(cfg, PESOS, yamls[VARIANTE], "val", CORRIDA, "val_val", imgsz=IMGSZ)
display(pd.DataFrame([map_val, map_test]))
""")

md("""
## 7 · Exportación, paquete inmutable e inferencia de ida y vuelta

Se exporta a ONNX para el servicio en CPU (misma rama con NMS que usa la predicción de referencia), se publica el
paquete con contrato, métricas y hashes, se copian dos escenas reales con su máscara de calidad y se reabre todo con
**el mismo cargador que usa Streamlit**: en este proceso (GPU) y en un proceso limpio en CPU con ambos backends.
""")
code("""
import onnxruntime, torch, ultralytics
onnx_info = ent.exportar_onnx(cfg, PESOS, IMGSZ, CORRIDA / "export" / "best.onnx")
versiones = {"python": gpu["python"], "torch": gpu["torch"], "torchvision": gpu["torchvision"],
             "ultralytics": gpu["ultralytics"], "cuda": gpu["cuda_compilada"], "numpy": np.__version__,
             "onnx": onnx_info["onnx"], "onnxruntime": onnx_info["onnxruntime"], "so": gpu["so"], "gpu": gpu["gpu"]}
ID = f"si-{VARIANTE}-yolo26n-{RUN_ID}"
registro_elegido = reg_aj if "3_ajustada" in SELECCION["checkpoint"] else reg_ini

def agregados(resultados, instancias):
    salida = {s: cuaderno.solo_escalares(r["agregado"]) | {"nd": r["agregado"]["nd"]} for s, r in resultados.items()}
    salida["instancias"] = instancias
    return salida

metricas = {
    "run_id": RUN_ID, "paquete": ID, "dataset": identidad, "seleccion": SELECCION,
    "referencia": {"origen": identidad["origen_referencia"], "revision_humana": identidad["revision_humana"],
                   "soporte_principal": flujo.SOPORTES[SOPORTE]},
    "val": agregados(res_val, map_val), "test": agregados(res_test, map_test),
    "test_por_escena": res_test[SOPORTE]["por_escena"],
    "recall_parches_test": tabla_parches.to_dict("records"),
    "comparador_ndsi_test": {s: cuaderno.solo_escalares(r["agregado"]) for s, r in ndsi.items()},
    "protocolo": protocolo.resumen(),
    "entrenamiento": {"integracion": reg_int, "inicial": reg_ini, "ajustada": reg_aj},
    "bitacora": {"probado con fixture": "pytest (hitos A–D)", "entrenado en RTX": f"sí ({gpu['gpu']})",
                 "evaluado con datos reales": f"sí ({aud.version}, pseudoetiquetas NDSI)",
                 "verificado en Render": "pendiente (sin acceso a una cuenta)"}}
contrato = cuaderno.contrato_paquete(ID, VARIANTE, cfg, aud, SELECCION, registro_elegido, versiones, onnx_info)
PAQUETE = paquete.publicar_paquete(MODELOS, contrato, {
    "pesos/best.pt": PESOS, "pesos/best.onnx": onnx_info["archivo"],
    "metricas.json": json.dumps(cuaderno._limpio(metricas), ensure_ascii=False, indent=1).encode("utf-8"),
    "roc_test.png": REPORTES / "roc_test.png", "roc_val.png": REPORTES / "roc_val.png",
    "metricas_test_por_escena.csv": REPORTES / f"metricas_test_por_escena_{SOPORTE}.csv"}, activar=True)
cuaderno.guardar_json(metricas, REPORTES / "metricas.json")
print("Paquete publicado:", PAQUETE)
print((PAQUETE / "SHA256SUMS").read_text())
""")
md("""
**Dos escenas reales para la demostración.** Mismo mes en años distintos, ninguna de entrenamiento, con su máscara
de calidad. Se copian de la caché del productor (solo lectura) con fuente y crédito; es lo único que la app necesita
empaquetar para Render.
""")
code("""
sys.path.insert(0, str(RAIZ / "tools"))
import preparar_ejemplos
PAR = cuaderno.elegir_par_demo(tabla_ref)
print("Par elegido (mes inicial, mes final):", PAR)
claves = preparar_ejemplos.preparar(PAR, EJEMPLOS) if PAR else []

servicio = paquete.cargar_paquete(PAQUETE, dispositivo=0, backend="ultralytics")   # mismo cargador que la app
analisis = [flujo.analizar_escena(leer_escena(EJEMPLOS / c), servicio, cfg) for c in claves]
cambio = flujo.comparar_fechas(*analisis, cfg) if len(analisis) == 2 else None
filas_demo = [{"escena": a.escena.clave, "fecha": f"{a.escena.fecha:%Y-%m-%d}", **a.medicion.como_fila()}
              for a in analisis]
display(pd.DataFrame(filas_demo)[["fecha", "estado", "area_nieve_ha", "area_observable_ha", "area_aoi_ha",
                                  "pct_nieve_aoi", "pct_observable_aoi", "qa"]])
if cambio is not None:
    display(pd.Series(cambio.como_fila(), name="comparación de dos fechas").to_frame())
    fig, ejes = plt.subplots(1, 3, figsize=(14, 5.4))
    for eje, a in zip(ejes, analisis):
        eje.imshow(visual.superponer(visual.realzar(a.escena.rgb8), a.prediccion.mascara, a.escena.observable))
        eje.set_title(f"{a.escena.fecha:%Y-%m-%d} · {a.medicion.area_nieve_ha:.2f} ha estimadas", fontsize=10)
    ejes[2].imshow(visual.mapa_cambio_rgb(cambio.mapa, visual.realzar(analisis[-1].escena.rgb8)))
    ejes[2].set_title(f"pérdida {cambio.perdida_ha:.2f} ha · ganancia {cambio.ganancia_ha:.2f} ha · neto "
                      f"{cambio.cambio_neto_ha:+.2f} ha", fontsize=10)
    for eje in ejes:
        eje.axis("off")
    fig.suptitle("Demostración de dos fechas · rojo pérdida · verde ganancia · azul persistencia · gris sin "
                 "observación común", fontsize=10)
    fig.tight_layout(); fig.savefig(REPORTES / "fig_demo_dos_fechas.png", dpi=110); plt.show()
    cuaderno.guardar_csv([cambio.como_fila()], REPORTES / "demo_comparacion.csv")
cuaderno.guardar_csv(filas_demo, REPORTES / "demo_escenas.csv");
""")
md("""
**Proceso limpio en CPU.** Un intérprete nuevo reabre el paquete (verifica hashes y contrato), analiza las dos
escenas con ONNX Runtime y con PyTorch en CPU y las compara. Debe dar las mismas áreas que la GPU.
""")
code("""
verificacion = {"backends": {}}
for backend in ("onnx", "ultralytics"):          # un proceso limpio por backend: tiempos y memoria no se mezclan
    salida = subprocess.run([sys.executable, str(RAIZ / "tools" / "verificar_paquete.py"), str(PAQUETE),
                             "--ejemplos", str(EJEMPLOS), "--backends", backend],
                            capture_output=True, text=True, encoding="utf-8")
    if salida.returncode != 0:
        raise RuntimeError(f"La verificación en proceso limpio ({backend}) falló:\\n" + salida.stderr[-3000:])
    datos = json.loads(salida.stdout[salida.stdout.index("{"):])
    verificacion["backends"].update(datos["backends"])
    verificacion.update({k: v for k, v in datos.items() if k != "backends"})
    print(f"{backend}: torch cargado en el proceso = {datos['backends'][backend]['torch_cargado']} · memoria pico "
          f"{datos['backends'][backend]['memoria_pico_mib']} MiB")
filas = [{"backend": "ultralytics · GPU (este cuaderno)", "clave": a.escena.clave,
          "area_nieve_ha": a.medicion.area_nieve_ha, "pixeles_nieve": a.medicion.pixeles_nieve,
          "parches": a.prediccion.instancias_operativas, "segundos_inferencia": round(a.prediccion.segundos, 3)}
         for a in analisis]
for nombre, datos in verificacion["backends"].items():
    filas += [{"backend": f"{nombre} · CPU (proceso limpio)", **{k: x[k] for k in ("clave", "area_nieve_ha",
               "pixeles_nieve", "parches", "segundos_inferencia")}, "memoria_pico_mib": datos["memoria_pico_mib"],
               "carga_modelo_s": datos["segundos_carga_modelo"]} for x in datos["escenas"]]
tabla_backends = pd.DataFrame(filas)
display(tabla_backends)
tabla_backends.to_csv(REPORTES / "verificacion_backends.csv", index=False, encoding="utf-8-sig")
cuaderno.guardar_json(verificacion, REPORTES / "verificacion_cpu.json")
ref = {a.escena.clave: a.medicion.pixeles_nieve for a in analisis}
dif = {f"{n}/{x['clave'][:8]}": x["pixeles_nieve"] - ref[x["clave"]] for n, d in verificacion["backends"].items()
       for x in d["escenas"]}
print("Diferencia en píxeles de nieve frente a la GPU:", dif)
assert all(abs(v) <= max(5, 0.01 * max(ref.values())) for v in dif.values()), "los backends no son equivalentes"
print("\\nCorrida", RUN_ID, "terminada. Paquete:", PAQUETE, "· evidencias:", REPORTES)
""")

nb = nbf.v4.new_notebook(cells=celdas, metadata={
    "kernelspec": {"display_name": "Python (yolo_gee)", "language": "python", "name": "yolo_gee"},
    "language_info": {"name": "python"}})
destino = RAIZ / "notebooks" / "01_entrenar_evaluar.ipynb"
destino.parent.mkdir(exist_ok=True)
nbf.write(nb, destino)
print("Escrito", destino, "con", len(celdas), "celdas")
