"""Apoyo del notebook: tablas, figuras para el informe y contrato del paquete (requiere matplotlib)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from . import dataset, flujo, visual
from .escenas import RECETAS_COMPATIBLES, RECETA_VERSION, TEXTO_RECETA_ESPECTRAL, TEXTO_RECETA_RGB

TRAMOS_PX = [(0, 10, "5–9 px"), (10, 25, "10–24 px"), (25, 100, "25–99 px"), (100, 10**9, "≥ 100 px")]


# --- Persistencia ---------------------------------------------------------------------------------------------------
def _limpio(o):
    if isinstance(o, dict):
        return {str(k): _limpio(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_limpio(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def guardar_json(datos, ruta) -> Path:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(_limpio(datos), ensure_ascii=False, indent=1), encoding="utf-8")
    return ruta


def guardar_csv(filas, ruta) -> Path:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    filas = [_limpio(f) for f in filas]
    campos = list(dict.fromkeys(k for f in filas for k in f))
    with open(ruta, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)
    return ruta


def solo_escalares(d: dict) -> dict:
    return {k: v for k, v in d.items() if isinstance(v, (int, float, str, bool, type(None)))}


# --- Exploración ------------------------------------------------------------------------------------------------------
def tabla_tamano_minimo(auditoria, cfg, imgsz_lista, mask_ratio) -> list:
    """Tamaño de los parches después de escalar la imagen a la entrada de la red y de reducir las máscaras en
    entrenamiento (mask_ratio): cuántas instancias ocupan menos de 1 o de 4 celdas de máscara."""
    ancho, alto = int(cfg["dataset"]["ancho"]), int(cfg["dataset"]["alto"])
    areas_px = np.array([i["area_px"] for i in auditoria.instancias])
    filas = []
    for imgsz in imgsz_lista:
        escala = imgsz / max(ancho, alto)
        celdas = areas_px * escala ** 2 / mask_ratio ** 2
        filas.append({"imgsz": imgsz, "escala": round(escala, 3), "mask_ratio": mask_ratio,
                      "lado_mascara_px": imgsz // mask_ratio,
                      "celda_de_mascara_en_px_originales": round(mask_ratio / escala, 2),
                      "parche_minimo_px": float(areas_px.min()),
                      "parche_minimo_en_celdas": round(float(celdas.min()), 2),
                      "instancias_bajo_1_celda": int((celdas < 1).sum()),
                      "instancias_bajo_4_celdas": int((celdas < 4).sum()), "instancias": len(areas_px)})
    return filas


def tabla_referencias(auditoria, cfg) -> list:
    """Cruce de la referencia rasterizada con QA y soporte de la caché del productor, por escena."""
    ancho, alto = int(cfg["dataset"]["ancho"]), int(cfg["dataset"]["alto"])
    filas = []
    for f in auditoria.filas:
        ref = dataset.mascara_referencia(f, ancho, alto)
        fila = {"split": f["split"], "mes": f["mes"], "fecha_utc": f["adquisicion_utc"], "perfil": f["perfil"],
                "instancias": int(f["instancias"]), "referencia_ha": round(ref.sum() * 0.01, 2),
                "origen": f["origen"], "revision_humana": f["revision_humana"], "cache": False}
        carpeta = dataset.resolver_cache(cfg, f, auditoria.resumen.get("id_aoi"))
        if carpeta is not None:
            capas = dataset.capas_cache(carpeta, f.get("reglas_huella"))
            obs, sop = capas["observable"], capas["soporte"]
            with np.errstate(invalid="ignore"):
                ndsi = sop & (capas["ndsi"] >= (capas.get("umbral_ndsi") or 0.4))
            fila.update(cache=True, observable_pct=round(100.0 * obs.sum() / max(int(sop.sum()), 1), 1),
                        referencia_no_observable_ha=round((ref & ~obs).sum() * 0.01, 2),
                        ambiguos_ha=None if capas["ambiguos"] is None else round(capas["ambiguos"].sum() * 0.01, 2),
                        componentes_ambiguos=(capas["decision"] or {}).get("ambiguos"),
                        ndsi_umbral_ha=round(ndsi.sum() * 0.01, 2),
                        ndsi_umbral_observable_ha=round((ndsi & obs).sum() * 0.01, 2))
        filas.append(fila)
    return filas


def elegir_ejemplos(tabla) -> dict:
    """Escena positiva clara (mayor área de referencia y pocos ambiguos) y escena difícil (más área ambigua).
    Devuelve también por qué no hay ejemplo negativo."""
    con_cache = [t for t in tabla if t.get("cache")]
    base = con_cache or tabla
    positiva = max(base, key=lambda t: t["referencia_ha"] - 2 * (t.get("ambiguos_ha") or 0))
    dificil = max(base, key=lambda t: (t.get("ambiguos_ha") or 0, -t.get("observable_pct", 100)))
    return {"positivo": positiva["mes"], "dificil": dificil["mes"], "negativo": None}


def elegir_par_demo(tabla) -> list:
    """Par de fechas para la demostración: mismo mes del calendario en años distintos, ninguna escena de
    entrenamiento, con el mayor intervalo y, a igualdad, la mayor observabilidad mínima."""
    fuera = [t for t in tabla if t["split"] != "train" and t.get("cache")]
    pares = [(a, b) for a in fuera for b in fuera
             if a["mes"] < b["mes"] and a["mes"][5:] == b["mes"][5:] and a["mes"][:4] != b["mes"][:4]]
    if pares:
        a, b = max(pares, key=lambda p: (int(p[1]["mes"][:4]) - int(p[0]["mes"][:4]),
                                         min(p[0].get("observable_pct", 0), p[1].get("observable_pct", 0))))
        return [a["mes"], b["mes"]]
    return [fuera[0]["mes"], fuera[-1]["mes"]] if len(fuera) >= 2 else []


def figura_curvas(carpetas: dict, ruta=None):
    """Pérdida de segmentación y mAP50 de máscaras por época a partir de results.csv de cada corrida."""
    import matplotlib.pyplot as plt
    fig, ejes = plt.subplots(1, 3, figsize=(13, 3.6))
    for nombre, carpeta in carpetas.items():
        archivo = Path(carpeta) / "results.csv"
        if not archivo.is_file():
            continue
        with open(archivo, newline="", encoding="utf-8") as f:
            filas = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
        x = [int(r["epoch"]) for r in filas]
        for eje, columna in zip(ejes, ("train/seg_loss", "val/seg_loss", "metrics/mAP50(M)")):
            eje.plot(x, [float(r[columna]) for r in filas], label=nombre, lw=1.4)
    for eje, titulo in zip(ejes, ("Pérdida de segmentación · train", "Pérdida de segmentación · val",
                                  "mAP50 de máscaras · val (7 instancias)")):
        eje.set(title=titulo, xlabel="época")
        eje.grid(alpha=0.3)
    ejes[1].set_yscale("log")
    ejes[0].legend(fontsize=8)
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=150)
    return fig


def figura_histogramas(auditoria, ruta=None):
    import matplotlib.pyplot as plt
    fig, ejes = plt.subplots(1, 2, figsize=(11, 3.6))
    colores = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}
    bordes = np.logspace(np.log10(4), np.log10(4000), 28)
    for s in dataset.SPLITS:
        a = [i["area_px"] for i in auditoria.instancias if i["split"] == s]
        lado = [max(i["ancho_px"], i["alto_px"]) for i in auditoria.instancias if i["split"] == s]
        ejes[0].hist(a, bins=bordes, alpha=0.6, label=f"{s} ({len(a)})", color=colores[s])
        ejes[1].hist(lado, bins=np.logspace(np.log10(2), np.log10(200), 24), alpha=0.6, label=s, color=colores[s])
    ejes[0].set(xscale="log", xlabel="Área del parche (px de 10 m; 1 px = 0,01 ha)", ylabel="Instancias",
                title="Área de los parches etiquetados")
    ejes[1].set(xscale="log", xlabel="Lado mayor de la caja (px)", ylabel="Instancias",
                title="Tamaño de los parches")
    ejes[0].legend()
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=150)
    return fig


def figura_ejemplos(auditoria, cfg, meses: dict, ruta=None):
    """Imagen y etiqueta: polígonos de la referencia (magenta), componentes ambiguos (rojo) y no observable (gris)."""
    import matplotlib.pyplot as plt
    from PIL import Image
    ancho, alto = int(cfg["dataset"]["ancho"]), int(cfg["dataset"]["alto"])
    roles = [(r, m) for r, m in meses.items() if m]
    fig, ejes = plt.subplots(1, len(roles) * 2, figsize=(5.2 * len(roles) * 2 / 1.6, 5.4))
    for k, (rol, mes) in enumerate(roles):
        f = next(x for x in auditoria.filas if x["mes"] == mes)
        rgb = np.asarray(Image.open(f["_imagen_rgb"]).convert("RGB"))
        ref = dataset.mascara_referencia(f, ancho, alto)
        carpeta = dataset.resolver_cache(cfg, f, auditoria.resumen.get("id_aoi"))
        capas = dataset.capas_cache(carpeta, f.get("reglas_huella")) if carpeta else None
        vista = visual.superponer(visual.realzar(rgb), None, None if capas is None else capas["observable"],
                                  referencia=ref)
        if capas is not None and capas["ambiguos"] is not None:
            vista[visual.contorno(capas["ambiguos"])] = (255, 40, 40)
        ejes[2 * k].imshow(visual.realzar(rgb))
        ejes[2 * k].set_title(f"{rol}: {f['adquisicion_utc'][:10]} ({f['perfil']}, {f['split']})", fontsize=9)
        ejes[2 * k + 1].imshow(vista)
        ejes[2 * k + 1].set_title(f"{f['instancias']} polígono(s) · {ref.sum() * 0.01:.1f} ha etiquetadas", fontsize=9)
    for e in ejes:
        e.axis("off")
    fig.suptitle("Imagen RGB (realzada para ver) y etiqueta: referencia en magenta, componentes NDSI ambiguos "
                 "omitidos en rojo, no observable en gris", fontsize=9)
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=150)
    return fig


# --- Resultados ---------------------------------------------------------------------------------------------------------
def punto_operativo(metricas_agregadas) -> tuple:
    m = metricas_agregadas
    fpr = m["fp"] / (m["fp"] + m["tn"]) if (m["fp"] + m["tn"]) else None
    return fpr, m["recall"]


def figura_roc(curvas: dict, ruta=None, titulo="Curva ROC por píxel"):
    """curvas: {etiqueta: {"roc": resultado de curva_roc, "punto": (fpr, tpr) o None}}."""
    import matplotlib.pyplot as plt
    fig, eje = plt.subplots(figsize=(5.2, 4.8))
    for etiqueta, c in curvas.items():
        roc = c["roc"]
        if not roc["definida"]:
            eje.plot([], [], " ", label=f"{etiqueta}: ROC/AUC no definida")
            continue
        linea, = eje.plot(roc["fpr"], roc["tpr"], lw=1.8, marker=".", ms=3,
                          label=f"{etiqueta} (AUC = {roc['auc']:.3f})")
        if c.get("punto") and c["punto"][0] is not None and c["punto"][1] is not None:
            eje.plot(*c["punto"], "o", ms=9, mfc="none", mew=2, color=linea.get_color())
    eje.plot([0, 1], [0, 1], "k--", lw=0.8, label="azar")
    eje.set(xlabel="FPR = FP / (FP + TN)", ylabel="TPR (recall) = TP / (TP + FN)", title=titulo, xlim=(-0.01, 1.01),
            ylim=(-0.01, 1.01))
    eje.legend(fontsize=8, loc="lower right", title="○ = punto operativo", title_fontsize=8)
    eje.grid(alpha=0.3)
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=150)
    return fig


def figura_pr(curvas: dict, ruta=None, titulo="Curva precisión–recall por píxel"):
    import matplotlib.pyplot as plt
    fig, eje = plt.subplots(figsize=(5.2, 4.8))
    for etiqueta, c in curvas.items():
        pr = c["pr"]
        if not pr.get("definida"):
            continue
        linea, = eje.plot(pr["recall"], pr["precision"], lw=1.8, marker=".", ms=3, label=etiqueta)
        eje.axhline(pr["prevalencia"], color=linea.get_color(), ls=":", lw=0.8)
        if c.get("punto") and None not in c["punto"]:
            eje.plot(*c["punto"], "o", ms=9, mfc="none", mew=2, color=linea.get_color())
    eje.set(xlabel="Recall", ylabel="Precisión = TP / (TP + FP)", title=titulo, xlim=(-0.01, 1.01), ylim=(-0.01, 1.01))
    eje.legend(fontsize=8, title="○ = punto operativo · línea punteada = prevalencia", title_fontsize=7)
    eje.grid(alpha=0.3)
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=150)
    return fig


def figura_matriz(m, ruta=None, titulo="Matriz de confusión por píxel"):
    import matplotlib.pyplot as plt
    datos = np.array([[m["tp"], m["fn"]], [m["fp"], m["tn"]]], dtype=float)
    fig, eje = plt.subplots(figsize=(4.6, 3.9))
    eje.imshow(np.log10(datos + 1), cmap="Blues")
    nombres = [["TP", "FN"], ["FP", "TN"]]
    for i in range(2):
        for j in range(2):
            eje.text(j, i, f"{nombres[i][j]}\n{int(datos[i, j]):,}".replace(",", "\u202f"), ha="center", va="center",
                     fontsize=11, color="white" if np.log10(datos[i, j] + 1) > np.log10(datos.max() + 1) * 0.6
                     else "black")
    eje.set_xticks([0, 1], ["Predicho nieve", "Predicho fondo"])
    eje.set_yticks([0, 1], ["Ref. nieve", "Ref. fondo"])
    eje.set_title(titulo, fontsize=10)
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=150)
    return fig


def mapa_errores(elemento, umbral, soporte="observable") -> np.ndarray:
    """TP verde, FP rojo, FN azul, no válido gris, sobre la imagen atenuada."""
    rgb = visual.realzar(elemento["escena"].rgb8).astype(np.float32)
    base = (0.5 * rgb.mean(axis=2, keepdims=True) + 60).repeat(3, axis=2)
    ref, pred = elemento["referencia"], elemento["puntuacion"] >= np.float32(umbral)
    valido = flujo.soporte_valido(elemento, soporte)
    valido = np.ones(ref.shape, bool) if valido is None else valido
    base[~valido] = (120, 120, 120)
    base[ref & pred & valido] = (40, 200, 60)
    base[~ref & pred & valido] = (230, 40, 40)
    base[ref & ~pred & valido] = (40, 90, 255)
    if elemento.get("ambiguos") is not None:
        base[visual.contorno(elemento["ambiguos"])] = (255, 170, 0)
    return np.clip(base, 0, 255).astype(np.uint8)


def figura_predicciones(elementos, umbral, ruta=None, titulo="", soporte="observable"):
    """Por escena: original, predicción superpuesta con la referencia y mapa de aciertos y errores."""
    import matplotlib.pyplot as plt
    n = len(elementos)
    fig, ejes = plt.subplots(n, 3, figsize=(10.5, 3.9 * n), squeeze=False)
    for k, e in enumerate(elementos):
        rgb = visual.realzar(e["escena"].rgb8)
        pred = e["puntuacion"] >= np.float32(umbral)
        ejes[k, 0].imshow(rgb)
        ejes[k, 0].set_title(f"{e['fila']['adquisicion_utc'][:10]} · {e['perfil']}", fontsize=9)
        ejes[k, 1].imshow(visual.superponer(rgb, pred, e["observable"], 0.45, referencia=e["referencia"]))
        ejes[k, 1].set_title(f"predicción {pred.sum() * 0.01:.2f} ha (cian) · referencia "
                             f"{e['referencia'].sum() * 0.01:.2f} ha (magenta)", fontsize=8)
        ejes[k, 2].imshow(mapa_errores(e, umbral, soporte))
        ejes[k, 2].set_title("TP verde · FP rojo · FN azul · ambiguo naranja · no válido gris", fontsize=8)
        for eje in ejes[k]:
            eje.axis("off")
    if titulo:
        fig.suptitle(titulo, fontsize=10)
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=95)
    return fig


def recall_por_tamano(elementos, umbral, cobertura_minima=0.5) -> list:
    """Parches de la referencia recuperados (≥ 50 % de sus píxeles predichos), por tramo de tamaño."""
    from .evaluacion import rasterizar_poligonos
    conteo = {t[2]: [0, 0] for t in TRAMOS_PX}
    for e in elementos:
        alto, ancho = e["referencia"].shape
        pred = e["puntuacion"] >= np.float32(umbral)
        for v in e["fila"]["_poligonos"]:
            m = rasterizar_poligonos([v], ancho, alto)
            n = int(m.sum())
            if n == 0:
                continue
            tramo = next(t[2] for t in TRAMOS_PX if t[0] <= n < t[1])
            conteo[tramo][0] += 1
            conteo[tramo][1] += int((pred & m).sum() / n >= cobertura_minima)
    return [{"tramo": t, "parches": n, "recuperados": r, "recall_de_parches": (r / n if n else None)}
            for t, (n, r) in conteo.items()]


# --- Paquete ------------------------------------------------------------------------------------------------------------
def contrato_paquete(identificador, variante, cfg, auditoria, seleccion, registro, versiones, onnx=None) -> dict:
    """Contrato de modelo.json a partir de la configuración congelada con validación."""
    inf = cfg["inferencia"]
    ident = auditoria.identidad()
    backends = {"ultralytics": {"archivo": "pesos/best.pt", "ultralytics": versiones.get("ultralytics"),
                                "torch": versiones.get("torch")}}
    if onnx:
        backends["onnx"] = {"archivo": "pesos/best.onnx", "opset": onnx.get("opset"), "onnx": onnx.get("onnx"),
                            "onnxruntime": onnx.get("onnxruntime"), "imgsz_fijo": onnx.get("imgsz")}
    return {
        "id": identificador, "variante": variante,
        "receta": {"version": RECETA_VERSION, "compatibles": list(RECETAS_COMPATIBLES),
                   "codificacion": TEXTO_RECETA_RGB if variante == "rgb" else TEXTO_RECETA_ESPECTRAL,
                   "huellas_productor": sorted({f.get("receta_huella", "") for f in auditoria.filas})},
        "entrada": {"imgsz": int(seleccion["imgsz"]),
                    "alineacion": "letterbox centrado a imgsz×imgsz con relleno 114, sin rectángulo mínimo "
                                  "(rect=false); se deshace antes de medir",
                    "canales": "RGB uint8 de la receta; Ultralytics recibe el arreglo en BGR; ONNX recibe RGB "
                               "CHW/255",
                    "tesela_px": int(inf["tesela_px"]), "solape_px": int(inf["solape_px"])},
        "posproceso": {"umbral_propuestas": float(inf["umbral_propuestas"]),
                       "umbral_operativo": float(seleccion["umbral"]), "iou_nms": float(inf["iou_nms"]),
                       "max_det": int(inf["max_det"]), "mascaras": "retina_masks (resolución original)",
                       "puntuacion": "máximo de las confianzas de las instancias cuya máscara cubre el píxel; 0 "
                                     "fuera de propuestas (ranking con empates, no probabilidad calibrada)",
                       "mascara_final": "unión de instancias con confianza ≥ umbral operativo"},
        "backends": backends, "backend_referencia": "ultralytics",
        "backend_servicio": "onnx" if onnx else "ultralytics",
        "dataset": {**ident, "variante": variante, "seleccion": "val", "prueba": "test"},
        "referencia": {"origen": ", ".join(ident["origen_referencia"]), "revision_humana": ident["revision_humana"],
                       "aviso": "Referencia automática (pseudoetiquetas por reglas NDSI del generador mensual, sin "
                                "revisión humana). Las métricas miden concordancia con esas pseudoetiquetas, no "
                                "verificación glaciológica independiente."},
        "seleccion": seleccion,
        "entrenamiento": {k: registro.get(k) for k in ("nombre", "modelo_base", "segundos", "dispositivo", "gpu",
                                                      "pico_asignado_mib", "pico_reservado_mib", "epocas_pedidas",
                                                      "epocas_ejecutadas", "criterio_best")},
        "versiones": versiones, "metricas": "metricas.json"}
