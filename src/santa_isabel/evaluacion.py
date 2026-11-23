"""Evaluación binaria por píxel observable, ROC/AUC por conteos y errores de superficie (SDD 04 §6).

Positivo = nieve/hielo visible. La puntuación por píxel es el máximo de las confianzas de las instancias cuya
máscara lo incluye (0 fuera de propuestas): es una puntuación de ranking con empates, no una probabilidad
calibrada. Para controlar la RAM se acumulan conteos por valor de puntuación y clase.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

ND = None  # valor «no definido»


# --- Etiquetas YOLO ------------------------------------------------------------------------------------------------
def leer_txt_yolo(ruta, clases_validas=(0,)):
    """[(clase, vértices normalizados Nx2)]. Un TXT vacío es un negativo explícito; un TXT ausente es un error
    de integridad."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"Falta la etiqueta {ruta} (un negativo debe ser un TXT vacío explícito).")
    instancias = []
    for n, linea in enumerate(ruta.read_text(encoding="utf-8").splitlines(), 1):
        campos = linea.split()
        if not campos:
            continue
        if not re.fullmatch(r"\d+", campos[0]):
            raise ValueError(f"{ruta.name}, línea {n}: clase no entera ({campos[0]!r}).")
        clase = int(campos[0])
        if clases_validas is not None and clase not in clases_validas:
            raise ValueError(f"{ruta.name}, línea {n}: clase {clase} fuera de {tuple(clases_validas)}.")
        try:
            valores = np.array([float(c) for c in campos[1:]], dtype=np.float64)
        except ValueError as e:
            raise ValueError(f"{ruta.name}, línea {n}: coordenada no numérica.") from e
        if len(valores) < 6 or len(valores) % 2:
            raise ValueError(f"{ruta.name}, línea {n}: un polígono necesita al menos 3 pares x y "
                             f"({len(valores)} valores).")
        if not np.isfinite(valores).all() or (valores < 0).any() or (valores > 1).any():
            raise ValueError(f"{ruta.name}, línea {n}: coordenadas fuera de [0, 1].")
        instancias.append((clase, valores.reshape(-1, 2)))
    return instancias


def rasterizar_poligonos(poligonos, ancho, alto, normalizados=True) -> np.ndarray:
    """Unión de polígonos sobre la malla del PNG. Regla del productor: un píxel es positivo si su centro
    (col + 0,5; fila + 0,5) cae dentro del polígono (coordenadas en bordes de píxel)."""
    from rasterio.features import geometry_mask
    from rasterio.transform import Affine
    geoms = []
    for v in poligonos:
        v = np.asarray(v, dtype=np.float64)
        if normalizados:
            v = v * [ancho, alto]
        anillo = v.tolist()
        geoms.append({"type": "Polygon", "coordinates": [anillo + [anillo[0]]]})
    if not geoms:
        return np.zeros((alto, ancho), bool)
    return geometry_mask(geoms, out_shape=(alto, ancho), transform=Affine.identity(), invert=True)


def area_poligono_px(vertices, ancho, alto) -> float:
    v = np.asarray(vertices, dtype=np.float64) * [ancho, alto]
    x, y = v[:, 0], v[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


# --- Matriz de confusión y métricas ----------------------------------------------------------------------------------
def confusion(referencia, prediccion, valido=None) -> dict:
    r, p = np.asarray(referencia, bool), np.asarray(prediccion, bool)
    if r.shape != p.shape:
        raise ValueError(f"Referencia {r.shape} y predicción {p.shape} no están en la misma cuadrícula.")
    v = np.ones(r.shape, bool) if valido is None else np.asarray(valido, bool)
    return {"tp": int((r & p & v).sum()), "fp": int((~r & p & v).sum()), "fn": int((r & ~p & v).sum()),
            "tn": int((~r & ~p & v).sum()), "validos": int(v.sum()), "ignorados": int((~v).sum())}


def _cociente(num, den, motivo):
    return (num / den, None) if den else (ND, motivo)


def metricas_binarias(tp, fp, fn, tn) -> dict:
    """Precisión, recall, F1, IoU y exactitud global. Un denominador cero da N/D con su motivo."""
    m, motivos = {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}, {}
    definiciones = {
        "precision": (tp, tp + fp, "sin píxeles predichos como nieve (TP + FP = 0)"),
        "recall": (tp, tp + fn, "sin píxeles de nieve en la referencia (TP + FN = 0)"),
        "f1": (2 * tp, 2 * tp + fp + fn, "sin nieve en referencia ni en predicción (2TP + FP + FN = 0)"),
        "iou": (tp, tp + fp + fn, "sin nieve en referencia ni en predicción (TP + FP + FN = 0)"),
        "especificidad": (tn, tn + fp, "sin píxeles de fondo en la referencia (TN + FP = 0)"),
        "exactitud": (tp + tn, tp + fp + fn + tn, "sin píxeles válidos"),
    }
    for nombre, (num, den, motivo) in definiciones.items():
        m[nombre], por_que = _cociente(num, den, motivo)
        if por_que:
            motivos[nombre] = por_que
    m["soporte_positivo"], m["soporte_negativo"] = int(tp + fn), int(fp + tn)
    m["nd"] = motivos
    return m


# --- ROC por conteos --------------------------------------------------------------------------------------------------
class AcumuladorROC:
    """Acumula, por valor de puntuación, cuántos píxeles válidos son positivos y cuántos negativos."""

    def __init__(self):
        self._partes = []

    def agregar(self, referencia, puntuacion, valido=None):
        r = np.asarray(referencia, bool)
        s = np.asarray(puntuacion, dtype=np.float32)
        if r.shape != s.shape:
            raise ValueError(f"Referencia {r.shape} y puntuación {s.shape} no están en la misma cuadrícula.")
        if valido is not None:
            v = np.asarray(valido, bool)
            r, s = r[v], s[v]
        r, s = r.ravel(), s.ravel()
        if not np.isfinite(s).all():
            raise ValueError("Hay puntuaciones no finitas.")
        u, inv = np.unique(s, return_inverse=True)
        pos = np.bincount(inv, weights=r, minlength=len(u)).astype(np.int64)
        tot = np.bincount(inv, minlength=len(u)).astype(np.int64)
        self._partes.append((u, pos, tot - pos))
        return self

    def tabla(self):
        """(puntuaciones únicas ascendentes, positivos, negativos)."""
        if not self._partes:
            vacio = np.zeros(0, np.int64)
            return np.zeros(0, np.float32), vacio, vacio
        if len(self._partes) > 1:
            s = np.concatenate([p[0] for p in self._partes])
            u, inv = np.unique(s, return_inverse=True)
            pos = np.bincount(inv, weights=np.concatenate([p[1] for p in self._partes]), minlength=len(u))
            neg = np.bincount(inv, weights=np.concatenate([p[2] for p in self._partes]), minlength=len(u))
            self._partes = [(u, pos.astype(np.int64), neg.astype(np.int64))]
        return self._partes[0]

    @property
    def soportes(self):
        _, pos, neg = self.tabla()
        return int(pos.sum()), int(neg.sum())

    def curva_roc(self) -> dict:
        """FPR, TPR y umbrales (descendentes; el primero es +inf). AUC por trapecios.

        Si falta una clase, la ROC/AUC no está definida y se devuelve el motivo. Con puntuaciones constantes
        y ambas clases resulta la curva degenerada (0,0)→(1,1) con AUC 0,5.
        """
        s, pos, neg = self.tabla()
        p, n = int(pos.sum()), int(neg.sum())
        salida = {"positivos": p, "negativos": n, "definida": bool(p and n), "motivo": None, "auc": ND,
                  "fpr": None, "tpr": None, "umbrales": None}
        if not (p and n):
            falta = "positivos (nieve)" if not p else "negativos (fondo)"
            if not p and not n:
                falta = "píxeles válidos"
            salida["motivo"] = f"ROC/AUC no definida: no hay {falta} en el soporte evaluado."
            return salida
        orden = np.argsort(-s, kind="stable")
        tps = np.concatenate([[0], np.cumsum(pos[orden])]).astype(np.float64)
        fps = np.concatenate([[0], np.cumsum(neg[orden])]).astype(np.float64)
        tpr, fpr = tps / p, fps / n
        salida.update(fpr=fpr, tpr=tpr, umbrales=np.concatenate([[np.inf], s[orden].astype(np.float64)]),
                      auc=float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0)))
        return salida

    def precision_recall(self) -> dict:
        s, pos, neg = self.tabla()
        p = int(pos.sum())
        if not p or len(s) == 0:
            return {"definida": False, "motivo": "Curva precisión–recall no definida: no hay positivos."}
        orden = np.argsort(-s, kind="stable")
        tps, fps = np.cumsum(pos[orden]).astype(np.float64), np.cumsum(neg[orden]).astype(np.float64)
        return {"definida": True, "precision": tps / np.maximum(tps + fps, 1), "recall": tps / p,
                "umbrales": s[orden].astype(np.float64), "prevalencia": p / (p + int(neg.sum()))}

    def confusion_en(self, umbral) -> dict:
        """Matriz de confusión con la regla predicción = puntuación ≥ umbral."""
        s, pos, neg = self.tabla()
        sel = s >= np.float32(umbral)
        tp, fp = int(pos[sel].sum()), int(neg[sel].sum())
        return {"tp": tp, "fp": fp, "fn": int(pos.sum()) - tp, "tn": int(neg.sum()) - fp}

    def mejor_umbral_f1(self, minimo=0.0) -> dict | None:
        """Umbral > `minimo` que maximiza F1; los empates conservan el umbral mayor. None si no hay candidato."""
        s, pos, neg = self.tabla()
        cand = s > np.float32(minimo)
        if not cand.any() or not int(pos.sum()):
            return None
        orden = np.argsort(-s, kind="stable")
        s_d, tps, fps = s[orden], np.cumsum(pos[orden]), np.cumsum(neg[orden])
        fns = int(pos.sum()) - tps
        den = 2 * tps + fps + fns
        f1 = np.where(den > 0, 2 * tps / np.maximum(den, 1), -1.0)
        f1[~(s_d > np.float32(minimo))] = -1.0
        mejor = float(f1.max())
        if mejor < 0:
            return None
        k = int(np.flatnonzero(f1 == mejor)[0])  # orden descendente: el primero es el umbral mayor
        return {"umbral": float(s_d[k]), "f1": mejor, "tp": int(tps[k]), "fp": int(fps[k]), "fn": int(fns[k]),
                "tn": int(neg.sum()) - int(fps[k]), "empates": int((f1 == mejor).sum())}


def curva_roc(etiquetas, puntuaciones, valido=None) -> dict:
    return AcumuladorROC().agregar(np.asarray(etiquetas), np.asarray(puntuaciones), valido).curva_roc()


# --- Superficie ---------------------------------------------------------------------------------------------------------
def error_area(area_ref_ha, area_pred_ha) -> dict:
    """Error firmado (predicción − referencia) y absoluto, en hectáreas."""
    if area_ref_ha is None or area_pred_ha is None:
        return {"error_ha": ND, "error_abs_ha": ND, "error_pct": ND}
    e = float(area_pred_ha) - float(area_ref_ha)
    return {"error_ha": e, "error_abs_ha": abs(e), "error_pct": 100.0 * e / area_ref_ha if area_ref_ha else ND}


def resumen_areas(errores) -> dict:
    """Sesgo medio firmado y MAE (ha) sobre las escenas con error definido."""
    e = [x["error_ha"] for x in errores if x.get("error_ha") is not None]
    if not e:
        return {"escenas": 0, "sesgo_ha": ND, "mae_ha": ND}
    return {"escenas": len(e), "sesgo_ha": float(np.mean(e)), "mae_ha": float(np.mean(np.abs(e)))}


def evaluar_escena(referencia, puntuacion, umbral, valido=None, ha_por_celda=None) -> dict:
    """Métricas de una escena: matriz, precisión/recall/F1/IoU, ROC/AUC (si está definida) y error de área."""
    r = np.asarray(referencia, bool)
    s = np.asarray(puntuacion, np.float32)
    pred = s >= np.float32(umbral)
    c = confusion(r, pred, valido)
    m = metricas_binarias(c["tp"], c["fp"], c["fn"], c["tn"])
    roc = curva_roc(r, s, valido)
    m.update(validos=c["validos"], ignorados=c["ignorados"], auc=roc["auc"], roc_motivo=roc["motivo"],
             umbral=float(umbral))
    if ha_por_celda is not None:
        ref_ha, pred_ha = (c["tp"] + c["fn"]) * ha_por_celda, (c["tp"] + c["fp"]) * ha_por_celda
        m.update(area_ref_ha=ref_ha, area_pred_ha=pred_ha, **error_area(ref_ha, pred_ha))
    return m
