"""Inferencia compartida por el notebook y la app: misma preparación, mismo posprocesado, mismo umbral.

Una sola inferencia por imagen con umbral de propuestas bajo y positivo; se guardan confianza y máscara de
cada instancia. Puntuación por píxel = máximo de las confianzas de las instancias cuya máscara lo incluye
(0 fuera de propuestas). Es una puntuación de ranking con empates, no una probabilidad calibrada. La máscara
final es la unión de instancias con confianza ≥ umbral operativo, es decir puntuación ≥ umbral operativo.
Las máscaras se devuelven siempre con el tamaño de la imagen original (letterbox y reescalado deshechos).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .escenas import Escena, ErrorEscena, cargar_reflectancia, codificar_espectral

RELLENO = 114


@dataclass
class Prediccion:
    puntuacion: np.ndarray            # float32 (alto, ancho): ranking por píxel antes del umbral operativo
    mascara: np.ndarray               # bool (alto, ancho): puntuación ≥ umbral operativo
    alto: int
    ancho: int
    umbral_propuestas: float
    umbral_operativo: float
    instancias: list = field(default_factory=list)   # {"confianza", "area_px", "caja", "tesela"}
    max_det: int = 300
    tope_alcanzado: bool = False
    teselas: int = 1
    imgsz: int = 0
    backend: str = ""
    dispositivo: str = ""
    segundos: float = 0.0

    @property
    def instancias_operativas(self) -> int:
        return sum(i["confianza"] >= self.umbral_operativo for i in self.instancias)

    def con_umbral(self, umbral) -> "Prediccion":
        """Misma inferencia con otro umbral operativo (≥ umbral de propuestas)."""
        umbral = validar_umbral(umbral, self.umbral_propuestas)
        return Prediccion(self.puntuacion, self.puntuacion >= np.float32(umbral), self.alto, self.ancho,
                          self.umbral_propuestas, umbral, self.instancias, self.max_det, self.tope_alcanzado,
                          self.teselas, self.imgsz, self.backend, self.dispositivo, self.segundos)


def validar_umbral(umbral, umbral_propuestas) -> float:
    umbral = float(umbral)
    if not (0.0 < umbral <= 1.0):
        raise ValueError(f"El umbral operativo debe estar en (0, 1]; se recibió {umbral}.")
    if umbral < float(umbral_propuestas):
        raise ValueError(f"El umbral operativo ({umbral}) no puede ser menor que el umbral de propuestas "
                         f"({umbral_propuestas}): esas instancias no se conservaron en la inferencia.")
    return umbral


# --- Preparación --------------------------------------------------------------------------------------------------
def preparar_entrada(escena: Escena, contrato: dict) -> np.ndarray:
    """Imagen uint8 (alto, ancho, 3) en el orden de canales de la variante del contrato, con la receta
    radiométrica del entrenamiento (RGB: B4/B3/B2; espectral: NDSI/NDVI/B4 desde reflectancias)."""
    variante = contrato["variante"]
    if variante == "rgb":
        if escena.rgb8 is None:
            raise ErrorEscena("La escena no tiene imagen RGB codificada (B4/B3/B2).")
        imagen = escena.rgb8
    elif variante == "espectral":
        try:
            refl = escena.reflectancia if escena.reflectancia is not None else cargar_reflectancia(escena)
        except ErrorEscena as e:
            raise ErrorEscena("El paquete activo es espectral (NDSI/NDVI/B4): cargue un GeoTIFF multibanda con "
                              "B3, B4, B8 y B11. Los índices no se derivan de un RGB.", "mapeo_bandas") from e
        imagen = codificar_espectral(refl)
    else:
        raise ErrorEscena(f"Variante de modelo desconocida: {variante!r}.")
    if imagen.dtype != np.uint8 or imagen.ndim != 3 or imagen.shape[2] != 3:
        raise ErrorEscena(f"La entrada debe ser uint8 de 3 canales; se obtuvo {imagen.dtype} {imagen.shape}.")
    return np.ascontiguousarray(imagen)


def geometria_letterbox(alto, ancho, tam):
    """Ganancia, tamaño sin relleno y relleno superior/izquierdo de LetterBox(tam, auto=False, center=True)."""
    r = min(tam / alto, tam / ancho)
    nuevo_w, nuevo_h = max(round(ancho * r), 1), max(round(alto * r), 1)
    dw, dh = (tam - nuevo_w) / 2, (tam - nuevo_h) / 2
    return {"r": r, "nuevo_w": nuevo_w, "nuevo_h": nuevo_h, "arriba": round(dh - 0.1), "abajo": round(dh + 0.1),
            "izquierda": round(dw - 0.1), "derecha": round(dw + 0.1)}


def letterbox(imagen, tam):
    """Reescala conservando la proporción y rellena con 114 hasta tam×tam (igual que Ultralytics)."""
    import cv2
    alto, ancho = imagen.shape[:2]
    g = geometria_letterbox(alto, ancho, tam)
    if (ancho, alto) != (g["nuevo_w"], g["nuevo_h"]):
        imagen = cv2.resize(imagen, (g["nuevo_w"], g["nuevo_h"]), interpolation=cv2.INTER_LINEAR)
    lienzo = cv2.copyMakeBorder(imagen, g["arriba"], g["abajo"], g["izquierda"], g["derecha"],
                                cv2.BORDER_CONSTANT, value=(RELLENO, RELLENO, RELLENO))
    return lienzo, g


def deshacer_letterbox(logits, alto, ancho, tam) -> np.ndarray:
    """Lleva un mapa de la red (resolución de prototipos, tam/4) al tamaño original: amplía a la entrada de la
    red, recorta el relleno en píxeles enteros y reescala (bilineal) a alto×ancho."""
    import cv2
    mh, mw = logits.shape
    gain = min(4 * mh / alto, 4 * mw / ancho)
    pad = (4 * mh - max(round(alto * gain), 1)) | (4 * mw - max(round(ancho * gain), 1))
    f = 4 // math.gcd(4, pad | pad // 2)
    if f > 1:
        logits = cv2.resize(logits, (mw * f, mh * f), interpolation=cv2.INTER_LINEAR)
    h1, w1 = logits.shape
    g = min(h1 / alto, w1 / ancho)
    ch, cw = max(round(alto * g), 1), max(round(ancho * g), 1)
    arriba, izquierda = round((h1 - ch) / 2 - 0.1), round((w1 - cw) / 2 - 0.1)
    recorte = logits[arriba:arriba + ch, izquierda:izquierda + cw]
    if recorte.shape != (alto, ancho):
        recorte = cv2.resize(np.ascontiguousarray(recorte), (ancho, alto), interpolation=cv2.INTER_LINEAR)
    return recorte


def cajas_a_original(cajas, alto, ancho, tam) -> np.ndarray:
    """Cajas xyxy de la entrada de la red a coordenadas de la imagen original, recortadas a sus límites."""
    g = geometria_letterbox(alto, ancho, tam)
    gx, gy = g["nuevo_w"] / ancho, g["nuevo_h"] / alto
    c = np.array(cajas, dtype=np.float32, copy=True)
    c[:, [0, 2]] = (c[:, [0, 2]] - g["izquierda"]) / gx
    c[:, [1, 3]] = (c[:, [1, 3]] - g["arriba"]) / gy
    c[:, [0, 2]] = c[:, [0, 2]].clip(0, ancho)
    c[:, [1, 3]] = c[:, [1, 3]].clip(0, alto)
    return c


# --- Backends -------------------------------------------------------------------------------------------------------
class BackendUltralytics:
    """Backend de referencia: PyTorch + Ultralytics (GPU en entrenamiento, CPU en verificación)."""
    nombre = "ultralytics"

    def __init__(self, ruta_pesos, dispositivo="cpu"):
        from ultralytics import YOLO
        self.modelo = YOLO(str(ruta_pesos), task="segment")
        self.dispositivo = dispositivo

    def inferir(self, rgb, imgsz, conf, iou, max_det):
        """(confianzas float32[N], máscaras bool[N, alto, ancho]). Ultralytics espera BGR en arreglos."""
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        r = self.modelo.predict(bgr, imgsz=int(imgsz), conf=float(conf), iou=float(iou), max_det=int(max_det),
                                retina_masks=True, rect=False, device=self.dispositivo, verbose=False)[0]
        if r.masks is None or len(r.boxes) == 0:
            return np.zeros(0, np.float32), np.zeros((0, *rgb.shape[:2]), bool)
        mascaras = r.masks.data.cpu().numpy() > 0
        if mascaras.shape[1:] != rgb.shape[:2]:
            raise RuntimeError(f"El backend devolvió máscaras {mascaras.shape[1:]} y la imagen es "
                               f"{rgb.shape[:2]}: no se deshizo el letterbox.")
        return r.boxes.conf.cpu().numpy().astype(np.float32), mascaras


class BackendOnnx:
    """Backend de servicio: ONNX Runtime en CPU, sin torch ni Ultralytics, con posprocesado equivalente."""
    nombre = "onnx"

    def __init__(self, ruta_onnx, dispositivo="cpu", hilos=None):
        import onnxruntime as ort
        opciones = ort.SessionOptions()
        if hilos:
            opciones.intra_op_num_threads = int(hilos)
        self.sesion = ort.InferenceSession(str(ruta_onnx), sess_options=opciones,
                                           providers=["CPUExecutionProvider"])
        self.entrada = self.sesion.get_inputs()[0]
        self.dispositivo = "cpu"
        forma = self.entrada.shape
        self.tam_fijo = int(forma[2]) if isinstance(forma[2], int) else None

    def inferir(self, rgb, imgsz, conf, iou, max_det):
        alto, ancho = rgb.shape[:2]
        if self.tam_fijo and self.tam_fijo != int(imgsz):
            raise RuntimeError(f"El ONNX se exportó con imgsz={self.tam_fijo} y el contrato pide {imgsz}.")
        lienzo, _ = letterbox(rgb, int(imgsz))
        x = np.ascontiguousarray(lienzo.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        salidas = self.sesion.run(None, {self.entrada.name: x})
        det, protos = (salidas[0], salidas[1]) if salidas[0].ndim == 3 else (salidas[1], salidas[0])
        det, protos = det[0], protos[0]
        nm = protos.shape[0]
        if det.shape[-1] != 6 + nm:
            raise RuntimeError(f"Salida ONNX inesperada {det.shape}: se espera un modelo de extremo a extremo "
                               f"(N, 6 + {nm}) con cajas xyxy, confianza, clase y coeficientes de máscara.")
        det = det[det[:, 4] > conf][:int(max_det)]
        if len(det) == 0:
            return np.zeros(0, np.float32), np.zeros((0, alto, ancho), bool)
        cajas = cajas_a_original(det[:, :4], alto, ancho, int(imgsz))
        logits = (det[:, 6:].astype(np.float32) @ protos.reshape(nm, -1).astype(np.float32)).reshape(
            -1, *protos.shape[1:])
        mascaras = np.zeros((len(det), alto, ancho), bool)
        col, fil = np.arange(ancho, dtype=np.float32), np.arange(alto, dtype=np.float32)
        for k in range(len(det)):
            m = deshacer_letterbox(np.ascontiguousarray(logits[k]), alto, ancho, int(imgsz)) > 0
            x1, y1, x2, y2 = cajas[k]
            m &= ((col >= x1) & (col < x2))[None, :]
            m &= ((fil >= y1) & (fil < y2))[:, None]
            mascaras[k] = m
        conserva = mascaras.any(axis=(1, 2))
        return det[conserva, 4].astype(np.float32), mascaras[conserva]


# --- Predicción -------------------------------------------------------------------------------------------------------
def mapa_puntuacion(confianzas, mascaras, forma) -> np.ndarray:
    """Máximo de las confianzas de las instancias que cubren cada píxel; 0 fuera de propuestas."""
    punt = np.zeros(forma, np.float32)
    for c, m in zip(confianzas, mascaras):
        np.maximum(punt, np.float32(c), out=punt, where=m)
    return punt


def ventanas(alto, ancho, tesela, solape):
    """Ventanas (fila, columna, alto, ancho) a escala nativa. Una sola si la imagen cabe en la tesela."""
    def inicios(n):
        if n <= tesela:
            return [0]
        paso = max(1, tesela - solape)
        xs = list(range(0, n - tesela, paso))
        return xs + [n - tesela]
    return [(f, c, min(tesela, alto), min(tesela, ancho)) for f in inicios(alto) for c in inicios(ancho)]


def predecir(imagen, paquete, umbral_operativo=None) -> Prediccion:
    """Predicción sobre una imagen ya preparada (`preparar_entrada`) con un paquete cargado (`cargar_paquete`)."""
    if imagen.dtype != np.uint8 or imagen.ndim != 3 or imagen.shape[2] != 3:
        raise ValueError(f"La imagen debe ser uint8 (alto, ancho, 3); se recibió {imagen.dtype} {imagen.shape}.")
    pos, ent = paquete.contrato["posproceso"], paquete.contrato["entrada"]
    conf = float(pos["umbral_propuestas"])
    if conf <= 0:
        raise ValueError("El umbral de propuestas debe ser positivo.")
    umbral = validar_umbral(pos["umbral_operativo"] if umbral_operativo is None else umbral_operativo, conf)
    max_det, imgsz = int(pos["max_det"]), int(ent["imgsz"])
    alto, ancho = imagen.shape[:2]
    punt = np.zeros((alto, ancho), np.float32)
    instancias, tope = [], False
    lista = ventanas(alto, ancho, int(ent.get("tesela_px", max(alto, ancho))), int(ent.get("solape_px", 0)))
    t0 = time.perf_counter()
    for n, (f, c, h, w) in enumerate(lista):
        recorte = np.ascontiguousarray(imagen[f:f + h, c:c + w])
        with paquete.candado:
            confs, mascaras = paquete.backend.inferir(recorte, imgsz, conf, float(pos.get("iou_nms", 0.7)), max_det)
        tope |= len(confs) >= max_det
        vista = punt[f:f + h, c:c + w]
        np.maximum(vista, mapa_puntuacion(confs, mascaras, (h, w)), out=vista)
        for k in range(len(confs)):
            filas, cols = np.nonzero(mascaras[k])
            instancias.append({"confianza": float(confs[k]), "area_px": int(len(filas)), "tesela": n,
                               "caja": [int(cols.min()) + c, int(filas.min()) + f, int(cols.max()) + c + 1,
                                        int(filas.max()) + f + 1]})
    return Prediccion(punt, punt >= np.float32(umbral), alto, ancho, conf, umbral, instancias, max_det, tope,
                      len(lista), imgsz, paquete.backend.nombre, str(paquete.backend.dispositivo),
                      time.perf_counter() - t0)
