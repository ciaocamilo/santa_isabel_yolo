"""Composición visual compartida por el notebook y la app (solo NumPy y Pillow)."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from .areas import CATEGORIAS, COLORES_CAMBIO, FONDO, SIN_OBSERVACION

COLOR_NIEVE = (0, 200, 255)
COLOR_CONTORNO = (255, 255, 0)
COLOR_NO_OBSERVABLE = (120, 120, 120)
COLOR_REFERENCIA = (255, 0, 255)
LEYENDA_ESCENA = [("nieve/hielo visible (predicción)", COLOR_NIEVE), ("contorno de la predicción", COLOR_CONTORNO),
                  ("no observable (nube, sombra o sin dato)", COLOR_NO_OBSERVABLE),
                  ("fondo observado", (245, 240, 225))]


def contorno(mascara) -> np.ndarray:
    """Píxeles de borde de una máscara (4-vecindad)."""
    m = np.asarray(mascara, bool)
    interior = m.copy()
    interior[1:, :] &= m[:-1, :]
    interior[:-1, :] &= m[1:, :]
    interior[:, 1:] &= m[:, :-1]
    interior[:, :-1] &= m[:, 1:]
    interior[0, :] = interior[-1, :] = False
    interior[:, 0] = interior[:, -1] = False
    return m & ~interior


def _mezclar(base, mascara, color, alfa):
    if mascara is None or not np.any(mascara):
        return
    base[mascara] = (1 - alfa) * base[mascara] + alfa * np.array(color, np.float32)


def realzar(rgb, percentil=99.5) -> np.ndarray:
    """Estira el brillo solo para visualizar (la inferencia usa siempre la imagen sin realzar)."""
    rgb = np.asarray(rgb, np.float32)
    validos = rgb[rgb.sum(axis=2) > 0]
    tope = float(np.percentile(validos, percentil)) if validos.size else 255.0
    return np.clip(rgb * (255.0 / max(tope, 1.0)), 0, 255).astype(np.uint8)


def superponer(rgb, mascara=None, observable=None, opacidad=0.45, referencia=None, con_contorno=True) -> np.ndarray:
    """Imagen con la nieve predicha en cian, su contorno en amarillo, lo no observable en gris y, si se da,
    el contorno de la referencia en magenta."""
    base = np.asarray(rgb, np.float32).copy()
    if observable is not None:
        _mezclar(base, ~np.asarray(observable, bool), COLOR_NO_OBSERVABLE, 0.65)
    if mascara is not None:
        m = np.asarray(mascara, bool)
        _mezclar(base, m, COLOR_NIEVE, float(opacidad))
        if con_contorno:
            _mezclar(base, contorno(m), COLOR_CONTORNO, 1.0)
    if referencia is not None:
        _mezclar(base, contorno(referencia), COLOR_REFERENCIA, 1.0)
    return np.clip(base, 0, 255).astype(np.uint8)


def mapa_cambio_rgb(mapa, fondo_rgb=None) -> np.ndarray:
    """Mapa categórico de cambio en color. Si se da una imagen de fondo, el «fondo observado» la muestra
    atenuada para poder localizar los parches en el terreno."""
    mapa = np.asarray(mapa)
    salida = np.zeros((*mapa.shape, 3), np.uint8)
    for k, color in COLORES_CAMBIO.items():
        salida[mapa == k] = color[:3]
    if fondo_rgb is not None and np.asarray(fondo_rgb).shape[:2] == mapa.shape:
        gris = np.asarray(fondo_rgb, np.float32).mean(axis=2, keepdims=True)
        tenue = np.clip(0.55 * gris + 105, 0, 255).astype(np.uint8).repeat(3, axis=2)
        salida[mapa == FONDO] = tenue[mapa == FONDO]
        oscuro = np.clip(0.35 * gris + 60, 0, 255).astype(np.uint8).repeat(3, axis=2)
        salida[mapa == SIN_OBSERVACION] = oscuro[mapa == SIN_OBSERVACION]
    return salida


def leyenda_cambio():
    return [(CATEGORIAS[k], COLORES_CAMBIO[k][:3]) for k in (3, 4, 2, 1, 0)]


def leyenda_html(elementos) -> str:
    """Leyenda separada de la imagen, como HTML en línea."""
    partes = []
    for texto, color in elementos:
        partes.append(f"<span style='display:inline-block;margin:0 14px 4px 0;white-space:nowrap'>"
                      f"<span style='display:inline-block;width:14px;height:14px;border:1px solid #444;"
                      f"vertical-align:middle;background:rgb{tuple(int(c) for c in color)}'></span> {texto}</span>")
    return "<div style='font-size:0.85rem;line-height:1.6'>" + "".join(partes) + "</div>"


def ampliar(imagen, factor=2) -> np.ndarray:
    """Amplía por vecino más próximo (solo visualización; las áreas se calculan a resolución de análisis)."""
    return np.repeat(np.repeat(np.asarray(imagen), int(factor), axis=0), int(factor), axis=1)


def a_png(imagen) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(imagen)).save(buf, format="PNG")
    return buf.getvalue()


def numero(valor, decimales=2, sufijo="", signo=False) -> str:
    """Número con coma decimal; None se muestra como N/D."""
    if valor is None:
        return "N/D"
    texto = f"{valor:+,.{decimales}f}" if signo else f"{valor:,.{decimales}f}"
    return texto.replace(",", "\u202f").replace(".", ",") + sufijo
