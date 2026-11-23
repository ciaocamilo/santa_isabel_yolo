"""Fixtures deterministas: no usan GEE, pesos ni el dataset definitivo.

La predicción simulada (`BackendFalso`, `prediccion_de`) solo existe aquí: la app y el notebook nunca la usan.
"""
import santa_isabel  # noqa: F401  (fija PROJ/GDAL y la capa de hilos de MKL antes de importar numpy)

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import Affine

from santa_isabel import cargar_config
from santa_isabel.areas import Analisis
from santa_isabel.escenas import (NODATA, Escena, Georef, codificar_espectral, codificar_rgb, sha256_archivo,
                                  soporte_de)
from santa_isabel.inferencia import Prediccion
from santa_isabel.paquete import CLASE, ESQUEMA, TAREA, Modelo

CRS = "EPSG:32618"
X0, Y1 = 455970.0, 533560.0


def georef(alto=40, ancho=50, x0=X0, y1=Y1, res=10.0, crs=CRS) -> Georef:
    return Georef(crs, (res, 0.0, x0, 0.0, -res, y1), ancho, alto)


def fecha(texto="2024-08-15"):
    return datetime.fromisoformat(texto).replace(tzinfo=timezone.utc)


def escena(clave="e", cuando="2024-08-15", g="auto", observable="todo", extension=None, alto=40, ancho=50,
           perfil="L2A") -> Escena:
    g = georef(alto, ancho) if g == "auto" else g
    if g is not None:
        alto, ancho = g.alto, g.ancho
    obs = np.ones((alto, ancho), bool) if isinstance(observable, str) else observable
    return Escena(clave=clave, tipo="paquete_productor", alto=alto, ancho=ancho,
                  fecha=None if cuando is None else fecha(cuando), perfil=perfil,
                  rgb8=np.zeros((alto, ancho, 3), np.uint8),
                  extension=np.ones((alto, ancho), bool) if extension is None else extension,
                  observable=obs, procedencia_observable="fixture" if obs is not None else "sin QA",
                  georef=g, escala_verificada=g is not None)


def prediccion_de(mascara, confianza=0.9, umbral=0.5) -> Prediccion:
    mascara = np.asarray(mascara, bool)
    return Prediccion(puntuacion=mascara.astype(np.float32) * np.float32(confianza), mascara=mascara,
                      alto=mascara.shape[0], ancho=mascara.shape[1], umbral_propuestas=0.01,
                      umbral_operativo=umbral, backend="fixture", dispositivo="cpu")


def analisis(mascara, **kw) -> Analisis:
    mascara = np.asarray(mascara, bool)
    kw.setdefault("alto", mascara.shape[0])
    kw.setdefault("ancho", mascara.shape[1])
    return Analisis(escena(**kw), prediccion_de(mascara))


class BackendFalso:
    """Devuelve instancias prefijadas. `por_llamada` permite una respuesta distinta por tesela."""
    nombre = "falso"
    dispositivo = "cpu"

    def __init__(self, confianzas=(), mascaras=None, por_llamada=None):
        self.confianzas, self.mascaras, self.por_llamada, self.llamadas = confianzas, mascaras, por_llamada, []

    def inferir(self, rgb, imgsz, conf, iou, max_det):
        self.llamadas.append({"forma": rgb.shape, "imgsz": imgsz, "conf": conf, "max_det": max_det})
        if self.por_llamada is not None:
            confs, masc = self.por_llamada(len(self.llamadas) - 1, rgb)
        else:
            confs, masc = self.confianzas, self.mascaras
        confs = np.asarray(confs, np.float32)
        masc = np.zeros((0, *rgb.shape[:2]), bool) if masc is None else np.asarray(masc, bool)
        sel = confs > conf
        return confs[sel][:max_det], masc[sel][:max_det]


def modelo_falso(backend, variante="rgb", umbral=0.5, tesela=640, solape=96, imgsz=1024, max_det=300) -> Modelo:
    contrato = {"esquema": ESQUEMA, "id": "fixture", "tarea": TAREA, "clase": CLASE, "variante": variante,
                "entrada": {"imgsz": imgsz, "tesela_px": tesela, "solape_px": solape},
                "posproceso": {"umbral_propuestas": 0.01, "umbral_operativo": umbral, "iou_nms": 0.7,
                               "max_det": max_det},
                "backends": {"falso": {"archivo": "n/a"}}, "dataset": {"version": "fixture"},
                "referencia": {"origen": "fixture"}}
    return Modelo(ruta=Path("."), contrato=contrato, backend=backend)


def reflectancia_sintetica(alto=30, ancho=36, semilla=0) -> dict:
    """Cinco bandas con un «glaciar» brillante, vegetación y un borde sin dato."""
    rng = np.random.default_rng(semilla)
    refl = {b: rng.uniform(0.03, 0.25, (alto, ancho)).astype(np.float32) for b in ("B2", "B3", "B4", "B8", "B11")}
    refl["B8"] += 0.2
    nieve = np.zeros((alto, ancho), bool)
    nieve[8:18, 10:24] = True
    for b, v in (("B2", 0.75), ("B3", 0.7), ("B4", 0.66), ("B8", 0.55), ("B11", 0.06)):
        refl[b][nieve] = v
    for v in refl.values():
        v[:, :2] = np.nan
    return refl


def escribir_tif(ruta, datos, g: Georef | None, nodata=None, descripciones=None, etiquetas=None):
    datos = np.asarray(datos)
    if datos.ndim == 2:
        datos = datos[None]
    perfil = dict(driver="GTiff", width=datos.shape[2], height=datos.shape[1], count=datos.shape[0],
                  dtype=str(datos.dtype), nodata=nodata)
    if g is not None:
        perfil.update(crs=g.crs, transform=Affine(*g.afin))
    with rasterio.open(ruta, "w", **perfil) as ds:
        ds.write(datos)
        if descripciones:
            ds.descriptions = tuple(descripciones)
        if etiquetas:
            ds.update_tags(**etiquetas)
    return Path(ruta)


def crear_paquete_productor(carpeta, clave="20240815_T18NVL_20240815T152651_20240815T152653",
                            adquisicion="2024-08-15T15:31:40Z", refl=None, calidad=None, con_qa=True) -> Path:
    """Paquete de escena con la misma estructura que la caché del generador mensual."""
    carpeta = Path(carpeta)
    (carpeta / "rgb").mkdir(parents=True, exist_ok=True)
    refl = reflectancia_sintetica() if refl is None else refl
    alto, ancho = refl["B4"].shape
    g = georef(alto, ancho)
    soporte = soporte_de(refl)
    Image.fromarray(codificar_rgb(refl, soporte)).save(carpeta / "rgb" / f"{clave}.png")
    pila = np.stack([np.where(np.isfinite(refl[b]), refl[b], NODATA) for b in ("B2", "B3", "B4", "B8", "B11")])
    escribir_tif(carpeta / "reflectancia.tif", pila.astype(np.float32), g, NODATA, ("B2", "B3", "B4", "B8", "B11"))
    if con_qa:
        if calidad is None:
            calidad = np.where(soporte, 2, 0).astype(np.uint8)
            calidad[:5, -6:] = np.where(soporte[:5, -6:], 1, 0)  # una «nube» observada y no visible
        escribir_tif(carpeta / "calidad.tif", calidad.astype(np.uint8), g)
        escribir_tif(carpeta / "soporte_png.tif", soporte.astype(np.uint8), g)
    rel = f"rgb/{clave}.png"
    meta = {"estado": "completa", "clave": clave, "nombre_escena": clave, "perfil": "L2A",
            "nivel": "L2A reflectancia de superficie", "id_asset": f"COPERNICUS/S2_SR_HARMONIZED/{clave}",
            "adquisicion_utc": adquisicion, "metodo_qa": "SCL clases útiles 4,5,6,11",
            "malla": {"crs": g.crs, "transform": list(g.afin), "ancho": ancho, "alto": alto},
            "receta": {"version": "2.0", "huella": "fixture00000"},
            "rutas": {"rgb": rel, "reflectancia": "reflectancia.tif"},
            "archivos": {rel: {"sha256": sha256_archivo(carpeta / rel)}},
            "atribucion": "Datos sintéticos de prueba (no son Sentinel-2)."}
    (carpeta / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return carpeta


@pytest.fixture
def cfg_real():
    """Configuración del proyecto (experimento.toml + local.toml si existe)."""
    return cargar_config()


@pytest.fixture
def cfg_sin_aoi():
    return {"aoi": {"crs": CRS}, "comparacion": {"meses_tolerancia": 1}}


__all__ = ["georef", "fecha", "escena", "prediccion_de", "analisis", "BackendFalso", "modelo_falso",
           "reflectancia_sintetica", "escribir_tif", "crear_paquete_productor", "codificar_espectral"]
