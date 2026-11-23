"""Lectura de escenas Sentinel-2 del Santa Isabel a una estructura común (`Escena`).

Admite cuatro tipos de carga:

* paquete de escena del generador mensual (metadata.json + PNG + calidad.tif/soporte_png.tif);
* GeoTIFF RGB de 8 bits ya codificado, o multibanda de reflectancia (B2, B3, B4, B8, B11);
* PNG/JPG RGB de 8 bits (inferencia visual; recupera malla y QA si su hash es de una escena conocida).

La receta radiométrica es la del productor (receta 2.0): R=B4, G=B3, B=B2 con
round(255·clip(refl/0,60; 0; 1)) y 0 fuera del soporte; la variante espectral es NDSI/NDVI/B4.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

RECETA_VERSION = "2.0"
RECETAS_COMPATIBLES = ("1.0", "2.0")  # la 1.0 del selector produce los mismos PNG (regresión del productor)
RGB_MAX_REFL = 0.60
NODATA = -9999.0
ESCALA_DN = 0.0001
BANDAS_PRODUCTOR = ("B2", "B3", "B4", "B8", "B11")
BANDAS_RGB = ("B4", "B3", "B2")
BANDAS_ESPECTRAL = ("B3", "B4", "B8", "B11")
TEXTO_RECETA_RGB = "R=B4,G=B3,B=B2; round(255*clip(refl/0.60,0,1)); fuera de soporte = 0"
TEXTO_RECETA_ESPECTRAL = "R=NDSI,G=NDVI: round(255*(clip(i,-1,1)+1)/2); B=B4 escala RGB; fuera de soporte = 0"
AUXILIARES = {"calidad.tif", "soporte_png.tif", "qa.tif", "indices.tif"}
LIMITES_POR_DEFECTO = {"max_bytes": 60_000_000, "max_pixeles": 4_000_000, "max_bandas": 16,
                       "zip_max_archivos": 40, "zip_max_bytes_expandido": 200_000_000}


class ErrorEscena(ValueError):
    """Entrada no utilizable. `requiere` indica qué dato debe aportar el usuario (p. ej. 'mapeo_bandas')."""

    def __init__(self, mensaje, requiere=None):
        super().__init__(mensaje)
        self.requiere = requiere


@dataclass(frozen=True)
class Georef:
    """Malla georreferenciada: afín (a, b, c, d, e, f) con x = a·col + b·fila + c, y = d·col + e·fila + f."""
    crs: str
    afin: tuple
    ancho: int
    alto: int

    @property
    def affine(self):
        from rasterio.transform import Affine
        return Affine(*self.afin)

    @property
    def ha_por_celda(self) -> float:
        a, b, _, d, e, _ = self.afin
        return abs(a * e - b * d) / 10000.0

    @property
    def es_metrico(self) -> bool:
        from rasterio.crs import CRS
        try:
            crs = CRS.from_user_input(self.crs)
        except Exception:
            return False
        return bool(crs.is_projected) and str(crs.linear_units).lower() in ("metre", "meter", "m")

    @property
    def limites(self):
        """(oeste, sur, este, norte) de las cuatro esquinas, válido también con afín rotada."""
        t = self.affine
        xs, ys = zip(*(t * p for p in ((0, 0), (self.ancho, 0), (0, self.alto), (self.ancho, self.alto))))
        return min(xs), min(ys), max(xs), max(ys)

    def misma_malla(self, otra, tol=1e-6) -> bool:
        from rasterio.crs import CRS
        return (otra is not None and (self.ancho, self.alto) == (otra.ancho, otra.alto)
                and CRS.from_user_input(self.crs) == CRS.from_user_input(otra.crs)
                and all(abs(p - q) <= tol for p, q in zip(self.afin, otra.afin)))

    def como_dict(self):
        return {"crs": self.crs, "transform": list(self.afin), "ancho": self.ancho, "alto": self.alto,
                "ha_por_celda": self.ha_por_celda}


@dataclass
class Escena:
    clave: str
    tipo: str                      # paquete_productor | geotiff_rgb | geotiff_reflectancia | imagen_rgb
    alto: int
    ancho: int
    fecha: datetime | None = None  # adquisición UTC
    perfil: str | None = None      # L2A | L1C
    rgb8: np.ndarray | None = None           # (alto, ancho, 3) uint8 en orden R, G, B con la receta RGB
    reflectancia: dict | None = None         # {"B4": float32[alto, ancho], ...} en reflectancia 0–1, NaN sin dato
    extension: np.ndarray | None = None      # bool: píxeles con datos (soporte del PNG)
    observable: np.ndarray | None = None     # bool: píxeles observables (QA); None si no hay QA
    procedencia_observable: str = "sin QA: no se distingue cielo observable de nubes o sombras"
    georef: Georef | None = None
    escala_verificada: bool = False
    receta: dict = field(default_factory=dict)
    fuente: str = ""
    atribucion: str = ""
    sha256: str | None = None
    avisos: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @property
    def forma(self):
        return self.alto, self.ancho

    @property
    def tiene_qa(self) -> bool:
        return self.observable is not None

    def resumen(self) -> dict:
        return {"clave": self.clave, "tipo": self.tipo, "fecha_utc": iso_utc(self.fecha), "perfil": self.perfil,
                "ancho": self.ancho, "alto": self.alto, "georreferenciada": self.georef is not None,
                "crs": self.georef.crs if self.georef else None, "qa": self.procedencia_observable,
                "receta": self.receta, "fuente": self.fuente, "sha256": self.sha256, "avisos": list(self.avisos)}


# --- Utilidades -------------------------------------------------------------------------------------------------
def sha256_archivo(ruta, bloque=1 << 20) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        while True:
            b = f.read(bloque)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def iso_utc(fecha) -> str | None:
    return None if fecha is None else fecha.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def interpretar_fecha(valor) -> datetime | None:
    """Acepta datetime, date, 'AAAA-MM-DD', ISO con Z y el formato TIFF 'AAAA:MM:DD HH:MM:SS'."""
    if valor is None or valor == "":
        return None
    if isinstance(valor, datetime):
        return valor if valor.tzinfo else valor.replace(tzinfo=timezone.utc)
    if hasattr(valor, "year") and hasattr(valor, "month"):
        return datetime(valor.year, valor.month, valor.day, tzinfo=timezone.utc)
    texto = str(valor).strip()
    if re.fullmatch(r"\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}", texto):
        return datetime.strptime(texto, "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)
    try:
        f = datetime.fromisoformat(texto.replace("Z", "+00:00"))
    except ValueError as e:
        raise ErrorEscena(f"Fecha no reconocida: {valor!r}. Use AAAA-MM-DD o ISO 8601 en UTC.", "fecha") from e
    return f if f.tzinfo else f.replace(tzinfo=timezone.utc)


def fecha_de_nombre(nombre) -> datetime | None:
    """Fecha de un nombre tipo Sentinel-2 (…20240123T152649…) o con AAAAMMDD / AAAA-MM-DD aislado."""
    m = re.search(r"(?<!\d)(20[1-4]\d)(0[1-9]|1[0-2])([0-3]\d)T(\d{2})(\d{2})(\d{2})", str(nombre))
    try:
        if m:
            return datetime(*(int(x) for x in m.groups()), tzinfo=timezone.utc)
        m = re.search(r"(?<!\d)(20[1-4]\d)-?(0[1-9]|1[0-2])-?([0-3]\d)(?!\d)", str(nombre))
        if m:
            return datetime(*(int(x) for x in m.groups()), tzinfo=timezone.utc)
    except ValueError:
        return None
    return None


# --- Receta radiométrica del productor --------------------------------------------------------------------------
def a_uint8_reflectancia(r):
    return np.floor(np.clip(np.asarray(r, dtype=np.float64) / RGB_MAX_REFL, 0, 1) * 255 + 0.5).astype(np.uint8)


def a_uint8_indice(i):
    return np.floor((np.clip(np.asarray(i, dtype=np.float64), -1, 1) + 1) / 2 * 255 + 0.5).astype(np.uint8)


def _validas(refl: dict, bandas):
    ok = None
    for b in bandas:
        v = np.asarray(refl[b])
        m = np.isfinite(v) & (v != NODATA)
        ok = m if ok is None else ok & m
    return ok


def indices(refl: dict):
    """NDSI=(B3−B11)/(B3+B11) y NDVI=(B8−B4)/(B8+B4) en float32; NaN donde no hay reflectancia válida o el
    denominador es cero. Nunca se derivan de un RGB."""
    faltan = [b for b in BANDAS_ESPECTRAL if b not in refl]
    if faltan:
        raise ErrorEscena(f"Para NDSI/NDVI faltan las bandas {', '.join(faltan)} (se exigen B3, B4, B8 y B11).",
                          "mapeo_bandas")
    b3, b4, b8, b11 = (np.asarray(refl[b], dtype=np.float32).astype(np.float64) for b in BANDAS_ESPECTRAL)
    valida = _validas(refl, [b for b in BANDAS_PRODUCTOR if b in refl])
    den_s, den_v = b3 + b11, b8 + b4
    with np.errstate(divide="ignore", invalid="ignore"):
        ndsi = np.where(valida & (den_s != 0), (b3 - b11) / den_s, np.nan).astype(np.float32)
        ndvi = np.where(valida & (den_v != 0), (b8 - b4) / den_v, np.nan).astype(np.float32)
    return ndsi, ndvi


def soporte_de(refl: dict) -> np.ndarray:
    """Soporte del PNG. Con las cinco bandas reproduce el del productor (NDSI y NDVI definidos); si solo hay
    bandas visibles, píxeles con B4, B3 y B2 válidas."""
    if all(b in refl for b in BANDAS_ESPECTRAL):
        ndsi, ndvi = indices(refl)
        return np.isfinite(ndsi) & np.isfinite(ndvi)
    return _validas(refl, BANDAS_RGB)


def codificar_rgb(refl: dict, soporte=None) -> np.ndarray:
    faltan = [b for b in BANDAS_RGB if b not in refl]
    if faltan:
        raise ErrorEscena(f"Para la variante RGB faltan las bandas {', '.join(faltan)}.", "mapeo_bandas")
    soporte = soporte_de(refl) if soporte is None else np.asarray(soporte, bool)
    rgb = np.stack([a_uint8_reflectancia(np.where(soporte, np.asarray(refl[b], np.float32).astype(np.float64), 0))
                    for b in BANDAS_RGB], axis=-1)
    rgb[~soporte] = 0
    return rgb


def codificar_espectral(refl: dict, soporte=None) -> np.ndarray:
    ndsi, ndvi = indices(refl)
    soporte = (np.isfinite(ndsi) & np.isfinite(ndvi)) if soporte is None else np.asarray(soporte, bool)
    b4 = np.asarray(refl["B4"], np.float32).astype(np.float64)
    esp = np.stack([a_uint8_indice(np.where(soporte, ndsi, 0)), a_uint8_indice(np.where(soporte, ndvi, 0)),
                    a_uint8_reflectancia(np.where(soporte, b4, 0))], axis=-1)
    esp[~soporte] = 0
    return esp


# --- Entradas -----------------------------------------------------------------------------------------------------
def extraer_zip_seguro(ruta_zip, destino, limites=None) -> Path:
    """Extrae un ZIP de escena limitando número de archivos y bytes expandidos; rechaza rutas que salgan
    del directorio de destino y enlaces."""
    lim = {**LIMITES_POR_DEFECTO, **(limites or {})}
    destino = Path(destino).resolve()
    destino.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ruta_zip) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        if len(infos) > lim["zip_max_archivos"]:
            raise ErrorEscena(f"El ZIP contiene {len(infos)} archivos; el máximo admitido es "
                              f"{lim['zip_max_archivos']}.")
        total = sum(i.file_size for i in infos)
        if total > lim["zip_max_bytes_expandido"]:
            raise ErrorEscena(f"El ZIP ocuparía {total / 1e6:.1f} MB al expandirse; el máximo admitido es "
                              f"{lim['zip_max_bytes_expandido'] / 1e6:.0f} MB.")
        for i in infos:
            nombre = i.filename.replace("\\", "/")
            objetivo = (destino / nombre).resolve()
            if nombre.startswith("/") or ".." in Path(nombre).parts or destino not in objetivo.parents:
                raise ErrorEscena(f"El ZIP contiene una ruta fuera del directorio temporal: {i.filename!r}.")
            if (i.external_attr >> 16) & 0o170000 == 0o120000:
                raise ErrorEscena(f"El ZIP contiene un enlace simbólico: {i.filename!r}.")
            objetivo.parent.mkdir(parents=True, exist_ok=True)
            escrito = 0
            with z.open(i) as src, open(objetivo, "wb") as dst:
                while True:
                    bloque = src.read(1 << 20)
                    if not bloque:
                        break
                    escrito += len(bloque)
                    if escrito > i.file_size or escrito > lim["zip_max_bytes_expandido"]:
                        raise ErrorEscena(f"{i.filename!r} se expande más de lo declarado.")
                    dst.write(bloque)
    return destino


def _normalizar_archivos(archivos, dir_temporal=None):
    """{nombre relativo posix: Path}. Acepta carpeta, archivo, lista de rutas o {nombre: bytes}."""
    if isinstance(archivos, dict):
        base = Path(tempfile.mkdtemp(prefix="escena_", dir=dir_temporal))
        rutas = {}
        for nombre, contenido in archivos.items():
            nombre = Path(str(nombre).replace("\\", "/")).name
            destino = base / nombre
            destino.write_bytes(contenido if isinstance(contenido, (bytes, bytearray)) else contenido.read())
            rutas[nombre] = destino
        return _expandir_zip(rutas, dir_temporal)
    if isinstance(archivos, (str, Path)):
        p = Path(archivos)
        if p.is_dir():
            return _expandir_zip({f.relative_to(p).as_posix(): f for f in sorted(p.rglob("*")) if f.is_file()},
                                 dir_temporal)
        if not p.is_file():
            raise ErrorEscena(f"No existe el archivo o carpeta de escena: {p}")
        return _expandir_zip({p.name: p}, dir_temporal)
    rutas = {}
    for a in archivos:
        p = Path(a)
        if not p.is_file():
            raise ErrorEscena(f"No existe el archivo de escena: {p}")
        rutas[p.name] = p
    return _expandir_zip(rutas, dir_temporal)


def _expandir_zip(rutas, dir_temporal):
    zips = [n for n in rutas if n.lower().endswith(".zip")]
    if not zips:
        return rutas
    if len(rutas) > 1:
        raise ErrorEscena("Cargue un único ZIP de escena, o bien los archivos sueltos de la escena.")
    destino = Path(tempfile.mkdtemp(prefix="zip_", dir=dir_temporal))
    try:
        extraer_zip_seguro(rutas[zips[0]], destino)
    except ErrorEscena:
        shutil.rmtree(destino, ignore_errors=True)
        raise
    return {f.relative_to(destino).as_posix(): f for f in sorted(destino.rglob("*")) if f.is_file()}


def _comprobar_bytes(ruta, lim):
    n = Path(ruta).stat().st_size
    if n > lim["max_bytes"]:
        raise ErrorEscena(f"{Path(ruta).name} pesa {n / 1e6:.1f} MB; el máximo admitido es "
                          f"{lim['max_bytes'] / 1e6:.0f} MB. Recorte la escena al AOI del Santa Isabel.")


def _comprobar_pixeles(ancho, alto, lim, nombre):
    if ancho * alto > lim["max_pixeles"]:
        raise ErrorEscena(f"{nombre} mide {ancho}×{alto} px ({ancho * alto / 1e6:.2f} Mpx); el máximo admitido "
                          f"es {lim['max_pixeles'] / 1e6:.1f} Mpx. Recorte la escena al AOI del Santa Isabel "
                          "(≈ 545×598 px a 10 m).")


def _leer_imagen_rgb8(ruta, lim):
    _comprobar_bytes(ruta, lim)
    avisos = []
    try:
        with Image.open(ruta) as im:
            _comprobar_pixeles(im.width, im.height, lim, Path(ruta).name)
            if im.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
                raise ErrorEscena(f"{Path(ruta).name} no es de 8 bits (modo {im.mode}). Cargue un RGB de 8 bits "
                                  "o un GeoTIFF de reflectancia.")
            if im.mode != "RGB":
                avisos.append(f"La imagen venía en modo {im.mode}; se convirtió a RGB de 8 bits.")
            arr = np.asarray(im.convert("RGB"))
    except ErrorEscena:
        raise
    except Exception as e:
        raise ErrorEscena(f"No se pudo decodificar {Path(ruta).name}: {e}. Formatos admitidos: PNG, JPG, "
                          "GeoTIFF o paquete de escena del generador.") from e
    return np.ascontiguousarray(arr), avisos


def _georef_de(ds):
    if ds.crs is None or ds.transform.is_identity:
        return None
    t = ds.transform
    crs = ds.crs.to_string()
    epsg = ds.crs.to_epsg()
    return Georef(f"EPSG:{epsg}" if epsg else crs, (t.a, t.b, t.c, t.d, t.e, t.f), ds.width, ds.height)


def _leer_raster_1b(ruta, georef, nombre):
    import rasterio
    with rasterio.open(ruta) as ds:
        g = _georef_de(ds)
        if georef is not None and (g is None or not georef.misma_malla(g)):
            raise ErrorEscena(f"{nombre} no está en la malla de la escena (CRS, afín o dimensiones distintos).")
        if georef is None and (ds.width, ds.height) != (None, None) and g is not None:
            pass
        return ds.read(1), g


# --- Lectores por tipo --------------------------------------------------------------------------------------------
def _leer_paquete_productor(rutas, nombre_meta, lim):
    base = str(Path(nombre_meta).parent.as_posix())
    prefijo = "" if base == "." else base + "/"
    try:
        meta = json.loads(rutas[nombre_meta].read_text(encoding="utf-8"))
    except Exception as e:
        raise ErrorEscena(f"metadata.json no es un JSON válido: {e}") from e
    for campo in ("clave", "adquisicion_utc", "malla"):
        if campo not in meta:
            raise ErrorEscena(f"metadata.json no tiene el campo obligatorio «{campo}».")
    malla = meta["malla"]
    for campo in ("crs", "transform", "ancho", "alto"):
        if campo not in malla:
            raise ErrorEscena(f"metadata.json: malla sin «{campo}».")
    georef = Georef(str(malla["crs"]), tuple(float(v) for v in malla["transform"][:6]),
                    int(malla["ancho"]), int(malla["alto"]))
    clave = str(meta["clave"])
    rel_rgb = (meta.get("rutas") or {}).get("rgb") or f"rgb/{clave}.png"
    candidatos = [prefijo + rel_rgb, prefijo + "rgb.png", prefijo + f"{clave}.png"]
    nombre_rgb = next((c for c in candidatos if c in rutas), None)
    if nombre_rgb is None:
        raise ErrorEscena(f"El paquete de escena {clave} no incluye la imagen RGB ({rel_rgb}).")
    rgb, avisos = _leer_imagen_rgb8(rutas[nombre_rgb], lim)
    if rgb.shape[:2] != (georef.alto, georef.ancho):
        raise ErrorEscena(f"La imagen mide {rgb.shape[1]}×{rgb.shape[0]} px y metadata.json declara "
                          f"{georef.ancho}×{georef.alto}: la escena no corresponde a su metadato.")
    sha = sha256_archivo(rutas[nombre_rgb])
    esperado = ((meta.get("archivos") or {}).get(rel_rgb) or {}).get("sha256")
    if esperado and esperado != sha:
        raise ErrorEscena(f"Hash erróneo en {rel_rgb}: el PNG no es el que registró el generador "
                          f"(esperado {esperado[:12]}…, leído {sha[:12]}…). Vuelva a copiar la escena desde la caché.")
    receta_meta = meta.get("receta") or {}
    version = str(receta_meta.get("version", ""))
    if version and version not in RECETAS_COMPATIBLES:
        raise ErrorEscena(f"Receta radiométrica {version} no compatible (admitidas: "
                          f"{', '.join(RECETAS_COMPATIBLES)}).")
    perfil = meta.get("perfil") or ("L2A" if "S2_SR" in str(meta.get("coleccion", meta.get("id_asset", "")))
                                    else "L1C")
    escena = Escena(
        clave=clave, tipo="paquete_productor", alto=georef.alto, ancho=georef.ancho,
        fecha=interpretar_fecha(meta["adquisicion_utc"]), perfil=perfil, rgb8=rgb, georef=georef,
        escala_verificada=True, sha256=sha, avisos=avisos,
        receta={"version": version or None, "huella": receta_meta.get("huella"), "verificada": bool(esperado),
                "rgb": TEXTO_RECETA_RGB, "variantes": ["rgb"]},
        fuente=f"{meta.get('id_asset', clave)} · {meta.get('nivel') or perfil}",
        atribucion=meta.get("atribucion", ""),
        extra={"id_asset": meta.get("id_asset"), "satelite": meta.get("satelite"), "tesela": meta.get("tesela"),
               "metodo_qa": meta.get("metodo_qa"), "porcentajes": (meta.get("porcentajes") or {}).get("local_10m"),
               "origen_ejemplo": meta.get("origen_ejemplo")})
    n_cal, n_sop = prefijo + "calidad.tif", prefijo + "soporte_png.tif"
    if n_cal in rutas and n_sop in rutas:
        calidad, _ = _leer_raster_1b(rutas[n_cal], georef, "calidad.tif")
        soporte, _ = _leer_raster_1b(rutas[n_sop], georef, "soporte_png.tif")
        escena.extension = soporte == 1
        escena.observable = (calidad == 2) & escena.extension
        escena.procedencia_observable = (f"calidad.tif = 2 ({meta.get('metodo_qa') or 'QA del productor'}) "
                                         "y soporte_png.tif = 1")
        escena.extra["calidad"] = calidad
    else:
        escena.extension = np.ones(georef.alto * georef.ancho, bool).reshape(georef.alto, georef.ancho)
        escena.avisos.append("El paquete no incluye calidad.tif y soporte_png.tif: sin máscara de observabilidad.")
    n_refl = prefijo + "reflectancia.tif"
    if n_refl in rutas:
        escena.extra["ruta_reflectancia"] = str(rutas[n_refl])
        escena.receta["variantes"] = ["rgb", "espectral"]
    return escena


def cargar_reflectancia(escena: Escena) -> dict:
    """Reflectancia de un paquete del productor (se lee solo cuando hace falta, p. ej. variante espectral)."""
    if escena.reflectancia is not None:
        return escena.reflectancia
    ruta = escena.extra.get("ruta_reflectancia")
    if not ruta:
        raise ErrorEscena("Esta escena no trae reflectancia multibanda (B3/B4/B8/B11): solo admite la variante "
                          "RGB.", "mapeo_bandas")
    import rasterio
    with rasterio.open(ruta) as ds:
        nombres = [str(d).upper() if d else "" for d in ds.descriptions]
        if not all(b in nombres for b in BANDAS_PRODUCTOR):
            nombres = list(BANDAS_PRODUCTOR) if ds.count == 5 else nombres
        datos = ds.read().astype(np.float32)
        nodata = ds.nodata
    refl = {}
    for k, nombre in enumerate(nombres):
        if nombre in BANDAS_PRODUCTOR:
            v = datos[k]
            refl[nombre] = np.where((v == NODATA) | (False if nodata is None else v == nodata), np.nan, v)
    escena.reflectancia = refl
    return refl


def _nombres_bandas(ds, mapeo_bandas):
    """{nombre de banda: índice 1..n} a partir del mapeo explícito o de las descripciones del GeoTIFF."""
    if mapeo_bandas:
        mapeo = {str(k).upper(): int(v) for k, v in mapeo_bandas.items()}
        fuera = {k: v for k, v in mapeo.items() if not 1 <= v <= ds.count}
        if fuera:
            raise ErrorEscena(f"El mapeo de bandas apunta fuera del archivo ({fuera}); el GeoTIFF tiene "
                              f"{ds.count} banda(s).", "mapeo_bandas")
        if len(set(mapeo.values())) != len(mapeo):
            raise ErrorEscena("El mapeo de bandas asigna el mismo índice a dos bandas distintas.", "mapeo_bandas")
        return mapeo
    mapeo = {}
    for k, d in enumerate(ds.descriptions, 1):
        m = re.fullmatch(r"B0?(\d{1,2}A?)", str(d or "").strip().upper())
        if m:
            mapeo.setdefault("B" + m.group(1), k)
    return mapeo


def _leer_geotiff(ruta, perfil, fecha, mapeo_bandas, escala, desplazamiento, lim):
    import rasterio
    _comprobar_bytes(ruta, lim)
    nombre = Path(ruta).name
    try:
        ds = rasterio.open(ruta)
    except Exception as e:
        raise ErrorEscena(f"No se pudo abrir {nombre} como GeoTIFF: {e}") from e
    with ds:
        _comprobar_pixeles(ds.width, ds.height, lim, nombre)
        if ds.count > lim["max_bandas"]:
            raise ErrorEscena(f"{nombre} tiene {ds.count} bandas; el máximo admitido es {lim['max_bandas']}.")
        georef = _georef_de(ds)
        etiquetas = ds.tags()
        fecha = (interpretar_fecha(fecha) or interpretar_fecha(etiquetas.get("adquisicion_utc"))
                 or interpretar_fecha(etiquetas.get("TIFFTAG_DATETIME")) or fecha_de_nombre(nombre))
        avisos = []
        if georef is None:
            avisos.append("El TIFF no tiene CRS ni transformada: sin georreferencia no se calculan hectáreas.")
        descripciones = [str(d or "").strip().upper() for d in ds.descriptions]
        tipos = set(ds.dtypes)
        es_rgb8 = ds.count == 3 and tipos == {"uint8"} and not mapeo_bandas
        if es_rgb8:
            orden = [1, 2, 3]
            if descripciones == ["B2", "B3", "B4"] or descripciones == ["B", "G", "R"]:
                orden = [3, 2, 1]
                avisos.append("Las bandas venían en orden B2/B3/B4: se reordenaron a R=B4, G=B3, B=B2.")
            rgb = np.ascontiguousarray(np.stack([ds.read(k) for k in orden], axis=-1))
            mascara = ds.dataset_mask() > 0
            verificada = etiquetas.get("receta_version") in RECETAS_COMPATIBLES
            if not verificada:
                avisos.append("GeoTIFF RGB de 8 bits sin receta declarada: se asume la codificación del "
                              "productor (B4/B3/B2, reflectancia 0–0,60 → 0–255). Si procede de otra "
                              "visualización, la detección puede degradarse.")
            escena = Escena(clave=etiquetas.get("clave") or Path(nombre).stem, tipo="geotiff_rgb", alto=ds.height,
                            ancho=ds.width, fecha=fecha, perfil=perfil or etiquetas.get("perfil"), rgb8=rgb,
                            extension=mascara, georef=georef, escala_verificada=georef is not None,
                            receta={"version": etiquetas.get("receta_version"), "verificada": verificada,
                                    "rgb": TEXTO_RECETA_RGB, "variantes": ["rgb"]},
                            fuente=f"GeoTIFF RGB de 8 bits ({nombre})", avisos=avisos)
        else:
            mapeo = _nombres_bandas(ds, mapeo_bandas)
            if not all(b in mapeo for b in BANDAS_RGB):
                raise ErrorEscena(
                    f"{nombre} tiene {ds.count} banda(s) de tipo {', '.join(sorted(tipos))} sin nombres "
                    f"reconocibles ({descripciones}). Indique el mapeo explícito de bandas, por ejemplo "
                    "B2=1, B3=2, B4=3, B8=4, B11=5 (para la variante espectral se exigen B3, B4, B8 y B11).",
                    "mapeo_bandas")
            entero = all(np.issubdtype(np.dtype(t), np.integer) for t in tipos)
            datos = {b: ds.read(k).astype(np.float32) for b, k in mapeo.items() if b in BANDAS_PRODUCTOR}
            invalida = ~(ds.dataset_mask() > 0)
            for b, v in datos.items():
                v[invalida | (v == NODATA) | ~np.isfinite(v)] = np.nan
            maximo = max((float(np.nanmax(v)) for v in datos.values() if np.isfinite(v).any()), default=0.0)
            if escala is None:
                if entero or maximo > 2.0:
                    if perfil not in ("L2A", "L1C"):
                        raise ErrorEscena(
                            f"{nombre} trae valores digitales (máximo {maximo:g}) y no se indicó el nivel de "
                            "procesamiento. Seleccione L2A o L1C (reflectancia = DN × 0,0001, colecciones "
                            "HARMONIZED de Earth Engine) o indique la escala.", "perfil")
                    escala = ESCALA_DN
                    avisos.append(f"Unidades: valores digitales × {ESCALA_DN} (perfil {perfil}, colección "
                                  "HARMONIZED sin desplazamiento).")
                else:
                    escala = 1.0
                    avisos.append("Unidades: reflectancia 0–1 (sin reescalar).")
            refl = {b: (v * float(escala) + float(desplazamiento)).astype(np.float32) for b, v in datos.items()}
            soporte = soporte_de(refl)
            variantes = ["rgb"] + (["espectral"] if all(b in refl for b in BANDAS_ESPECTRAL) else [])
            if "espectral" not in variantes:
                avisos.append("Sin B8 y B11 solo es posible la variante RGB (los índices no se derivan de un RGB).")
            escena = Escena(clave=etiquetas.get("clave") or Path(nombre).stem, tipo="geotiff_reflectancia",
                            alto=ds.height, ancho=ds.width, fecha=fecha, perfil=perfil,
                            rgb8=codificar_rgb(refl, soporte), reflectancia=refl, extension=soporte,
                            georef=georef, escala_verificada=georef is not None,
                            receta={"version": RECETA_VERSION, "verificada": True, "rgb": TEXTO_RECETA_RGB,
                                    "variantes": variantes, "escala": float(escala),
                                    "desplazamiento": float(desplazamiento), "bandas": mapeo},
                            fuente=f"GeoTIFF de reflectancia ({nombre}; bandas {', '.join(sorted(mapeo))})",
                            avisos=avisos)
    escena.sha256 = sha256_archivo(ruta)
    if escena.fecha is None:
        escena.avisos.append("No se encontró la fecha de adquisición: indíquela para poder comparar fechas.")
    return escena


def _adjuntar_qa(escena, rutas):
    """Adjunta calidad.tif y soporte_png.tif sueltos a una escena GeoTIFF si están en la misma malla."""
    if "calidad.tif" not in rutas or escena.georef is None:
        return
    calidad, _ = _leer_raster_1b(rutas["calidad.tif"], escena.georef, "calidad.tif")
    extension = escena.extension
    if "soporte_png.tif" in rutas:
        soporte, _ = _leer_raster_1b(rutas["soporte_png.tif"], escena.georef, "soporte_png.tif")
        extension = extension & (soporte == 1)
    escena.observable = (calidad == 2) & extension
    escena.procedencia_observable = "calidad.tif = 2 (QA del productor) aportado junto al GeoTIFF"
    escena.extra["calidad"] = calidad


def _leer_imagen(ruta, fecha, indice, perfil, lim):
    rgb, avisos = _leer_imagen_rgb8(ruta, lim)
    sha = sha256_archivo(ruta)
    conocida = (indice or {}).get(sha)
    if conocida:
        escena = leer_escena(conocida["carpeta"], perfil)
        escena.avisos.append(f"PNG original del generador reconocido por su hash ({sha[:12]}…): se recuperaron "
                             f"malla y QA de «{conocida.get('origen', 'escena conocida')}».")
        escena.extra["recuperada_por_hash"] = True
        return escena
    nombre = Path(ruta).name
    avisos.append("Receta radiométrica desconocida: inferencia visual. Sin escala verificable no se calculan "
                  "hectáreas y sin georreferencia no hay comparación espacial.")
    if nombre.lower().endswith((".jpg", ".jpeg")):
        avisos.append("JPG es un formato con pérdida: los bordes de los parches pueden alterarse.")
    return Escena(clave=Path(nombre).stem, tipo="imagen_rgb", alto=rgb.shape[0], ancho=rgb.shape[1],
                  fecha=interpretar_fecha(fecha) or fecha_de_nombre(nombre), perfil=perfil, rgb8=rgb,
                  extension=np.ones(rgb.shape[:2], bool), sha256=sha,
                  receta={"version": None, "verificada": False, "rgb": "desconocida", "variantes": ["rgb"]},
                  fuente=f"Imagen RGB de 8 bits ({nombre})", avisos=avisos)


def leer_escena(archivos, perfil=None, *, fecha=None, mapeo_bandas=None, escala=None, desplazamiento=0.0,
                indice=None, limites=None, dir_temporal=None) -> Escena:
    """Lee una escena desde una carpeta, un archivo, una lista de archivos o {nombre: bytes}.

    `perfil` es 'L2A' o 'L1C' (necesario para GeoTIFF en valores digitales). `indice` es
    {sha256: {"carpeta": ruta, "origen": texto}} con escenas conocidas del generador, para que un PNG
    original recupere su malla y QA.
    """
    if perfil is not None and perfil not in ("L2A", "L1C"):
        raise ErrorEscena(f"Perfil {perfil!r} no admitido: use 'L2A' o 'L1C'.", "perfil")
    lim = {**LIMITES_POR_DEFECTO, **(limites or {})}
    rutas = _normalizar_archivos(archivos, dir_temporal)
    if not rutas:
        raise ErrorEscena("No se recibió ningún archivo de escena.")
    metas = [n for n in rutas if Path(n).name == "metadata.json"]
    if len(metas) > 1:
        raise ErrorEscena("Se recibieron varios metadata.json: cargue una sola escena por análisis.")
    if metas:
        escena = _leer_paquete_productor(rutas, metas[0], lim)
        if fecha is not None and interpretar_fecha(fecha).date() != escena.fecha.date():
            escena.avisos.append("La fecha indicada difiere de la del metadato: se conserva la del metadato.")
        return escena
    planos = {Path(n).name.lower(): r for n, r in rutas.items()}
    tifs = [n for n in planos if n.endswith((".tif", ".tiff")) and n not in AUXILIARES]
    imgs = [n for n in planos if n.endswith((".png", ".jpg", ".jpeg"))]
    if len(tifs) + len(imgs) != 1:
        if not tifs and not imgs:
            raise ErrorEscena("No se reconoce ninguna imagen de escena. Formatos admitidos: paquete del generador "
                              "(con metadata.json), GeoTIFF RGB o multibanda, PNG o JPG RGB de 8 bits.")
        raise ErrorEscena("Se recibieron varias imágenes para una misma escena: cargue una sola (más sus "
                          "archivos acompañantes calidad.tif y soporte_png.tif, si los tiene).")
    if tifs:
        escena = _leer_geotiff(planos[tifs[0]], perfil, fecha, mapeo_bandas, escala, desplazamiento, lim)
        _adjuntar_qa(escena, planos)
        return escena
    return _leer_imagen(planos[imgs[0]], fecha, indice, perfil, lim)


def indice_ejemplos(carpeta_ejemplos) -> dict:
    """{sha256 del PNG: {"carpeta", "origen"}} de los ejemplos incluidos en el repositorio."""
    indice = {}
    carpeta_ejemplos = Path(carpeta_ejemplos)
    if not carpeta_ejemplos.is_dir():
        return indice
    for meta in sorted(carpeta_ejemplos.glob("*/metadata.json")):
        try:
            datos = json.loads(meta.read_text(encoding="utf-8"))
            rel = (datos.get("rutas") or {}).get("rgb")
            sha = ((datos.get("archivos") or {}).get(rel) or {}).get("sha256")
        except Exception:
            continue
        if sha:
            indice[sha] = {"carpeta": str(meta.parent), "origen": f"ejemplo {meta.parent.name}"}
    return indice
