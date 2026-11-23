"""Contrato con el generador mensual (SDD 03): auditoría de una versión publicada, YAML locales y caché.

El productor conserva la propiedad de escenas y particiones: aquí solo se lee. Los `data.yaml` originales traen
un `path:` absoluto del equipo que generó los datos; se generan copias locales en la carpeta de corrida
sustituyendo únicamente esa línea.
"""
from __future__ import annotations

import csv
import json
import re
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from .config import resolver
from .escenas import sha256_archivo
from .evaluacion import area_poligono_px, leer_txt_yolo, rasterizar_poligonos

SPLITS = ("train", "val", "test")
MARCADOR_VARIANTE = "<variante>"
# Encabezados reales del manifiesto v0003 y su equivalencia con los conceptos del SDD 04.
EQUIVALENCIAS = {
    "mes": "mes planificado del calendario (AAAA-MM); decide el split",
    "split": "partición del generador (train/val/test)",
    "nombre": "nombre base de imagen y etiqueta = <mes>_<clave_escena>",
    "perfil": "nivel de procesamiento: L1C o L2A",
    "clave_escena": "clave de la escena (fecha + tesela + system:index, sufijo _l1c en L1C); resuelve la caché",
    "id_asset": "asset de Earth Engine",
    "adquisicion_utc": "fecha y hora UTC de adquisición",
    "grupo_temporal": "grupo AOI+fecha: una adquisición y sus variantes no pueden cruzar splits",
    "decision": "estado de la etiqueta: positiva (polígonos) o negativa (TXT vacío)",
    "instancias": "número de polígonos de la clase 0",
    "area_pseudoetiquetada_m2": "área de los polígonos en m²",
    "origen": "procedencia de la referencia (automatico_ndsi)",
    "revision_humana": "si hubo revisión humana registrada",
    "provisional": "mes en curso, puede cambiar",
    "seleccion_incompleta": "un candidato mejor falló por error técnico",
    "imagen_rgb": "ruta relativa a la versión del PNG RGB (B4/B3/B2, 8 bits)",
    "imagen_espectral": "ruta relativa a la versión del PNG espectral (NDSI/NDVI/B4)",
    "etiqueta": f"ruta relativa del TXT con el marcador literal {MARCADOR_VARIANTE} (rgb o espectral)",
    "sha256_rgb": "hash del PNG RGB", "sha256_espectral": "hash del PNG espectral",
    "sha256_txt": "hash del TXT (idéntico en ambas variantes)",
    "receta_huella": "huella de la receta de productos; subcarpeta de la escena en la caché",
    "reglas_huella": "huella de las reglas de pseudoetiquetado; subcarpeta etiquetas/ en la caché",
}
OBLIGATORIAS = ("mes", "split", "nombre", "perfil", "clave_escena", "id_asset", "adquisicion_utc", "grupo_temporal",
                "decision", "instancias", "origen", "revision_humana", "imagen_rgb", "imagen_espectral", "etiqueta",
                "sha256_rgb", "sha256_espectral", "sha256_txt")


@dataclass
class Auditoria:
    version: str
    carpeta: Path
    filas: list = field(default_factory=list)
    bloqueos: list = field(default_factory=list)
    avisos: list = field(default_factory=list)
    encabezados: list = field(default_factory=list)
    resumen: dict = field(default_factory=dict)
    configuracion: dict = field(default_factory=dict)
    instancias: list = field(default_factory=list)   # una fila por polígono (área y caja en px)
    meses: dict = field(default_factory=dict)        # planificados y ausentes por split
    hashes_verificados: int = 0

    @property
    def ok(self) -> bool:
        return not self.bloqueos

    def tabla_splits(self) -> list:
        """Escenas útiles reales por split: positivas, negativas, instancias, años, perfiles y meses ausentes."""
        tabla = []
        for s in SPLITS:
            fs = [f for f in self.filas if f["split"] == s]
            perfiles = Counter(f["perfil"] for f in fs)
            tabla.append({
                "split": s, "meses_planificados": self.meses.get(s, {}).get("planificados"),
                "escenas": len(fs), "positivas": sum(f["decision"] == "positiva" for f in fs),
                "negativas": sum(f["decision"] == "negativa" for f in fs),
                "instancias": sum(int(f["instancias"]) for f in fs),
                "L1C": perfiles.get("L1C", 0), "L2A": perfiles.get("L2A", 0),
                "años": ", ".join(sorted({f["mes"][:4] for f in fs})),
                "meses_ausentes": len(self.meses.get(s, {}).get("ausentes", [])),
                "revision_humana": sum(_booleano(f["revision_humana"]) for f in fs)})
        return tabla

    def identidad(self) -> dict:
        """Identidad exacta del dataset para el contrato del paquete y la bitácora."""
        return {"version": self.version, "huella_contenido": self.resumen.get("huella_contenido"),
                "id_ejecucion": self.resumen.get("id_ejecucion"), "id_aoi": self.resumen.get("id_aoi"),
                "creado_utc": self.resumen.get("creado_utc"),
                "sha256_manifiesto": sha256_archivo(self.carpeta / "manifiesto.csv"),
                "escenas": {s: sum(f["split"] == s for f in self.filas) for s in SPLITS},
                "instancias": {s: sum(int(f["instancias"]) for f in self.filas if f["split"] == s) for s in SPLITS},
                "origen_referencia": sorted({f["origen"] for f in self.filas}),
                "revision_humana": any(_booleano(f["revision_humana"]) for f in self.filas)}


def _booleano(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "sí", "si")


def carpeta_version(cfg, version=None) -> Path:
    return resolver(cfg, cfg["dataset"]["raiz"]) / (version or cfg["dataset"]["version"])


def leer_manifiesto(ruta):
    with open(ruta, newline="", encoding="utf-8-sig") as f:
        lector = csv.DictReader(f)
        return list(lector.fieldnames or []), [dict(r) for r in lector]


def ruta_etiqueta(fila, variante) -> str:
    """Sustituye el marcador literal <variante> de la columna `etiqueta`."""
    return fila["etiqueta"].replace(MARCADOR_VARIANTE, variante)


def _meses_plan(cortes):
    ini, fin, n = cortes["meses_plan"]
    a, m = int(ini[:4]), int(ini[5:7])
    meses = []
    while True:
        meses.append(f"{a:04d}-{m:02d}")
        if meses[-1] == fin:
            break
        m += 1
        if m == 13:
            a, m = a + 1, 1
        if len(meses) > 5000:
            raise ValueError("meses_plan inconsistente")
    if len(meses) != int(n):
        raise ValueError(f"meses_plan declara {n} meses y el intervalo {ini}..{fin} tiene {len(meses)}")
    return meses


def split_de_mes(mes, cortes) -> str:
    if cortes.get("corte_val") is None or mes < cortes["corte_val"]:
        return "train"
    if cortes.get("corte_test") is None or mes < cortes["corte_test"]:
        return "val"
    return "test"


def auditar_dataset(cfg, version=None, verificar_hashes=True) -> Auditoria:
    """Valida una versión publicada. Cualquier elemento de `bloqueos` impide entrenar."""
    carpeta = carpeta_version(cfg, version)
    version = carpeta.name
    aud = Auditoria(version=version, carpeta=carpeta)
    blo, avi = aud.bloqueos, aud.avisos
    variantes = tuple(cfg["dataset"].get("variantes", ("rgb", "espectral")))
    if not carpeta.is_dir():
        blo.append(f"No existe la carpeta de la versión: {carpeta}. Extraiga el ZIP del generador en "
                   f"{carpeta.parent}.")
        return aud
    faltan = [n for n in ("manifiesto.csv", "configuracion.toml", "resumen.json",
                          *(f"{v}/data.yaml" for v in variantes)) if not (carpeta / n).is_file()]
    if faltan:
        blo += [f"Falta el archivo {carpeta / n}" for n in faltan]
        return aud
    aud.resumen = json.loads((carpeta / "resumen.json").read_text(encoding="utf-8"))
    with open(carpeta / "configuracion.toml", "rb") as f:
        aud.configuracion = tomllib.load(f)
    aud.encabezados, filas = leer_manifiesto(carpeta / "manifiesto.csv")
    falt_col = [c for c in OBLIGATORIAS if c not in aud.encabezados]
    if falt_col:
        blo.append(f"manifiesto.csv: faltan las columnas {', '.join(falt_col)}")
        return aud
    desconocidas = [c for c in aud.encabezados if c not in EQUIVALENCIAS]
    if desconocidas:
        avi.append(f"manifiesto.csv: columnas sin equivalencia documentada: {', '.join(desconocidas)}")
    res = aud.resumen
    if res.get("version") != version:
        blo.append(f"resumen.json declara la versión {res.get('version')!r} y la carpeta es {version!r}")
    clase_id, clase_nombre = int(cfg["dataset"]["clase_id"]), cfg["dataset"]["clase_nombre"]
    if {int(k): v for k, v in (res.get("clases") or {}).items()} != {clase_id: clase_nombre}:
        blo.append(f"resumen.json: clases {res.get('clases')} ≠ {{{clase_id}: {clase_nombre!r}}}")
    for v in variantes:
        estado = (res.get("estado_producto") or {}).get(v)
        if estado == "no_entrenable":
            blo.append(f"resumen.json: la variante {v} está marcada no_entrenable")
        elif estado == "entrenable_sin_prueba":
            avi.append(f"Variante {v}: entrenable sin prueba; las métricas finales quedan pendientes.")
        texto = (carpeta / v / "data.yaml").read_text(encoding="utf-8")
        if f"0: {clase_nombre}" not in texto:
            blo.append(f"{v}/data.yaml: no declara la clase 0 = {clase_nombre}")
        if not all(re.search(rf"^{s}:\s*images/{s}\s*$", texto, re.M) for s in SPLITS):
            blo.append(f"{v}/data.yaml: los splits no apuntan a images/{{train,val,test}}")
    ancho, alto = int(cfg["dataset"]["ancho"]), int(cfg["dataset"]["alto"])
    cortes = res.get("cortes") or {}
    nombres, claves, grupos = Counter(), Counter(), defaultdict(set)
    fechas = defaultdict(list)
    for n, f in enumerate(filas, 2):
        donde = f"manifiesto.csv fila {n} ({f.get('nombre', '?')})"
        nombres[f["nombre"]] += 1
        claves[f["clave_escena"]] += 1
        grupos[f["grupo_temporal"]].add(f["split"])
        if f["split"] not in SPLITS:
            blo.append(f"{donde}: split {f['split']!r} desconocido")
            continue
        if f["perfil"] not in ("L1C", "L2A"):
            blo.append(f"{donde}: perfil {f['perfil']!r} no es L1C ni L2A")
        es_sr = "S2_SR_HARMONIZED" in f["id_asset"]
        if (f["perfil"] == "L2A") != es_sr or (f["perfil"] == "L1C") != f["clave_escena"].endswith("_l1c"):
            blo.append(f"{donde}: perfil {f['perfil']} incoherente con el asset o la clave")
        try:
            fecha = datetime.strptime(f["adquisicion_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            f["_fecha"] = fecha
            fechas[fecha.date()].append(f["nombre"])
            if f"{fecha:%Y-%m}" != f["mes"] or not f["clave_escena"].startswith(f"{fecha:%Y%m%d}_"):
                blo.append(f"{donde}: la fecha {f['adquisicion_utc']} no concuerda con el mes o la clave")
        except ValueError:
            blo.append(f"{donde}: adquisicion_utc {f['adquisicion_utc']!r} no es una fecha UTC ISO (…Z)")
        if f["nombre"] != f"{f['mes']}_{f['clave_escena']}":
            blo.append(f"{donde}: nombre ≠ <mes>_<clave_escena>")
        if cortes and split_de_mes(f["mes"], cortes) != f["split"]:
            blo.append(f"{donde}: el split {f['split']} no respeta los cortes del generador "
                       f"({cortes.get('corte_val')}, {cortes.get('corte_test')})")
        if f["decision"] not in ("positiva", "negativa"):
            blo.append(f"{donde}: decision {f['decision']!r} no exportable")
        if _booleano(f.get("provisional", "False")):
            avi.append(f"{f['nombre']}: escena provisional (mes en curso)")
        if _booleano(f.get("seleccion_incompleta", "False")):
            avi.append(f"{f['nombre']}: selección incompleta (un candidato mejor falló por error técnico)")
        if MARCADOR_VARIANTE not in f["etiqueta"]:
            blo.append(f"{donde}: la columna etiqueta no contiene el marcador {MARCADOR_VARIANTE}")
        huellas_txt = set()
        instancias = None
        for v in variantes:
            img = carpeta / f[f"imagen_{v}"]
            txt = carpeta / ruta_etiqueta(f, v)
            f[f"_imagen_{v}"], f[f"_etiqueta_{v}"] = img, txt
            if f"/{f['split']}/" not in f[f"imagen_{v}"] or f"/{f['split']}/" not in f["etiqueta"]:
                blo.append(f"{donde}: la ruta de {v} no está en la carpeta del split {f['split']}")
            if not img.is_file():
                blo.append(f"{donde}: falta la imagen {img}")
            else:
                try:
                    with Image.open(img) as im:
                        im.load()
                        if im.size != (ancho, alto) or im.mode != "RGB":
                            blo.append(f"{donde}: {v} mide {im.size} modo {im.mode}; se esperaba "
                                       f"({ancho}, {alto}) RGB de 8 bits")
                except Exception as e:
                    blo.append(f"{donde}: imagen {v} corrupta ({e})")
                if verificar_hashes:
                    if sha256_archivo(img) != f[f"sha256_{v}"]:
                        blo.append(f"{donde}: el hash de la imagen {v} no coincide con el manifiesto")
                    else:
                        aud.hashes_verificados += 1
            if not txt.is_file():
                blo.append(f"{donde}: falta la etiqueta {txt} (un negativo debe ser un TXT vacío explícito)")
                continue
            if verificar_hashes:
                if sha256_archivo(txt) != f["sha256_txt"]:
                    blo.append(f"{donde}: el hash de la etiqueta {v} no coincide con el manifiesto")
                else:
                    aud.hashes_verificados += 1
            huellas_txt.add(txt.read_bytes())
            try:
                instancias = leer_txt_yolo(txt, clases_validas=(clase_id,))
            except ValueError as e:
                blo.append(f"{donde}: {e}")
                instancias = None
        if len(huellas_txt) > 1:
            blo.append(f"{donde}: las etiquetas difieren entre variantes")
        if instancias is not None:
            f["_poligonos"] = [v for _, v in instancias]
            if len(instancias) != int(f["instancias"]):
                blo.append(f"{donde}: {len(instancias)} instancias en el TXT y {f['instancias']} en el manifiesto")
            if (f["decision"] == "negativa") != (len(instancias) == 0):
                blo.append(f"{donde}: decisión {f['decision']} incoherente con el TXT")
            for k, (_, v) in enumerate(instancias):
                px = v * [ancho, alto]
                aud.instancias.append({"nombre": f["nombre"], "split": f["split"], "perfil": f["perfil"],
                                       "instancia": k, "area_px": area_poligono_px(v, ancho, alto),
                                       "ancho_px": float(np.ptp(px[:, 0])), "alto_px": float(np.ptp(px[:, 1])),
                                       "vertices": len(v)})
    blo += [f"nombre duplicado en el manifiesto: {n}" for n, k in nombres.items() if k > 1]
    blo += [f"clave de escena duplicada: {n}" for n, k in claves.items() if k > 1]
    blo += [f"fecha duplicada {d}: {', '.join(ns)}" for d, ns in fechas.items() if len(ns) > 1]
    blo += [f"fuga temporal: {g} aparece en varios splits {sorted(s)}" for g, s in grupos.items() if len(s) > 1]
    esperados = {(f["split"], f["nombre"]) for f in filas}
    for v in variantes:
        for s in SPLITS:
            imgs = {p.stem for p in (carpeta / v / "images" / s).glob("*") if p.is_file()}
            txts = {p.stem for p in (carpeta / v / "labels" / s).glob("*.txt")}
            esperado = {n for sp, n in esperados if sp == s}
            if imgs - esperado:
                blo.append(f"{v}/images/{s}: archivos que no están en el manifiesto: {sorted(imgs - esperado)[:3]}")
            if txts - esperado:
                blo.append(f"{v}/labels/{s}: etiquetas huérfanas: {sorted(txts - esperado)[:3]}")
    conteos = res.get("conteos") or {}
    for s in SPLITS:
        fs = [f for f in filas if f["split"] == s]
        real = {"imagenes": len(fs), "positivas": sum(f["decision"] == "positiva" for f in fs),
                "negativas": sum(f["decision"] == "negativa" for f in fs),
                "instancias": sum(int(f["instancias"]) for f in fs)}
        declarado = {k: (conteos.get(s) or {}).get(k) for k in real}
        if conteos and real != declarado:
            blo.append(f"resumen.json: conteos de {s} {declarado} ≠ manifiesto {real}")
    if cortes.get("meses_plan"):
        plan = _meses_plan(cortes)
        for s in SPLITS:
            del_split = [m for m in plan if split_de_mes(m, cortes) == s]
            presentes = {f["mes"] for f in filas if f["split"] == s}
            aud.meses[s] = {"planificados": len(del_split), "ausentes": [m for m in del_split if m not in presentes]}
    aud.filas = filas
    # Avisos de alcance (no bloquean, pero deben mostrarse junto a los resultados).
    por_split = {s: [f for f in filas if f["split"] == s] for s in SPLITS}
    if not por_split["train"] or not any(f["decision"] == "positiva" for f in por_split["train"]):
        blo.append("train no tiene escenas positivas: solo pueden ejecutarse auditoría y fixtures")
    if not por_split["val"]:
        blo.append("val no tiene escenas: no puede seleccionarse checkpoint ni umbral")
    if not por_split["test"]:
        avi.append("No hay escenas de test: el entrenamiento y la demostración serían provisionales.")
    n_val = sum(int(f["instancias"]) for f in por_split["val"])
    if por_split["val"] and (len(por_split["val"]) < 10 or n_val < 50):
        avi.append(f"Soporte de validación pequeño: {len(por_split['val'])} escena(s) y {n_val} instancia(s); "
                   "la selección de checkpoint y umbral es inestable.")
    if filas and not any(f["decision"] == "negativa" for f in filas):
        avi.append("No hay escenas totalmente negativas: no se mide el comportamiento sin nieve.")
    if filas and not any(_booleano(f["revision_humana"]) for f in filas):
        avi.append("Ninguna escena tiene revisión humana registrada: la referencia es automática "
                   f"({', '.join(sorted({f['origen'] for f in filas}))}); las métricas son concordancia con "
                   "pseudoetiquetas.")
    etiquetas_cfg = aud.configuracion.get("etiquetas") or {}
    if any(isinstance(v, dict) and v.get("ambiguos_bloquean") is False for v in etiquetas_cfg.values()):
        avi.append("Configuración efectiva del productor: ambiguos_bloquean = false. Los componentes NDSI "
                   "ambiguos (p. ej. en contacto con nubes) quedan como fondo en las etiquetas sin bloquear la "
                   "escena.")
    if res.get("aviso"):
        avi.append(f"Aviso del productor: {res['aviso']}")
    return aud


def yaml_locales(cfg, carpeta_corrida, version=None) -> dict:
    """Copias locales de los data.yaml en la carpeta de corrida: solo cambia `path:` por la ruta absoluta
    resuelta de cada variante extraída. Los originales no se editan."""
    carpeta = carpeta_version(cfg, version)
    destino = Path(carpeta_corrida) / "data_yaml"
    destino.mkdir(parents=True, exist_ok=True)
    salida = {}
    for v in cfg["dataset"].get("variantes", ("rgb", "espectral")):
        lineas = (carpeta / v / "data.yaml").read_text(encoding="utf-8").splitlines()
        indices = [k for k, linea in enumerate(lineas) if re.match(r"^path\s*:", linea)]
        if len(indices) != 1:
            raise ValueError(f"{v}/data.yaml: se esperaba exactamente una línea «path:» y hay {len(indices)}")
        raiz_variante = (carpeta / v).resolve().as_posix()
        lineas[indices[0]] = "path: " + json.dumps(raiz_variante, ensure_ascii=False)
        ruta = destino / f"{carpeta.name}_{v}.yaml"
        ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
        salida[v] = ruta
    return salida


# --- Caché del productor -------------------------------------------------------------------------------------------
def raiz_cache(cfg) -> Path | None:
    valor = (cfg.get("dataset") or {}).get("raiz_cache")
    if not valor:
        return None
    p = resolver(cfg, valor)
    return p if p.is_dir() else None


def resolver_cache(cfg, fila, id_aoi=None) -> Path | None:
    """Carpeta de la escena en la caché del productor: <raiz>/<id_aoi>/<perfil>/<clave>/<receta>/.

    El manifiesto no referencia la caché: se resuelve por clave y receta exigiendo coincidencia única.
    """
    raiz = raiz_cache(cfg)
    if raiz is None:
        return None
    clave, receta = fila["clave_escena"], fila.get("receta_huella") or "*"
    patron = f"{id_aoi or '*'}/{fila.get('perfil') or '*'}/{clave}/{receta}/metadata.json"
    hallados = [p.parent for p in raiz.glob(patron)]
    if len(hallados) != 1:
        return None
    return hallados[0]


def mascara_referencia(fila, ancho, alto) -> np.ndarray:
    """Unión de los polígonos de la escena rasterizada a la malla del PNG."""
    return rasterizar_poligonos(fila["_poligonos"], ancho, alto)


def capas_cache(carpeta_cache, reglas_huella=None) -> dict:
    """QA y soporte de una escena de la caché: observable (calidad = 2 y soporte), soporte, calidad, NDSI y,
    si existe la decisión de etiquetado, la máscara de componentes ambiguos (referencia incierta)."""
    import rasterio
    carpeta_cache = Path(carpeta_cache)
    meta = json.loads((carpeta_cache / "metadata.json").read_text(encoding="utf-8"))
    with rasterio.open(carpeta_cache / "calidad.tif") as ds:
        calidad = ds.read(1)
    with rasterio.open(carpeta_cache / "soporte_png.tif") as ds:
        soporte = ds.read(1) == 1
    with rasterio.open(carpeta_cache / "indices.tif") as ds:
        ndsi = ds.read(1)
        ndsi = np.where((ndsi == ds.nodata) | ~np.isfinite(ndsi), np.nan, ndsi)
    alto, ancho = calidad.shape
    capas = {"meta": meta, "calidad": calidad, "soporte": soporte, "observable": (calidad == 2) & soporte,
             "ndsi": ndsi, "ambiguos": None, "decision": None}
    carpetas = sorted((carpeta_cache / "etiquetas").glob(f"{reglas_huella or '*'}/decision.json"))
    if len(carpetas) == 1:
        decision = json.loads(carpetas[0].read_text(encoding="utf-8"))
        vertices = [c["vertices"] for c in decision.get("componentes", [])
                    if c.get("decision") == "ambiguo" and c.get("vertices")]
        capas["ambiguos"] = rasterizar_poligonos(vertices, ancho, alto, normalizados=False)
        capas["decision"] = {k: decision.get(k) for k in ("decision", "motivo", "aceptados", "ambiguos", "agua",
                                                         "reglas_disparadas", "visibilidad_sectores_pct")}
        capas["umbral_ndsi"] = (decision.get("reglas") or {}).get("ndsi_min")
    return capas
