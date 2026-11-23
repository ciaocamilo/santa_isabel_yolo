"""Medición planimétrica y comparación de dos fechas (SDD 04 §8).

Área por celda h = |a·e − b·d| / 10000 ha para la afín (a, b, c, d, e, f) de una malla métrica. Las áreas se
calculan con la unión booleana de instancias a resolución de análisis, nunca sobre la vista reescalada.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import numpy as np

from .escenas import Escena, Georef, iso_utc

SIN_OBSERVACION, FONDO, PERSISTENCIA, PERDIDA, GANANCIA = 0, 1, 2, 3, 4
CATEGORIAS = {SIN_OBSERVACION: "sin observación común", FONDO: "fondo observado en ambas fechas",
              PERSISTENCIA: "persistencia", PERDIDA: "pérdida observada", GANANCIA: "ganancia observada"}
COLORES_CAMBIO = {SIN_OBSERVACION: (120, 120, 120, 255), FONDO: (245, 240, 225, 255),
                  PERSISTENCIA: (40, 110, 200, 255), PERDIDA: (215, 40, 40, 255), GANANCIA: (20, 160, 90, 255)}
ETIQUETA_AREA = {"cuantitativa": "área estimada sobre píxeles observables",
                 "aparente_sin_qa": "área aparente, sin control de nubes",
                 "sin_escala": "N/D (sin escala verificable)", "sin_soporte": "N/D (sin soporte válido)",
                 "fuera_de_dominio": "N/D (fuera del dominio del proyecto)"}


@dataclass
class Analisis:
    """Escena con su predicción (y, opcionalmente, su medición individual)."""
    escena: Escena
    prediccion: object
    medicion: object = None


@dataclass
class Medicion:
    estado: str                       # cuantitativa | aparente_sin_qa | sin_escala | sin_soporte | fuera_de_dominio
    etiqueta_area: str
    area_nieve_ha: float | None = None
    area_observable_ha: float | None = None
    area_aoi_ha: float | None = None
    pct_nieve_aoi: float | None = None
    pct_nieve_observable: float | None = None
    pct_observable_aoi: float | None = None
    ha_por_celda: float | None = None
    pixeles_nieve: int = 0
    pixeles_observables: int = 0
    pixeles_aoi: int = 0
    pct_imagen: float | None = None
    motivos: list = field(default_factory=list)
    procedencia: dict = field(default_factory=dict)
    malla: Georef | None = None
    mascara: np.ndarray | None = None      # nieve contabilizada (en `malla`)
    observable: np.ndarray | None = None   # soporte usado como denominador (en `malla`)

    def como_fila(self) -> dict:
        return {"estado": self.estado, "etiqueta_area": self.etiqueta_area, "area_nieve_ha": self.area_nieve_ha,
                "area_observable_ha": self.area_observable_ha, "area_aoi_ha": self.area_aoi_ha,
                "pct_nieve_aoi": self.pct_nieve_aoi, "pct_nieve_observable": self.pct_nieve_observable,
                "pct_observable_aoi": self.pct_observable_aoi, "ha_por_celda": self.ha_por_celda,
                "pixeles_nieve": self.pixeles_nieve, "pixeles_observables": self.pixeles_observables,
                "pixeles_aoi": self.pixeles_aoi, "motivos": " | ".join(self.motivos), **self.procedencia}


@dataclass
class Cambio:
    estado: str                        # ok | no_disponible
    motivos: list = field(default_factory=list)
    avisos: list = field(default_factory=list)
    invertidas: bool = False
    fecha_inicial: object = None
    fecha_final: object = None
    clave_inicial: str = ""
    clave_final: str = ""
    malla: Georef | None = None
    reproyectada: bool = False
    mapa: np.ndarray | None = None     # uint8 con las categorías de CATEGORIAS
    ha_por_celda: float | None = None
    area_inicial_ha: float | None = None        # A0
    area_final_ha: float | None = None          # A1
    perdida_ha: float | None = None             # L
    ganancia_ha: float | None = None            # G
    persistencia_ha: float | None = None        # P
    perdida_pct: float | None = None            # 100·L/A0
    cambio_neto_ha: float | None = None         # Δ = A1 − A0 = G − L
    cambio_neto_pct: float | None = None        # 100·Δ/A0
    area_comun_ha: float | None = None          # área(V)
    area_aoi_ha: float | None = None
    cobertura_comun_pct: float | None = None    # 100·área(V)/área(AOI)
    pct_inicial_en_comun: float | None = None   # 100·A0/área(V)
    pct_final_en_comun: float | None = None     # 100·A1/área(V)
    inicial_fuera_comun_ha: float | None = None
    inicial_fuera_comun_pct: float | None = None
    procedencia: dict = field(default_factory=dict)

    def como_fila(self) -> dict:
        return {"estado": self.estado, "fecha_inicial_utc": iso_utc(self.fecha_inicial),
                "fecha_final_utc": iso_utc(self.fecha_final), "clave_inicial": self.clave_inicial,
                "clave_final": self.clave_final, "fechas_invertidas_en_la_carga": self.invertidas,
                "area_inicial_comparable_ha": self.area_inicial_ha, "area_final_comparable_ha": self.area_final_ha,
                "persistencia_ha": self.persistencia_ha, "perdida_bruta_ha": self.perdida_ha,
                "perdida_pct_del_inicial": self.perdida_pct, "ganancia_ha": self.ganancia_ha,
                "cambio_neto_ha": self.cambio_neto_ha, "cambio_neto_pct_del_inicial": self.cambio_neto_pct,
                "area_comun_observable_ha": self.area_comun_ha, "area_aoi_ha": self.area_aoi_ha,
                "cobertura_comun_pct_del_aoi": self.cobertura_comun_pct,
                "nieve_inicial_pct_del_area_comun": self.pct_inicial_en_comun,
                "nieve_final_pct_del_area_comun": self.pct_final_en_comun,
                "nieve_inicial_fuera_del_area_comun_ha": self.inicial_fuera_comun_ha,
                "nieve_inicial_fuera_del_area_comun_pct": self.inicial_fuera_comun_pct,
                "ha_por_celda": self.ha_por_celda, "malla_crs": self.malla.crs if self.malla else None,
                "reproyectada_vecino_mas_proximo": self.reproyectada,
                "motivos": " | ".join(self.motivos), "avisos": " | ".join(self.avisos), **self.procedencia}


# --- Mallas ---------------------------------------------------------------------------------------------------------
def hectareas_por_celda(afin) -> float:
    a, b, _, d, e, _ = afin
    return abs(a * e - b * d) / 10000.0


def _crs_metrico(cfg):
    return ((cfg or {}).get("aoi") or {}).get("crs") or "EPSG:32618"


def mascara_aoi(georef: Georef, caja_wgs84, paso_grados=0.0005) -> np.ndarray:
    """Celdas de la malla cuyo centro cae dentro de la caja WGS84 (bordes densificados antes de proyectar)."""
    from rasterio.features import geometry_mask
    from rasterio.warp import transform as transformar
    oeste, sur, este, norte = (float(v) for v in caja_wgs84)

    def tramo(p, q):
        n = max(1, int(math.ceil(max(abs(q[0] - p[0]), abs(q[1] - p[1])) / paso_grados)))
        return [(p[0] + (q[0] - p[0]) * k / n, p[1] + (q[1] - p[1]) * k / n) for k in range(n)]

    esquinas = [(oeste, sur), (este, sur), (este, norte), (oeste, norte)]
    anillo = [pt for k in range(4) for pt in tramo(esquinas[k], esquinas[(k + 1) % 4])]
    xs, ys = transformar("EPSG:4326", georef.crs, [p[0] for p in anillo], [p[1] for p in anillo])
    coords = [[float(x), float(y)] for x, y in zip(xs, ys)]
    coords.append(coords[0])
    return geometry_mask([{"type": "Polygon", "coordinates": [coords]}], out_shape=(georef.alto, georef.ancho),
                         transform=georef.affine, invert=True)


def _aoi_en(georef, cfg):
    caja = ((cfg or {}).get("aoi") or {}).get("caja_wgs84")
    if not caja:
        return np.ones((georef.alto, georef.ancho), bool), "sin AOI configurado: toda la malla"
    return mascara_aoi(georef, caja), f"caja WGS84 {list(caja)}"


def reproyectar_mascara(mascara, origen: Georef, destino: Georef) -> np.ndarray:
    """Lleva una máscara booleana a otra malla por vecino más próximo (0 fuera de la huella de origen)."""
    from rasterio.warp import Resampling, reproject
    salida = np.zeros((destino.alto, destino.ancho), np.uint8)
    reproject(source=np.ascontiguousarray(mascara, dtype=np.uint8), destination=salida,
              src_transform=origen.affine, src_crs=origen.crs, src_nodata=None,
              dst_transform=destino.affine, dst_crs=destino.crs, dst_nodata=None,
              resampling=Resampling.nearest, init_dest_nodata=True)
    return salida.astype(bool)


def _resolucion_en(georef: Georef, crs_destino) -> float:
    """Tamaño de celda efectivo (m) de una malla expresado en el CRS de destino."""
    from rasterio.crs import CRS
    from rasterio.warp import calculate_default_transform
    if CRS.from_user_input(georef.crs) == CRS.from_user_input(crs_destino):
        return math.sqrt(abs(georef.afin[0] * georef.afin[4] - georef.afin[1] * georef.afin[3]))
    t, _, _ = calculate_default_transform(georef.crs, crs_destino, georef.ancho, georef.alto, *georef.limites)
    return math.sqrt(abs(t.a * t.e))


def _limites_en(georef: Georef, crs_destino):
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds
    if CRS.from_user_input(georef.crs) == CRS.from_user_input(crs_destino):
        return georef.limites
    return transform_bounds(georef.crs, crs_destino, *georef.limites, densify_pts=21)


def malla_metrica(georef: Georef, crs_destino) -> Georef:
    """Malla norte-arriba en el CRS métrico que cubre la huella de `georef` a su resolución efectiva."""
    res = _resolucion_en(georef, crs_destino)
    oeste, sur, este, norte = _limites_en(georef, crs_destino)
    x0, y1 = math.floor(oeste / res) * res, math.ceil(norte / res) * res
    ancho = max(1, int(math.ceil((este - x0) / res - 1e-9)))
    alto = max(1, int(math.ceil((y1 - sur) / res - 1e-9)))
    return Georef(str(crs_destino), (res, 0.0, x0, 0.0, -res, y1), ancho, alto)


def malla_comun(g_a: Georef, g_b: Georef, cfg=None):
    """Malla de comparación. Devuelve (malla, reproyectada) o (None, False) si las huellas no se intersectan.

    Si ambas escenas comparten una malla métrica se usa tal cual. En otro caso se construye una malla
    norte-arriba en el CRS métrico del proyecto, con la resolución efectiva más gruesa de las dos y alineada
    al origen de la malla de referencia del productor cuando la resolución coincide.
    """
    if g_a.misma_malla(g_b) and g_a.es_metrico:
        return g_a, False
    crs = _crs_metrico(cfg)
    res = max(_resolucion_en(g_a, crs), _resolucion_en(g_b, crs))
    res = round(res, 6)
    la, lb = _limites_en(g_a, crs), _limites_en(g_b, crs)
    oeste, sur = max(la[0], lb[0]), max(la[1], lb[1])
    este, norte = min(la[2], lb[2]), min(la[3], lb[3])
    if este - oeste <= 0 or norte - sur <= 0:
        return None, False
    ref = ((cfg or {}).get("aoi") or {}).get("transform")
    ox, oy = (float(ref[2]), float(ref[5])) if ref and abs(float(ref[0]) - res) < 1e-6 else (0.0, 0.0)
    x0 = ox + math.floor((oeste - ox) / res + 1e-9) * res
    y1 = oy + math.ceil((norte - oy) / res - 1e-9) * res
    ancho = int(math.ceil((este - x0) / res - 1e-9))
    alto = int(math.ceil((y1 - sur) / res - 1e-9))
    if ancho <= 0 or alto <= 0:
        return None, False
    return Georef(str(crs), (res, 0.0, x0, 0.0, -res, y1), ancho, alto), True


def _capas(analisis: Analisis, destino: Georef | None):
    """Nieve, observable (o None) y extensión de un análisis, opcionalmente llevadas a otra malla."""
    esc = analisis.escena
    nieve = np.asarray(analisis.prediccion.mascara, bool)
    if nieve.shape != esc.forma:
        raise ValueError(f"La máscara ({nieve.shape[1]}×{nieve.shape[0]}) no tiene el tamaño de la escena "
                         f"({esc.ancho}×{esc.alto}): deshaga letterbox y reescalado antes de medir.")
    ext = np.ones(esc.forma, bool) if esc.extension is None else np.asarray(esc.extension, bool)
    obs = None if esc.observable is None else np.asarray(esc.observable, bool)
    if destino is None or esc.georef.misma_malla(destino):
        return nieve, obs, ext
    return (reproyectar_mascara(nieve, esc.georef, destino),
            None if obs is None else reproyectar_mascara(obs, esc.georef, destino),
            reproyectar_mascara(ext, esc.georef, destino))


def _intersecta_dominio(georef, cfg) -> bool:
    caja = ((cfg or {}).get("aoi") or {}).get("caja_wgs84")
    if not caja:
        return True
    o, s, e, n = _limites_en(georef, "EPSG:4326")
    return not (e <= caja[0] or o >= caja[2] or n <= caja[1] or s >= caja[3])


# --- Una escena -------------------------------------------------------------------------------------------------------
def medir(escena: Escena, prediccion, cfg=None) -> Medicion:
    """Superficie estimada de nieve/hielo visible de una escena.

    Con QA: área sobre píxeles observables. Sin QA pero con escala: «área aparente, sin control de nubes».
    Sin escala verificable: N/D en hectáreas (solo conteo de píxeles). Sin soporte válido: N/D.
    """
    analisis = Analisis(escena, prediccion)
    nieve0 = np.asarray(prediccion.mascara, bool)
    if nieve0.shape != escena.forma:
        raise ValueError(f"La máscara ({nieve0.shape[1]}×{nieve0.shape[0]}) no tiene el tamaño de la escena "
                         f"({escena.ancho}×{escena.alto}): deshaga letterbox y reescalado antes de medir.")
    procedencia = {"clave": escena.clave, "fecha_utc": iso_utc(escena.fecha), "perfil": escena.perfil,
                   "fuente": escena.fuente, "qa": escena.procedencia_observable,
                   "umbral_operativo": getattr(prediccion, "umbral_operativo", None),
                   "superficie": "planimétrica estimada"}
    total = nieve0.size
    if escena.georef is None or not escena.escala_verificada:
        ext = np.ones(escena.forma, bool) if escena.extension is None else escena.extension
        n = int((nieve0 & ext).sum())
        return Medicion("sin_escala", ETIQUETA_AREA["sin_escala"], pixeles_nieve=n,
                        pct_imagen=100.0 * n / total if total else None, procedencia=procedencia,
                        motivos=["La imagen no tiene escala ni georreferencia verificables: se omiten las "
                                 "hectáreas y solo se informa la fracción de la imagen."],
                        mascara=nieve0 & ext, observable=None)
    if not _intersecta_dominio(escena.georef, cfg):
        return Medicion("fuera_de_dominio", ETIQUETA_AREA["fuera_de_dominio"], procedencia=procedencia,
                        motivos=["La escena no intersecta el AOI del Nevado Santa Isabel: el análisis "
                                 "cuantitativo del MVP se restringe al dominio del proyecto."])
    malla = escena.georef
    motivos = []
    if not malla.es_metrico:
        malla = malla_metrica(escena.georef, _crs_metrico(cfg))
        motivos.append(f"La escena venía en {escena.georef.crs}: se reproyectó a {malla.crs} por vecino más "
                       f"próximo (celda {malla.afin[0]:.2f} m) antes de medir.")
        nieve, obs, ext = _capas(analisis, malla)
    else:
        nieve, obs, ext = _capas(analisis, None)
    aoi, texto_aoi = _aoi_en(malla, cfg)
    procedencia["aoi"] = texto_aoi
    h = malla.ha_por_celda
    area_aoi = h * int(aoi.sum())
    if obs is None:
        estado, soporte = "aparente_sin_qa", ext & aoi
        motivos.append("Sin QA: se mide sobre la extensión con datos; nubes y sombras pueden ocultar o simular "
                       "cobertura.")
    else:
        estado, soporte = "cuantitativa", obs & ext & aoi
    n_sop = int(soporte.sum())
    if n_sop == 0:
        return Medicion("sin_soporte", ETIQUETA_AREA["sin_soporte"], area_aoi_ha=area_aoi, ha_por_celda=h,
                        pixeles_aoi=int(aoi.sum()), procedencia=procedencia, malla=malla,
                        motivos=motivos + ["La escena no tiene ningún píxel observable dentro del AOI."],
                        mascara=np.zeros_like(soporte), observable=soporte)
    contada = nieve & soporte
    n = int(contada.sum())
    if n == 0:
        motivos.append("Sin detecciones sobre el soporte observable: 0 ha estimadas.")
    return Medicion(estado, ETIQUETA_AREA[estado], area_nieve_ha=h * n, area_observable_ha=h * n_sop,
                    area_aoi_ha=area_aoi, pct_nieve_aoi=100.0 * h * n / area_aoi if area_aoi else None,
                    pct_nieve_observable=100.0 * n / n_sop, pct_observable_aoi=100.0 * h * n_sop / area_aoi
                    if area_aoi else None, ha_por_celda=h, pixeles_nieve=n, pixeles_observables=n_sop,
                    pixeles_aoi=int(aoi.sum()), pct_imagen=100.0 * n / total, motivos=motivos,
                    procedencia=procedencia, malla=malla, mascara=contada, observable=soporte)


# --- Dos fechas -------------------------------------------------------------------------------------------------------
def _no_disponible(motivo, **datos) -> Cambio:
    return Cambio("no_disponible", motivos=[motivo], **datos)


def comparar(anterior: Analisis, posterior: Analisis, config=None) -> Cambio:
    """Cambio de cobertura entre dos fechas sobre el soporte común V.

    V = AOI ∩ huellas ∩ soporte ∩ píxeles observables de ambas fechas. A y B son las máscaras inicial y final
    restringidas a V. Se informa exclusivamente sobre V: no se extrapola a la zona oculta.
    """
    a, b = anterior, posterior
    fa, fb = a.escena.fecha, b.escena.fecha
    if fa is None or fb is None:
        falta = " y ".join(x.escena.clave for x in (a, b) if x.escena.fecha is None)
        return _no_disponible(f"Falta la fecha de adquisición de {falta}: indíquela para comparar.")
    if fa.date() == fb.date():
        return _no_disponible(f"Las dos escenas son de la misma fecha ({fa:%Y-%m-%d}): cargue dos adquisiciones "
                              "distintas.", fecha_inicial=fa, fecha_final=fb)
    invertidas = fa > fb
    if invertidas:
        a, b, fa, fb = b, a, fb, fa
    base = dict(invertidas=invertidas, fecha_inicial=fa, fecha_final=fb, clave_inicial=a.escena.clave,
                clave_final=b.escena.clave)
    avisos = []
    if invertidas:
        avisos.append("Las escenas se cargaron en orden inverso: se reordenaron cronológicamente.")
    for x in (a, b):
        if x.escena.georef is None or not x.escena.escala_verificada:
            return _no_disponible(
                f"{x.escena.clave} no tiene georreferencia verificable: sin ella no hay comparación espacial "
                "(dos capturas de igual tamaño no demuestran coincidencia espacial).", **base)
    for x in (a, b):
        if x.escena.observable is None:
            return _no_disponible(
                f"{x.escena.clave} no tiene QA ni máscara de observabilidad documentada: se exige para "
                "cuantificar la pérdida entre fechas.", **base)
    for x in (a, b):
        if not _intersecta_dominio(x.escena.georef, config):
            return _no_disponible(f"{x.escena.clave} no intersecta el AOI del Nevado Santa Isabel.", **base)
    malla, reproyectada = malla_comun(a.escena.georef, b.escena.georef, config)
    if malla is None:
        return _no_disponible("Las huellas de las dos escenas no se intersectan: no hay soporte común.", **base)
    nieve_a, obs_a, ext_a = _capas(a, malla)
    nieve_b, obs_b, ext_b = _capas(b, malla)
    aoi, texto_aoi = _aoi_en(malla, config)
    if reproyectada:
        avisos.append(f"Las mallas no coincidían: máscaras y QA se llevaron por vecino más próximo a una malla "
                      f"común de {malla.afin[0]:g} m en {malla.crs}. Puede quedar desregistro real: inspeccione "
                      "rasgos estables y descarte el par si hay desplazamiento visible.")
    if a.escena.perfil and b.escena.perfil and a.escena.perfil != b.escena.perfil:
        avisos.append(f"Niveles de procesamiento distintos ({a.escena.perfil} y {b.escena.perfil}): la diferencia "
                      "radiométrica limita la interpretación del cambio.")
    tolerancia = int(((config or {}).get("comparacion") or {}).get("meses_tolerancia", 1))
    dif = abs(fa.month - fb.month)
    dif = min(dif, 12 - dif)
    if dif > tolerancia:
        avisos.append(f"Épocas del año distintas (mes {fa.month} y mes {fb.month}): parte del cambio puede ser "
                      "nieve estacional.")
    v = aoi & ext_a & ext_b & obs_a & obs_b
    h = malla.ha_por_celda
    propia = nieve_a & aoi & ext_a & obs_a
    n_propia = int(propia.sum())
    n_fuera = int((propia & ~v).sum())
    area_aoi = h * int(aoi.sum())
    procedencia = {"aoi": texto_aoi, "qa_inicial": a.escena.procedencia_observable,
                   "qa_final": b.escena.procedencia_observable, "perfil_inicial": a.escena.perfil,
                   "perfil_final": b.escena.perfil,
                   "umbral_operativo": getattr(a.prediccion, "umbral_operativo", None),
                   "definicion_soporte": "V = AOI ∩ huellas ∩ soporte ∩ observables de ambas fechas",
                   "superficie": "planimétrica estimada"}
    comunes = dict(base, avisos=avisos, malla=malla, reproyectada=reproyectada, ha_por_celda=h,
                   area_aoi_ha=area_aoi, procedencia=procedencia,
                   inicial_fuera_comun_ha=h * n_fuera if n_propia else None,
                   inicial_fuera_comun_pct=100.0 * n_fuera / n_propia if n_propia else None)
    mapa = np.full(v.shape, SIN_OBSERVACION, np.uint8)
    n_v = int(v.sum())
    if n_v == 0:
        return Cambio("no_disponible", motivos=["El soporte común V está vacío (sin píxeles observables en ambas "
                                                "fechas dentro del AOI): todas las áreas comparativas son N/D."],
                      mapa=mapa, area_comun_ha=0.0, cobertura_comun_pct=0.0 if area_aoi else None, **comunes)
    ma, mb = nieve_a & v, nieve_b & v
    mapa[v] = FONDO
    mapa[ma & mb] = PERSISTENCIA
    mapa[ma & ~mb] = PERDIDA
    mapa[mb & ~ma] = GANANCIA
    a0, a1 = h * int(ma.sum()), h * int(mb.sum())
    perdida, ganancia = h * int((ma & ~mb).sum()), h * int((mb & ~ma).sum())
    persistencia = h * int((ma & mb).sum())
    delta = a1 - a0
    motivos = []
    if int(ma.sum()) == 0:
        motivos.append("A₀ = 0 ha en el soporte común: los porcentajes respecto al área inicial son N/D.")
    if n_fuera:
        avisos.append(f"{h * n_fuera:.2f} ha de la nieve inicial observable ({100.0 * n_fuera / n_propia:.1f} %) "
                      "quedaron fuera del soporte común por falta de datos en la otra fecha.")
    return Cambio("ok", motivos=motivos, mapa=mapa, area_inicial_ha=a0, area_final_ha=a1, perdida_ha=perdida,
                  ganancia_ha=ganancia, persistencia_ha=persistencia,
                  perdida_pct=100.0 * perdida / a0 if a0 > 0 else None, cambio_neto_ha=delta,
                  cambio_neto_pct=100.0 * delta / a0 if a0 > 0 else None, area_comun_ha=h * n_v,
                  cobertura_comun_pct=100.0 * n_v / int(aoi.sum()) if aoi.any() else None,
                  pct_inicial_en_comun=100.0 * int(ma.sum()) / n_v, pct_final_en_comun=100.0 * int(mb.sum()) / n_v,
                  **comunes)


# --- Exportación georreferenciada --------------------------------------------------------------------------------------
def _geotiff_uint8(datos, georef: Georef, colores, etiquetas) -> bytes:
    import rasterio
    from rasterio.io import MemoryFile
    perfil = dict(driver="GTiff", width=georef.ancho, height=georef.alto, count=1, dtype="uint8", crs=georef.crs,
                  transform=georef.affine, compress="deflate", nodata=None)
    with MemoryFile() as mem:
        with mem.open(**perfil) as ds:
            ds.write(np.ascontiguousarray(datos, dtype=np.uint8), 1)
            ds.write_colormap(1, colores)
            ds.update_tags(**{k: str(v) for k, v in etiquetas.items() if v is not None})
        return mem.read()


def exportar_cambio_geotiff(cambio: Cambio) -> bytes:
    """GeoTIFF categórico (0–4) con tabla de color y leyenda en las etiquetas del archivo."""
    if cambio.mapa is None or cambio.malla is None:
        raise ValueError("No hay mapa de cambio georreferenciado que exportar.")
    leyenda = "; ".join(f"{k} = {v}" for k, v in CATEGORIAS.items())
    return _geotiff_uint8(cambio.mapa, cambio.malla, COLORES_CAMBIO,
                          {"leyenda": leyenda, "fecha_inicial_utc": iso_utc(cambio.fecha_inicial),
                           "fecha_final_utc": iso_utc(cambio.fecha_final), "ha_por_celda": cambio.ha_por_celda,
                           "nota": "Pérdida observada de cobertura de nieve/hielo visible sobre el soporte común; "
                                   "no es balance de masa glaciar."})


def exportar_mascara_geotiff(medicion: Medicion, escena: Escena) -> bytes:
    """GeoTIFF de una escena: 0 = no observable, 1 = fondo observado, 2 = nieve/hielo visible."""
    if medicion.malla is None or medicion.mascara is None:
        raise ValueError("La escena no tiene georreferencia: no hay máscara georreferenciada que exportar.")
    datos = np.zeros(medicion.mascara.shape, np.uint8)
    if medicion.observable is not None:
        datos[medicion.observable] = 1
    datos[medicion.mascara] = 2
    return _geotiff_uint8(datos, medicion.malla, {0: (120, 120, 120, 255), 1: (245, 240, 225, 255),
                                                  2: (0, 200, 255, 255)},
                          {"leyenda": "0 = no observable; 1 = fondo observado; 2 = nieve/hielo visible",
                           "clave": escena.clave, "adquisicion_utc": iso_utc(escena.fecha),
                           "estado_medicion": medicion.estado, "ha_por_celda": medicion.ha_por_celda})


def filas_a_csv(filas) -> bytes:
    """CSV UTF-8 con BOM (se abre bien en Excel) a partir de una lista de diccionarios."""
    import csv
    filas = list(filas)
    campos = list(dict.fromkeys(k for f in filas for k in f))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=campos, lineterminator="\n")
    w.writeheader()
    for f in filas:
        w.writerow({k: ("" if v is None else v) for k, v in f.items()})
    return ("\ufeff" + buf.getvalue()).encode("utf-8")
