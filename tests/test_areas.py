"""Hito C · Geometría: ejemplo numérico de aceptación y casos límite del comparador (SDD 04 §8)."""
import math

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import Resampling, reproject

from conftest import analisis, escena, georef, prediccion_de
from santa_isabel import areas
from santa_isabel.areas import Analisis, comparar, hectareas_por_celda, medir
from santa_isabel.escenas import Georef
from santa_isabel.inferencia import (cajas_a_original, deshacer_letterbox, geometria_letterbox, letterbox,
                                     mapa_puntuacion)

CFG = {"aoi": {"crs": "EPSG:32618"}, "comparacion": {"meses_tolerancia": 1}}


def mascaras_ejemplo(alto=40, ancho=50):
    """A: 100 px. B: 80 px. Persisten 70 → pérdida 30, ganancia 10."""
    a = np.zeros((alto, ancho), bool)
    b = np.zeros((alto, ancho), bool)
    a[5:15, 5:15] = True                # 100 px
    b[5:15, 5:12] = True                # 70 px que persisten
    b[30:32, 20:25] = True              # 10 px de ganancia en otro lugar
    assert (a.sum(), b.sum(), (a & b).sum()) == (100, 80, 70)
    return a, b


def test_ejemplo_de_aceptacion():
    a, b = mascaras_ejemplo()
    c = comparar(analisis(a, clave="a", cuando="2024-08-15"), analisis(b, clave="b", cuando="2026-08-10"), CFG)
    assert c.estado == "ok" and not c.invertidas and not c.reproyectada
    assert c.ha_por_celda == pytest.approx(0.01)
    assert c.area_inicial_ha == pytest.approx(1.00)
    assert c.area_final_ha == pytest.approx(0.80)
    assert c.persistencia_ha == pytest.approx(0.70)
    assert c.perdida_ha == pytest.approx(0.30)
    assert c.perdida_pct == pytest.approx(30.0)
    assert c.ganancia_ha == pytest.approx(0.10)
    assert c.cambio_neto_ha == pytest.approx(-0.20)
    assert c.cambio_neto_pct == pytest.approx(-20.0)
    assert c.cambio_neto_ha == pytest.approx(c.ganancia_ha - c.perdida_ha)
    assert c.area_comun_ha == pytest.approx(20.0) and c.cobertura_comun_pct == pytest.approx(100.0)
    conteo = {k: int((c.mapa == k).sum()) for k in areas.CATEGORIAS}
    assert conteo == {areas.SIN_OBSERVACION: 0, areas.FONDO: 2000 - 110, areas.PERSISTENCIA: 70,
                      areas.PERDIDA: 30, areas.GANANCIA: 10}


def test_qa_oculta_diez_pixeles_de_perdida_en_b():
    a, b = mascaras_ejemplo()
    obs_b = np.ones(a.shape, bool)
    obs_b[5:15, 14] = False             # 10 px donde A tenía nieve y B no: pérdida oculta
    c = comparar(analisis(a, clave="a", cuando="2024-08-15"),
                 analisis(b, clave="b", cuando="2026-08-10", observable=obs_b), CFG)
    assert c.area_inicial_ha == pytest.approx(0.90)
    assert c.perdida_ha == pytest.approx(0.20)
    assert c.perdida_pct == pytest.approx(22.2222, abs=1e-3)
    assert c.persistencia_ha == pytest.approx(0.70) and c.ganancia_ha == pytest.approx(0.10)
    assert int((c.mapa == areas.SIN_OBSERVACION).sum()) == 10
    # Fracción de la nieve inicial observable que quedó fuera de V por falta de datos en B.
    assert c.inicial_fuera_comun_ha == pytest.approx(0.10)
    assert c.inicial_fuera_comun_pct == pytest.approx(10.0)


def test_fechas_invertidas_se_reordenan():
    a, b = mascaras_ejemplo()
    c = comparar(analisis(b, clave="final", cuando="2026-08-10"), analisis(a, clave="inicial", cuando="2024-08-15"),
                 CFG)
    assert c.invertidas and c.clave_inicial == "inicial" and c.clave_final == "final"
    assert c.perdida_ha == pytest.approx(0.30) and c.cambio_neto_pct == pytest.approx(-20.0)
    assert any("orden inverso" in t for t in c.avisos)


def test_fechas_identicas_se_bloquean():
    a, b = mascaras_ejemplo()
    c = comparar(analisis(a, cuando="2024-08-15"), analisis(b, cuando="2024-08-15"), CFG)
    assert c.estado == "no_disponible" and c.perdida_ha is None and "misma fecha" in c.motivos[0]


def test_fecha_ausente():
    a, b = mascaras_ejemplo()
    c = comparar(analisis(a, cuando=None), analisis(b, cuando="2026-08-10"), CFG)
    assert c.estado == "no_disponible" and "fecha" in c.motivos[0]


def test_denominador_cero_da_nd():
    a, b = mascaras_ejemplo()
    c = comparar(analisis(np.zeros_like(a), cuando="2024-08-15"), analisis(b, cuando="2026-08-10"), CFG)
    assert c.estado == "ok" and c.area_inicial_ha == 0.0
    assert c.perdida_pct is None and c.cambio_neto_pct is None
    assert c.ganancia_ha == pytest.approx(0.80) and c.cambio_neto_ha == pytest.approx(0.80)
    assert any("N/D" in m for m in c.motivos)


def test_soporte_comun_vacio():
    a, b = mascaras_ejemplo()
    izquierda, derecha = np.zeros(a.shape, bool), np.zeros(a.shape, bool)
    izquierda[:, :25], derecha[:, 25:] = True, True
    c = comparar(analisis(a, cuando="2024-08-15", observable=izquierda),
                 analisis(b, cuando="2026-08-10", observable=derecha), CFG)
    assert c.estado == "no_disponible" and c.area_comun_ha == 0.0
    for campo in ("area_inicial_ha", "area_final_ha", "perdida_ha", "ganancia_ha", "persistencia_ha",
                  "perdida_pct", "cambio_neto_ha", "cambio_neto_pct"):
        assert getattr(c, campo) is None
    assert (c.mapa == areas.SIN_OBSERVACION).all()


def test_sin_interseccion_de_huellas():
    a, b = mascaras_ejemplo()
    lejos = georef(40, 50, x0=455970.0 + 5000)
    c = comparar(analisis(a, cuando="2024-08-15"), analisis(b, cuando="2026-08-10", g=lejos), CFG)
    assert c.estado == "no_disponible" and "no se intersectan" in c.motivos[0]


def test_sin_qa_o_sin_georreferencia_no_cuantifica():
    a, b = mascaras_ejemplo()
    c = comparar(analisis(a, cuando="2024-08-15"), analisis(b, cuando="2026-08-10", observable=None), CFG)
    assert c.estado == "no_disponible" and "QA" in c.motivos[0]
    c = comparar(analisis(a, cuando="2024-08-15"), analisis(b, cuando="2026-08-10", g=None), CFG)
    assert c.estado == "no_disponible" and "igual tamaño" in c.motivos[0]


def test_celdas_desplazadas_un_numero_entero_de_pixeles():
    """B está en una malla corrida 3 celdas al este: la comparación usa solo la huella común."""
    a, b = mascaras_ejemplo()
    corrida = georef(40, 50, x0=455970.0 + 30)
    b_corrida = np.zeros_like(b)
    b_corrida[:, :-3] = b[:, 3:]        # mismo contenido geográfico visto desde la malla corrida
    c = comparar(analisis(a, cuando="2024-08-15"), analisis(b_corrida, cuando="2026-08-10", g=corrida), CFG)
    assert c.estado == "ok" and c.reproyectada and c.malla.ancho == 47
    assert c.area_inicial_ha == pytest.approx(1.00) and c.area_final_ha == pytest.approx(0.80)
    assert c.perdida_ha == pytest.approx(0.30) and c.ganancia_ha == pytest.approx(0.10)
    assert c.area_comun_ha == pytest.approx(0.01 * 40 * 47)


def test_celdas_desplazadas_media_celda_usa_vecino_mas_proximo():
    a, _ = mascaras_ejemplo()
    media = georef(40, 50, x0=455970.0 + 5, y1=533560.0 - 5)
    c = comparar(analisis(a, cuando="2024-08-15"), analisis(a, cuando="2026-08-10", g=media), CFG)
    assert c.estado == "ok" and c.reproyectada
    assert abs(c.area_final_ha - 1.00) <= 0.21      # a lo sumo una fila y una columna de diferencia
    assert c.persistencia_ha >= 0.80
    assert any("vecino más próximo" in t for t in c.avisos)


def test_distinto_crs_se_reproyecta_a_la_malla_metrica():
    a, _ = mascaras_ejemplo()
    g = georef(40, 50)
    oeste, sur, este, norte = rasterio.warp.transform_bounds(g.crs, "EPSG:4326", *g.limites)
    ancho, alto = 60, 48
    g_geo = Georef("EPSG:4326", ((este - oeste) / ancho, 0.0, oeste, 0.0, -(norte - sur) / alto, norte), ancho, alto)
    b_geo = np.zeros((alto, ancho), np.uint8)
    reproject(a.astype(np.uint8), b_geo, src_transform=g.affine, src_crs=g.crs, dst_transform=g_geo.affine,
              dst_crs=g_geo.crs, resampling=Resampling.nearest)
    assert not g_geo.es_metrico
    c = comparar(analisis(a, cuando="2024-08-15"),
                 Analisis(escena(cuando="2026-08-10", g=g_geo), prediccion_de(b_geo.astype(bool))), CFG)
    assert c.estado == "ok" and c.reproyectada and c.malla.crs == "EPSG:32618" and c.malla.es_metrico
    assert c.ha_por_celda == pytest.approx(0.01, rel=0.05)
    assert c.area_final_ha == pytest.approx(c.area_inicial_ha, rel=0.2)
    assert c.persistencia_ha >= 0.75 * c.area_inicial_ha
    # Una sola escena en coordenadas geográficas también se mide en la malla métrica.
    m = medir(escena(g=g_geo), prediccion_de(b_geo.astype(bool)), CFG)
    assert m.estado == "cuantitativa" and m.malla.es_metrico and m.area_nieve_ha == pytest.approx(1.0, rel=0.2)


def test_afin_rotada():
    t = math.radians(30)
    afin = (10 * math.cos(t), 10 * math.sin(t), 455970.0, 10 * math.sin(t), -10 * math.cos(t), 533560.0)
    assert hectareas_por_celda(afin) == pytest.approx(0.01)
    assert hectareas_por_celda((20.0, 0.0, 0.0, 0.0, -20.0, 0.0)) == pytest.approx(0.04)
    g_rot = Georef("EPSG:32618", afin, 50, 40)
    a, _ = mascaras_ejemplo()
    m = medir(escena(g=g_rot), prediccion_de(a), CFG)
    assert m.estado == "cuantitativa" and m.area_nieve_ha == pytest.approx(1.00)
    # Contra una malla norte-arriba que cubre la misma zona: se reproyecta y el área se conserva aprox.
    o, s, e, n = g_rot.limites
    g_norte = Georef("EPSG:32618", (10.0, 0.0, math.floor(o / 10) * 10, 0.0, -10.0, math.ceil(n / 10) * 10),
                     int((e - o) // 10) + 2, int((n - s) // 10) + 2)
    b = areas.reproyectar_mascara(a, g_rot, g_norte)
    obs = areas.reproyectar_mascara(np.ones(a.shape, bool), g_rot, g_norte)
    c = comparar(Analisis(escena(cuando="2024-08-15", g=g_rot), prediccion_de(a)),
                 Analisis(escena(cuando="2026-08-10", g=g_norte, observable=obs), prediccion_de(b)), CFG)
    assert c.estado == "ok" and c.reproyectada
    assert c.area_inicial_ha == pytest.approx(1.0, rel=0.15) and c.perdida_ha <= 0.15


def test_predicciones_superpuestas_cuentan_la_union():
    m1, m2 = np.zeros((20, 20), bool), np.zeros((20, 20), bool)
    m1[2:12, 2:12] = True               # 100 px
    m2[7:17, 7:17] = True               # 100 px, 25 px solapados
    punt = mapa_puntuacion([0.9, 0.4], [m1, m2], (20, 20))
    assert int((punt > 0).sum()) == 175
    assert float(punt[8, 8]) == pytest.approx(0.9) and float(punt[15, 15]) == pytest.approx(0.4)
    med = medir(escena(alto=20, ancho=20), prediccion_de(punt >= 0.3), CFG)
    assert med.area_nieve_ha == pytest.approx(1.75)
    assert int((punt >= np.float32(0.5)).sum()) == 100      # umbral operativo: solo la instancia de 0,9


@pytest.mark.parametrize("alto,ancho,tam", [(598, 545, 1024), (300, 640, 640), (996, 999, 1024)])
def test_letterbox_en_imagen_no_cuadrada_ida_y_vuelta(alto, ancho, tam):
    import cv2
    mascara = np.zeros((alto, ancho), np.uint8)
    mascara[alto // 5: alto // 2, ancho // 4: ancho // 2] = 1
    mascara[-40:-10, -60:-20] = 1
    lienzo, g = letterbox(np.repeat(mascara[..., None] * 255, 3, axis=2), tam)
    assert lienzo.shape == (tam, tam, 3)
    assert g == geometria_letterbox(alto, ancho, tam)
    assert (lienzo[:g["arriba"]] == 114).all() and (lienzo[:, :g["izquierda"]] == 114).all()
    # Mapa de la red a resolución de prototipos (tam/4), como logits ±.
    logits = cv2.resize(lienzo[..., 0].astype(np.float32), (tam // 4, tam // 4), interpolation=cv2.INTER_AREA)
    logits = logits / 255.0 - 0.5
    logits[lienzo[::4, ::4, 0] == 114] = -0.5
    vuelta = deshacer_letterbox(logits, alto, ancho, tam) > 0
    assert vuelta.shape == (alto, ancho)
    ref = mascara.astype(bool)
    iou = (vuelta & ref).sum() / (vuelta | ref).sum()
    assert iou > 0.95
    cajas = np.array([[g["izquierda"], g["arriba"], g["izquierda"] + g["nuevo_w"], g["arriba"] + g["nuevo_h"]]])
    np.testing.assert_allclose(cajas_a_original(cajas, alto, ancho, tam), [[0, 0, ancho, alto]], atol=1e-3)


def test_medir_exige_mascara_del_tamano_de_la_imagen():
    with pytest.raises(ValueError, match="letterbox"):
        medir(escena(alto=40, ancho=50), prediccion_de(np.zeros((64, 64), bool)), CFG)


def test_estados_de_medicion_de_una_escena():
    a, _ = mascaras_ejemplo()
    m = medir(escena(), prediccion_de(a), CFG)
    assert m.estado == "cuantitativa" and m.area_nieve_ha == pytest.approx(1.0)
    assert m.area_aoi_ha == pytest.approx(20.0) and m.pct_nieve_aoi == pytest.approx(5.0)
    sin_qa = medir(escena(observable=None), prediccion_de(a), CFG)
    assert sin_qa.estado == "aparente_sin_qa" and "sin control de nubes" in sin_qa.etiqueta_area
    assert sin_qa.area_nieve_ha == pytest.approx(1.0)
    sin_escala = medir(escena(g=None), prediccion_de(a), CFG)
    assert sin_escala.estado == "sin_escala" and sin_escala.area_nieve_ha is None and sin_escala.pixeles_nieve == 100
    sin_soporte = medir(escena(observable=np.zeros(a.shape, bool)), prediccion_de(a), CFG)
    assert sin_soporte.estado == "sin_soporte" and sin_soporte.area_nieve_ha is None
    despejada = medir(escena(), prediccion_de(np.zeros_like(a)), CFG)
    assert despejada.estado == "cuantitativa" and despejada.area_nieve_ha == 0.0
    parcial = np.ones(a.shape, bool)
    parcial[5:15, 5:10] = False         # la mitad del parche no es observable
    m = medir(escena(observable=parcial), prediccion_de(a), CFG)
    assert m.area_nieve_ha == pytest.approx(0.50) and m.area_observable_ha == pytest.approx(19.5)


def test_aoi_configurado_y_dominio():
    cfg = {"aoi": {"crs": "EPSG:32618", "caja_wgs84": [-75.397, 4.773, -75.348, 4.827],
                   "transform": [10.0, 0.0, 455970.0, 0.0, -10.0, 533560.0]}}
    g = Georef("EPSG:32618", (10.0, 0.0, 455970.0, 0.0, -10.0, 533560.0), 545, 598)
    aoi = areas.mascara_aoi(g, cfg["aoi"]["caja_wgs84"])
    assert 0.98 < aoi.mean() < 1.0 and aoi[300, 270]
    fuera = Georef("EPSG:32618", (10.0, 0.0, 600000.0, 0.0, -10.0, 533560.0), 50, 40)
    m = medir(escena(g=fuera), prediccion_de(np.ones((40, 50), bool)), cfg)
    assert m.estado == "fuera_de_dominio" and m.area_nieve_ha is None


def test_exportaciones_georreferenciadas():
    a, b = mascaras_ejemplo()
    c = comparar(analisis(a, cuando="2024-08-15"), analisis(b, cuando="2026-08-10"), CFG)
    with MemoryFile(areas.exportar_cambio_geotiff(c)) as mem, mem.open() as ds:
        assert ds.crs.to_epsg() == 32618 and ds.transform.a == 10.0 and ds.dtypes[0] == "uint8"
        assert np.array_equal(ds.read(1), c.mapa)
        assert "3 = pérdida observada" in ds.tags()["leyenda"] and ds.colormap(1)[areas.PERDIDA][0] > 200
    esc = escena()
    m = medir(esc, prediccion_de(a), CFG)
    with MemoryFile(areas.exportar_mascara_geotiff(m, esc)) as mem, mem.open() as ds:
        assert int((ds.read(1) == 2).sum()) == 100
    csv = areas.filas_a_csv([c.como_fila()]).decode("utf-8-sig")
    assert "perdida_bruta_ha" in csv.splitlines()[0] and "0.3" in csv.splitlines()[1]


def test_avisos_de_epoca_y_perfil():
    a, b = mascaras_ejemplo()
    c = comparar(analisis(a, cuando="2024-01-15", perfil="L1C"), analisis(b, cuando="2026-08-10", perfil="L2A"),
                 CFG)
    assert any("Épocas del año" in t for t in c.avisos) and any("Niveles de procesamiento" in t for t in c.avisos)
    c = comparar(analisis(a, cuando="2024-12-20"), analisis(b, cuando="2026-01-05"), CFG)
    assert not any("Épocas del año" in t for t in c.avisos)     # diciembre y enero son meses contiguos
