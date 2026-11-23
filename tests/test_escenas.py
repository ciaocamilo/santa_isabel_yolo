"""Hitos A y B · Lectura de escenas: paquete del generador, GeoTIFF y PNG; fallas controladas."""
import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from conftest import (BackendFalso, crear_paquete_productor, escribir_tif, georef, modelo_falso,
                      reflectancia_sintetica)
from santa_isabel import escenas as es
from santa_isabel.areas import comparar
from santa_isabel.escenas import ErrorEscena, leer_escena
from santa_isabel.flujo import analizar_escena
from santa_isabel.inferencia import preparar_entrada

CFG = {"aoi": {"crs": "EPSG:32618"}}


def tif_reflectancia(ruta, refl, g, enteros=False, descripciones=("B2", "B3", "B4", "B8", "B11")):
    pila = np.stack([np.where(np.isfinite(refl[b]), refl[b], es.NODATA) for b in ("B2", "B3", "B4", "B8", "B11")])
    if enteros:
        dn = np.where(pila == es.NODATA, 0, np.round(pila / es.ESCALA_DN)).astype(np.uint16)
        return escribir_tif(ruta, dn, g, 0, descripciones)
    return escribir_tif(ruta, pila.astype(np.float32), g, es.NODATA, descripciones)


def test_paquete_del_generador(tmp_path):
    carpeta = crear_paquete_productor(tmp_path / "escena")
    e = leer_escena(carpeta)
    assert e.tipo == "paquete_productor" and e.perfil == "L2A" and e.fecha.isoformat() == "2024-08-15T15:31:40+00:00"
    assert e.georef.crs == "EPSG:32618" and e.georef.ha_por_celda == pytest.approx(0.01) and e.escala_verificada
    assert e.rgb8.shape == (30, 36, 3) and e.rgb8.dtype == np.uint8 and e.receta["verificada"]
    assert not e.extension[:, :2].any() and (e.rgb8[:, :2] == 0).all()        # fuera de soporte = 0
    assert e.tiene_qa and not e.observable[:5, -6:].any() and e.observable[10, 12]
    assert "calidad.tif = 2" in e.procedencia_observable
    # La misma escena cargada como ZIP y como archivos sueltos {nombre: bytes}.
    zip_ruta = tmp_path / "escena.zip"
    with zipfile.ZipFile(zip_ruta, "w") as z:
        for f in carpeta.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(carpeta).as_posix())
    assert np.array_equal(leer_escena(zip_ruta, dir_temporal=tmp_path).rgb8, e.rgb8)


def test_png_con_metadatos_y_geotiff_de_la_misma_fecha_equivalen(tmp_path):
    """Hito B: color, forma y áreas equivalentes entre el PNG del generador y el GeoTIFF de reflectancia."""
    refl = reflectancia_sintetica()
    carpeta = crear_paquete_productor(tmp_path / "escena", refl=refl)
    del_png = leer_escena(carpeta)
    g = georef(*refl["B4"].shape)
    for enteros in (False, True):
        tif = tif_reflectancia(tmp_path / f"S2_20240815T153140_{'dn' if enteros else 'refl'}.tif", refl, g, enteros)
        del_tif = leer_escena(tif, "L2A")
        assert del_tif.tipo == "geotiff_reflectancia" and del_tif.fecha.date() == del_png.fecha.date()
        if enteros:     # la cuantización a DN (0,0001) puede mover un píxel al otro lado del redondeo de 8 bits
            assert np.abs(del_tif.rgb8.astype(int) - del_png.rgb8.astype(int)).max() <= 1
        else:
            assert np.array_equal(del_tif.rgb8, del_png.rgb8)                 # canal a canal
        assert np.array_equal(del_tif.extension, del_png.extension) and del_tif.georef.misma_malla(del_png.georef)
        assert del_tif.receta["variantes"] == ["rgb", "espectral"]
    # GeoTIFF RGB de 8 bits ya codificado.
    rgb_tif = escribir_tif(tmp_path / "rgb8_2024-08-15.tif", del_png.rgb8.transpose(2, 0, 1), g)
    del_rgb = leer_escena(rgb_tif)
    assert del_rgb.tipo == "geotiff_rgb" and np.array_equal(del_rgb.rgb8, del_png.rgb8)
    assert del_rgb.fecha.date() == del_png.fecha.date() and any("sin receta declarada" in a for a in del_rgb.avisos)
    # Misma predicción inyectada → misma área; el GeoTIFF sin QA la rotula como «aparente».
    mascara = np.zeros(refl["B4"].shape, bool)
    mascara[8:18, 10:24] = True
    modelo = modelo_falso(BackendFalso([0.9], [mascara]))
    a_png, a_tif = analizar_escena(del_png, modelo, CFG), analizar_escena(del_tif, modelo, CFG)
    assert a_png.medicion.estado == "cuantitativa" and a_tif.medicion.estado == "aparente_sin_qa"
    assert a_png.medicion.area_nieve_ha == pytest.approx(1.40) == a_tif.medicion.area_nieve_ha
    assert modelo.backend.llamadas[0]["forma"] == (30, 36, 3)
    # Con calidad.tif acompañando al GeoTIFF la medición vuelve a ser cuantitativa.
    con_qa = leer_escena([tif, carpeta / "calidad.tif", carpeta / "soporte_png.tif"], "L2A")
    assert con_qa.tiene_qa and np.array_equal(con_qa.observable, del_png.observable)


def test_orden_de_canales_y_variante_espectral(tmp_path):
    refl = reflectancia_sintetica()
    e = leer_escena(tif_reflectancia(tmp_path / "e.tif", refl, georef(*refl["B4"].shape)), "L2A")
    rgb = preparar_entrada(e, {"variante": "rgb"})
    assert rgb[12, 15].tolist() == [255, 255, 255]                    # nieve saturada en R=B4, G=B3, B=B2
    b4, b3, b2 = (refl[b][3, 5] for b in ("B4", "B3", "B2"))
    assert rgb[3, 5].tolist() == [int(np.floor(v / 0.6 * 255 + 0.5)) for v in (b4, b3, b2)]
    esp = preparar_entrada(e, {"variante": "espectral"})
    ndsi = (0.7 - 0.06) / (0.7 + 0.06)
    assert esp[12, 15, 0] == int(np.floor((ndsi + 1) / 2 * 255 + 0.5)) and esp[12, 15, 2] == 255
    assert np.array_equal(esp, es.codificar_espectral(refl))
    # El backend de Ultralytics recibe BGR: el canal 0 que se le entrega es B2.
    assert np.array_equal(rgb[..., ::-1][..., 0], rgb[..., 2])


def test_bandas_incompatibles_fallan_de_forma_controlada(tmp_path):
    refl = reflectancia_sintetica()
    g = georef(*refl["B4"].shape)
    sin_nombres = tif_reflectancia(tmp_path / "sin_nombres.tif", refl, g, descripciones=None)
    with pytest.raises(ErrorEscena, match="mapeo explícito") as exc:
        leer_escena(sin_nombres, "L2A")
    assert exc.value.requiere == "mapeo_bandas"
    ok = leer_escena(sin_nombres, "L2A", mapeo_bandas={"B2": 1, "B3": 2, "B4": 3, "B8": 4, "B11": 5})
    assert ok.receta["variantes"] == ["rgb", "espectral"]
    with pytest.raises(ErrorEscena, match="fuera del archivo"):
        leer_escena(sin_nombres, "L2A", mapeo_bandas={"B2": 1, "B3": 2, "B4": 9})
    # Solo visibles: RGB sí, espectral no (los índices no se derivan de un RGB).
    solo_rgb = escribir_tif(tmp_path / "visibles.tif", np.stack([refl["B2"], refl["B3"], refl["B4"]]), g,
                            descripciones=("B2", "B3", "B4"))
    e = leer_escena(solo_rgb, "L2A")
    assert e.receta["variantes"] == ["rgb"]
    with pytest.raises(ErrorEscena, match="B3, B4, B8 y B11"):
        preparar_entrada(e, {"variante": "espectral"})
    # Valores digitales sin nivel de procesamiento.
    dn = tif_reflectancia(tmp_path / "dn.tif", refl, g, enteros=True)
    with pytest.raises(ErrorEscena, match="L2A o L1C") as exc:
        leer_escena(dn)
    assert exc.value.requiere == "perfil"
    with pytest.raises(ErrorEscena, match="Perfil"):
        leer_escena(dn, "L3")


def test_fecha_ausente_impide_comparar_y_se_puede_indicar(tmp_path):
    refl = reflectancia_sintetica()
    carpeta = crear_paquete_productor(tmp_path / "a", refl=refl)
    sin_fecha = tif_reflectancia(tmp_path / "recorte.tif", refl, georef(*refl["B4"].shape))
    e = leer_escena([sin_fecha, carpeta / "calidad.tif", carpeta / "soporte_png.tif"], "L2A")
    assert e.fecha is None and any("fecha" in a for a in e.avisos)
    modelo = modelo_falso(BackendFalso())
    c = comparar(analizar_escena(leer_escena(carpeta), modelo, CFG), analizar_escena(e, modelo, CFG), CFG)
    assert c.estado == "no_disponible" and "fecha" in c.motivos[0]
    con_fecha = leer_escena([sin_fecha, carpeta / "calidad.tif", carpeta / "soporte_png.tif"], "L2A",
                            fecha="2026-08-10")
    c = comparar(analizar_escena(leer_escena(carpeta), modelo, CFG), analizar_escena(con_fecha, modelo, CFG), CFG)
    assert c.estado == "ok" and c.area_inicial_ha == 0.0
    with pytest.raises(ErrorEscena, match="Fecha no reconocida"):
        leer_escena(sin_fecha, "L2A", fecha="ayer")


def test_hash_erroneo_y_metadato_incoherente(tmp_path):
    carpeta = crear_paquete_productor(tmp_path / "escena")
    png = next((carpeta / "rgb").glob("*.png"))
    arr = np.asarray(Image.open(png)).copy()
    arr[10, 10] = (1, 2, 3)
    Image.fromarray(arr).save(png)
    with pytest.raises(ErrorEscena, match="Hash erróneo"):
        leer_escena(carpeta)
    carpeta = crear_paquete_productor(tmp_path / "otra")
    meta = json.loads((carpeta / "metadata.json").read_text(encoding="utf-8"))
    meta["malla"]["ancho"] += 1
    (carpeta / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ErrorEscena, match="no corresponde a su metadato"):
        leer_escena(carpeta)
    meta["malla"]["ancho"] -= 1
    del meta["adquisicion_utc"]
    (carpeta / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ErrorEscena, match="adquisicion_utc"):
        leer_escena(carpeta)


def test_png_suelto_sin_escala_y_png_original_reconocido_por_hash(tmp_path):
    carpeta = crear_paquete_productor(tmp_path / "ejemplos" / "escena")
    png = next((carpeta / "rgb").glob("*.png"))
    copia = tmp_path / "descarga.png"
    copia.write_bytes(png.read_bytes())
    suelto = leer_escena(copia)
    assert suelto.tipo == "imagen_rgb" and suelto.georef is None and not suelto.tiene_qa
    modelo = modelo_falso(BackendFalso([0.9], [np.ones((30, 36), bool)]))
    a = analizar_escena(suelto, modelo, CFG)
    assert a.medicion.estado == "sin_escala" and a.medicion.area_nieve_ha is None
    # Dos capturas de igual tamaño no demuestran coincidencia espacial.
    otra = leer_escena(copia, fecha="2026-08-10")
    suelto.fecha = es.interpretar_fecha("2024-08-15")
    c = comparar(a, analizar_escena(otra, modelo, CFG), CFG)
    assert c.estado == "no_disponible" and "igual tamaño" in c.motivos[0]
    # Con el índice de escenas conocidas, el PNG original recupera malla y QA por su hash.
    indice = es.indice_ejemplos(tmp_path / "ejemplos")
    assert len(indice) == 1
    reconocido = leer_escena(copia, indice=indice)
    assert reconocido.tipo == "paquete_productor" and reconocido.tiene_qa and reconocido.georef is not None
    assert any("reconocido por su hash" in t for t in reconocido.avisos)
    # Un JPG del mismo contenido no coincide en hash: sigue siendo inferencia visual.
    jpg = tmp_path / "captura.jpg"
    Image.open(png).save(jpg, quality=90)
    visual = leer_escena(jpg, indice=indice)
    assert visual.tipo == "imagen_rgb" and any("JPG" in t for t in visual.avisos)


def test_limites_y_formatos_rechazados(tmp_path):
    grande = tmp_path / "grande.png"
    Image.new("RGB", (400, 300)).save(grande)
    with pytest.raises(ErrorEscena, match="máximo admitido"):
        leer_escena(grande, limites={"max_pixeles": 100_000})
    with pytest.raises(ErrorEscena, match="pesa"):
        leer_escena(grande, limites={"max_bytes": 100})
    dieciseis = tmp_path / "dieciseis.png"
    Image.fromarray((np.arange(64, dtype=np.uint16).reshape(8, 8) * 900)).save(dieciseis)
    with pytest.raises(ErrorEscena, match="no es de 8 bits"):
        leer_escena(dieciseis)
    basura = tmp_path / "roto.png"
    basura.write_bytes(b"esto no es un PNG")
    with pytest.raises(ErrorEscena, match="No se pudo decodificar"):
        leer_escena(basura)
    with pytest.raises(ErrorEscena, match="No se reconoce"):
        leer_escena({"pesos.pt": b"\x00" * 10}, dir_temporal=tmp_path)
    with pytest.raises(ErrorEscena, match="varias imágenes"):
        leer_escena([grande, dieciseis])


def test_zip_malicioso_se_rechaza(tmp_path):
    def zip_con(nombre, datos=b"x"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr(nombre, datos)
        return {"escena.zip": buf.getvalue()}
    for nombre in ("../fuera.png", "a/../../fuera.png", "/absoluto.png"):
        with pytest.raises(ErrorEscena, match="fuera del directorio temporal"):
            leer_escena(zip_con(nombre), dir_temporal=tmp_path)
    assert not (tmp_path.parent / "fuera.png").exists()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("bomba.tif", b"\x00" * 3_000_000)
    with pytest.raises(ErrorEscena, match="al expandirse"):
        es.extraer_zip_seguro(io.BytesIO(buf.getvalue()), tmp_path / "x", {"zip_max_bytes_expandido": 1_000_000})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for k in range(5):
            z.writestr(f"f{k}.txt", b"x")
    with pytest.raises(ErrorEscena, match="máximo admitido"):
        es.extraer_zip_seguro(io.BytesIO(buf.getvalue()), tmp_path / "y", {"zip_max_archivos": 3})
