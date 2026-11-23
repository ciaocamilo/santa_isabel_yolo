"""Hito A · Inferencia con predicción inyectada (solo en pruebas) y contrato del paquete del modelo."""
import hashlib
import json

import numpy as np
import pytest

from conftest import BackendFalso, modelo_falso
from santa_isabel import paquete as pq
from santa_isabel.inferencia import predecir, validar_umbral, ventanas
from santa_isabel.paquete import ErrorPaquete


def imagen(alto=40, ancho=50):
    return np.zeros((alto, ancho, 3), np.uint8)


def test_puntuacion_es_el_maximo_y_la_mascara_la_union_sobre_el_umbral():
    m1, m2, m3 = (np.zeros((40, 50), bool) for _ in range(3))
    m1[5:15, 5:15], m2[10:20, 10:20], m3[30:35, 30:35] = True, True, True
    modelo = modelo_falso(BackendFalso([0.9, 0.6, 0.2], [m1, m2, m3]), umbral=0.5)
    p = predecir(imagen(), modelo)
    assert p.puntuacion.dtype == np.float32 and p.puntuacion.shape == (40, 50) == p.mascara.shape
    assert float(p.puntuacion[12, 12]) == pytest.approx(0.9) and float(p.puntuacion[18, 18]) == pytest.approx(0.6)
    assert float(p.puntuacion[32, 32]) == pytest.approx(0.2) and float(p.puntuacion[0, 0]) == 0.0
    assert np.array_equal(p.mascara, m1 | m2)                 # unión de instancias con confianza ≥ 0,5
    assert len(p.instancias) == 3 and p.instancias_operativas == 2 and not p.tope_alcanzado
    assert p.instancias[0]["caja"] == [5, 5, 15, 15] and p.instancias[0]["area_px"] == 100
    assert modelo.backend.llamadas == [{"forma": (40, 50, 3), "imgsz": 1024, "conf": 0.01, "max_det": 300}]
    # Cambiar el umbral operativo no repite la inferencia y es coherente con la puntuación.
    q = p.con_umbral(0.15)
    assert np.array_equal(q.mascara, m1 | m2 | m3) and len(modelo.backend.llamadas) == 1
    assert np.array_equal(predecir(imagen(), modelo, umbral_operativo=0.15).mascara, q.mascara)


def test_umbral_operativo_valido():
    assert validar_umbral(0.3, 0.01) == 0.3
    for malo in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            validar_umbral(malo, 0.01)
    with pytest.raises(ValueError, match="umbral de propuestas"):
        validar_umbral(0.005, 0.01)
    with pytest.raises(ValueError, match="uint8"):
        predecir(np.zeros((4, 4, 3), np.float32), modelo_falso(BackendFalso()))


def test_escena_sin_detecciones_y_tope_de_detecciones():
    p = predecir(imagen(), modelo_falso(BackendFalso()))
    assert not p.mascara.any() and p.instancias == [] and float(p.puntuacion.max()) == 0.0
    muchas = np.zeros((5, 40, 50), bool)
    for k in range(5):
        muchas[k, k, k] = True
    p = predecir(imagen(), modelo_falso(BackendFalso([0.9] * 5, muchas), max_det=5))
    assert p.tope_alcanzado                                    # avisa de que el tope pudo recortar parches


def test_teselas_conservan_el_origen_y_unen_por_maximo():
    assert ventanas(598, 545, 640, 96) == [(0, 0, 598, 545)]
    lista = ventanas(996, 999, 640, 96)
    assert lista == [(0, 0, 640, 640), (0, 359, 640, 640), (356, 0, 640, 640), (356, 359, 640, 640)]
    cubierta = np.zeros((996, 999), int)
    for f, c, h, w in lista:
        cubierta[f:f + h, c:c + w] += 1
    assert cubierta.min() >= 1 and cubierta.max() == 4

    def por_llamada(k, rgb):
        m = np.zeros(rgb.shape[:2], bool)
        m[100:110, 300:400] = True                             # mismo recuadro local en cada tesela
        return [0.5 + 0.1 * k], [m]
    p = predecir(np.zeros((996, 999, 3), np.uint8), modelo_falso(BackendFalso(por_llamada=por_llamada)))
    assert p.teselas == 4 and p.mascara.shape == (996, 999)
    assert float(p.puntuacion[105, 350]) == pytest.approx(0.5)             # solo tesela 0
    assert float(p.puntuacion[105, 700]) == pytest.approx(0.6)             # tesela 1: columnas 659–759
    assert float(p.puntuacion[460, 700]) == pytest.approx(0.8)             # tesela 3 (fila 356 + 100)
    assert p.instancias[1]["caja"] == [659, 100, 759, 110] and p.instancias[3]["caja"] == [659, 456, 759, 466]
    # Zona solapada entre teselas 0 y 1 (columnas 659–700 no; 359–400 sí): se conserva el máximo.
    def solapadas(k, rgb):
        m = np.zeros(rgb.shape[:2], bool)
        m[10:20, :] = True
        return [0.3 if k == 0 else 0.7], [m]
    p = predecir(np.zeros((640, 999, 3), np.uint8), modelo_falso(BackendFalso(por_llamada=solapadas)))
    assert float(p.puntuacion[15, 100]) == pytest.approx(0.3) and float(p.puntuacion[15, 500]) == pytest.approx(0.7)
    assert float(p.puntuacion[15, 900]) == pytest.approx(0.7)


# --- Paquete ----------------------------------------------------------------------------------------------------------
def contrato_minimo(identificador="si-rgb-prueba"):
    return {"id": identificador, "variante": "rgb",
            "entrada": {"imgsz": 1024, "tesela_px": 640, "solape_px": 96},
            "posproceso": {"umbral_propuestas": 0.01, "umbral_operativo": 0.3, "iou_nms": 0.7, "max_det": 300},
            "backends": {"onnx": {"archivo": "pesos/best.onnx"}}, "backend_servicio": "onnx",
            "dataset": {"version": "v0003"}, "referencia": {"origen": "automatico_ndsi"}}


def test_publicar_y_validar_contrato(tmp_path):
    ruta = pq.publicar_paquete(tmp_path, contrato_minimo(), {"pesos/best.onnx": b"pesos falsos",
                                                            "metricas.json": b"{}"})
    assert ruta.name == "si-rgb-prueba" and (tmp_path / "ACTIVO").read_text().strip() == "si-rgb-prueba"
    c = pq.leer_contrato(ruta)
    assert c["esquema"] == "1.0" and c["tarea"] == "segment" and c["clase"] == {"id": 0,
                                                                                "nombre": "nieve_hielo_visible"}
    assert c["backends"]["onnx"]["sha256"] == c["archivos"]["pesos/best.onnx"]
    assert pq.verificar_hashes(ruta, c) == 2 and pq.resolver_activo(tmp_path) == ruta
    # SHA256SUMS describe exactamente los bytes de cada archivo del paquete, incluido modelo.json.
    sumas = dict(reversed(linea.split("  ", 1)) for linea in (ruta / "SHA256SUMS").read_bytes().decode().splitlines())
    assert set(sumas) == {"modelo.json", "pesos/best.onnx", "metricas.json"}
    for rel, esperado in sumas.items():
        assert hashlib.sha256((ruta / rel).read_bytes()).hexdigest() == esperado, rel
    assert b"\r" not in (ruta / "modelo.json").read_bytes() + (ruta / "SHA256SUMS").read_bytes()
    with pytest.raises(ErrorPaquete, match="inmutables"):
        pq.publicar_paquete(tmp_path, contrato_minimo(), {"pesos/best.onnx": b"otros"})
    assert not list(tmp_path.glob("_tmp_*"))


def test_fallas_de_validacion_explican_como_resolver(tmp_path):
    with pytest.raises(ErrorPaquete, match="No hay un modelo publicado") as exc:
        pq.resolver_activo(tmp_path)
    assert "01_entrenar_evaluar.ipynb" in exc.value.como_resolver and "pesos" in exc.value.como_resolver
    ruta = pq.publicar_paquete(tmp_path, contrato_minimo(), {"pesos/best.onnx": b"pesos falsos"})
    (ruta / "pesos" / "best.onnx").write_bytes(b"pesos alterados")
    with pytest.raises(ErrorPaquete, match="Hash erróneo") as exc:
        pq.cargar_paquete(ruta)
    assert "inmutable" in exc.value.como_resolver and "Cómo resolverlo" in str(exc.value)
    (ruta / "pesos" / "best.onnx").unlink()
    with pytest.raises(ErrorPaquete, match="Falta el archivo"):
        pq.cargar_paquete(ruta)

    def alterado(**cambios):
        c = json.loads((ruta / "modelo.json").read_text(encoding="utf-8"))
        c.update(cambios)
        (ruta / "modelo.json").write_text(json.dumps(c), encoding="utf-8")
    for cambios, patron in (({"esquema": "9.9"}, "esquema"), ({"tarea": "detect"}, "tarea"),
                            ({"clase": {"id": 0, "nombre": "person"}}, "clase"), ({"variante": "sar"}, "Variante"),
                            ({"posproceso": {"umbral_propuestas": 0.5, "umbral_operativo": 0.3, "max_det": 300}},
                             "umbral"), ({"backends": {}}, "backend")):
        original = (ruta / "modelo.json").read_text(encoding="utf-8")
        alterado(**cambios)
        with pytest.raises(ErrorPaquete, match=patron) as exc:
            pq.leer_contrato(ruta)
        assert exc.value.como_resolver
        (ruta / "modelo.json").write_text(original, encoding="utf-8")
    (tmp_path / "ACTIVO").write_text("no_existe\n")
    with pytest.raises(ErrorPaquete, match="No existe la carpeta"):
        pq.resolver_activo(tmp_path)


def test_backend_no_incluido_o_pesos_ilegibles(tmp_path):
    ruta = pq.publicar_paquete(tmp_path, contrato_minimo(), {"pesos/best.onnx": b"no es un onnx"})
    with pytest.raises(ErrorPaquete, match="no incluye el backend"):
        pq.cargar_paquete(ruta, backend="ultralytics")
    pytest.importorskip("onnxruntime")
    with pytest.raises(ErrorPaquete, match="No se pudieron cargar los pesos"):
        pq.cargar_paquete(ruta, backend="onnx")
