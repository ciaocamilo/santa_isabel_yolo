"""Hito D · Evaluación: ROC/AUC por conteos, clases ausentes, empates, píxeles ignorados y protocolo de test."""
import numpy as np
import pytest

from santa_isabel import evaluacion as ev
from santa_isabel.flujo import Protocolo, mejor_candidato


def test_auc_del_caso_unitario():
    roc = ev.curva_roc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert roc["definida"] and roc["auc"] == pytest.approx(0.75)
    np.testing.assert_allclose(roc["fpr"], [0, 0, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(roc["tpr"], [0, 0.5, 0.5, 1.0, 1.0])
    assert np.isinf(roc["umbrales"][0]) and roc["umbrales"][1] == pytest.approx(0.8)


def test_una_sola_clase_no_define_roc():
    solo_fondo = ev.curva_roc([0, 0, 0], [0.2, 0.1, 0.9])
    assert not solo_fondo["definida"] and solo_fondo["auc"] is None and "positivos" in solo_fondo["motivo"]
    solo_nieve = ev.curva_roc([1, 1], [0.2, 0.9])
    assert not solo_nieve["definida"] and "negativos" in solo_nieve["motivo"]
    # El agregado sí puede contener ambas clases aunque una escena no.
    acu = ev.AcumuladorROC().agregar(np.array([0, 0, 0]), np.array([0.2, 0.1, 0.9]))
    acu.agregar(np.array([1, 1]), np.array([0.2, 0.9]))
    assert acu.curva_roc()["definida"]


def test_puntuaciones_empatadas():
    roc = ev.curva_roc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
    assert roc["auc"] == pytest.approx(0.5) and len(roc["fpr"]) == 2      # curva degenerada (0,0)→(1,1)
    roc = ev.curva_roc([0, 1, 1, 0, 1], [0.0, 0.7, 0.7, 0.7, 0.9])
    # Umbral 0,9: TPR 1/3, FPR 0. Umbral 0,7 (empate de 2 positivos y 1 negativo): TPR 1, FPR 1/2.
    np.testing.assert_allclose(roc["tpr"], [0, 1 / 3, 1, 1])
    np.testing.assert_allclose(roc["fpr"], [0, 0, 0.5, 1])
    assert roc["auc"] == pytest.approx((1 / 3 + 1) / 2 * 0.5 + 0.5)


def test_sin_predicciones_con_ambas_clases_da_curva_degenerada():
    roc = ev.curva_roc([0, 0, 1, 1], [0.0, 0.0, 0.0, 0.0])
    assert roc["definida"] and roc["auc"] == pytest.approx(0.5)
    acu = ev.AcumuladorROC().agregar(np.array([0, 0, 1, 1]), np.zeros(4))
    assert acu.mejor_umbral_f1(minimo=0.0) is None
    assert acu.confusion_en(0.25) == {"tp": 0, "fp": 0, "fn": 2, "tn": 2}


def test_pixeles_ignorados():
    ref = np.array([[1, 1, 0, 0], [1, 0, 0, 0]], bool)
    pred = np.array([[1, 0, 1, 0], [1, 1, 0, 0]], bool)
    valido = np.array([[1, 1, 1, 1], [0, 0, 1, 1]], bool)
    c = ev.confusion(ref, pred, valido)
    assert c == {"tp": 1, "fp": 1, "fn": 1, "tn": 3, "validos": 6, "ignorados": 2}
    punt = pred.astype(np.float32) * 0.8
    roc_todo, roc_valido = ev.curva_roc(ref, punt), ev.curva_roc(ref, punt, valido)
    assert (roc_todo["positivos"], roc_todo["negativos"]) == (3, 5)
    assert (roc_valido["positivos"], roc_valido["negativos"]) == (2, 4)
    e = ev.evaluar_escena(ref, punt, 0.5, valido, ha_por_celda=0.01)
    assert e["validos"] == 6 and e["area_ref_ha"] == pytest.approx(0.02) and e["error_ha"] == pytest.approx(0.0)


def test_metricas_y_denominadores_cero():
    m = ev.metricas_binarias(tp=6, fp=2, fn=4, tn=88)
    assert m["precision"] == pytest.approx(0.75) and m["recall"] == pytest.approx(0.6)
    assert m["f1"] == pytest.approx(12 / 18) and m["iou"] == pytest.approx(0.5)
    assert m["exactitud"] == pytest.approx(0.94)    # distinta de la precisión positiva (0,75)
    vacio = ev.metricas_binarias(0, 0, 0, 50)
    assert vacio["precision"] is None and vacio["recall"] is None and vacio["f1"] is None and vacio["iou"] is None
    assert set(vacio["nd"]) == {"precision", "recall", "f1", "iou"} and vacio["exactitud"] == 1.0
    sin_pred = ev.metricas_binarias(0, 0, 5, 50)
    assert sin_pred["precision"] is None and sin_pred["recall"] == 0.0 and "TP + FP = 0" in sin_pred["nd"]["precision"]


def test_umbral_f1_empate_conserva_el_mayor():
    ref = np.array([1, 1, 0, 0, 0], bool)
    punt = np.array([0.9, 0.6, 0.3, 0.0, 0.0], np.float32)
    mejor = ev.AcumuladorROC().agregar(ref, punt).mejor_umbral_f1()
    assert mejor["umbral"] == pytest.approx(0.6) and mejor["f1"] == pytest.approx(1.0)
    # Dos umbrales con el mismo F1 = 2/3: 0,8 (TP 1, FP 0, FN 1) y 0,5 (TP 2, FP 2, FN 0) → se conserva 0,8.
    ref = np.array([1, 0, 0, 1], bool)
    punt = np.array([0.8, 0.7, 0.6, 0.5], np.float32)
    mejor = ev.AcumuladorROC().agregar(ref, punt).mejor_umbral_f1()
    assert mejor["f1"] == pytest.approx(2 / 3) and mejor["umbral"] == pytest.approx(0.8) and mejor["empates"] == 2
    pred = punt >= np.float32(mejor["umbral"])
    assert ev.confusion(ref, pred)["tp"] == mejor["tp"]


def test_acumulado_por_bloques_coincide_con_scikit_learn():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(42)
    ref = rng.random((60, 80)) < 0.08
    punt = np.where(rng.random((60, 80)) < 0.3, np.round(rng.random((60, 80)), 2), 0).astype(np.float32)
    punt[ref] = np.maximum(punt[ref], np.round(rng.random(int(ref.sum())), 2) * 0.9).astype(np.float32)
    valido = rng.random((60, 80)) < 0.9
    acu = ev.AcumuladorROC()
    for k in range(0, 60, 7):           # por bloques
        acu.agregar(ref[k:k + 7], punt[k:k + 7], valido[k:k + 7])
    roc = acu.curva_roc()
    y, s = ref[valido], punt[valido]
    assert roc["auc"] == pytest.approx(metrics.roc_auc_score(y, s), abs=1e-12)
    fpr, tpr, _ = metrics.roc_curve(y, s, drop_intermediate=False)
    np.testing.assert_allclose(roc["fpr"], fpr, atol=1e-12)
    np.testing.assert_allclose(roc["tpr"], tpr, atol=1e-12)
    umbral = acu.mejor_umbral_f1()["umbral"]
    c = acu.confusion_en(umbral)
    assert ev.metricas_binarias(**c)["f1"] == pytest.approx(metrics.f1_score(y, s >= np.float32(umbral)))


def test_rasterizado_por_centro_de_pixel_y_txt(tmp_path):
    cuadro = [[0.2, 0.2], [0.6, 0.2], [0.6, 0.6], [0.2, 0.6]]
    m = ev.rasterizar_poligonos([cuadro], 10, 10)
    assert int(m.sum()) == 16 and m[2:6, 2:6].all()
    medio = ev.rasterizar_poligonos([[[0.27, 0.27], [0.67, 0.27], [0.67, 0.67], [0.27, 0.67]]], 10, 10)
    assert int(medio.sum()) == 16 and medio[3:7, 3:7].all()     # decide el centro (col + 0,5; fila + 0,5)
    assert ev.area_poligono_px(cuadro, 10, 10) == pytest.approx(16.0)
    txt = tmp_path / "e.txt"
    txt.write_text("0 0.20000000 0.20000000 0.60000000 0.20000000 0.60000000 0.60000000 0.20000000 0.60000000\n")
    (clase, v), = ev.leer_txt_yolo(txt)
    assert clase == 0 and v.shape == (4, 2)
    vacio = tmp_path / "negativo.txt"
    vacio.write_text("")
    assert ev.leer_txt_yolo(vacio) == []                         # TXT vacío explícito = negativo
    with pytest.raises(FileNotFoundError):                       # TXT ausente = error de integridad
        ev.leer_txt_yolo(tmp_path / "no_existe.txt")
    txt.write_text("1 0.1 0.1 0.2 0.1 0.2 0.2\n")
    with pytest.raises(ValueError, match="clase 1"):
        ev.leer_txt_yolo(txt)
    txt.write_text("0 0.1 0.1 1.2 0.1 0.2 0.2\n")
    with pytest.raises(ValueError, match=r"fuera de \[0, 1\]"):
        ev.leer_txt_yolo(txt)


def test_errores_de_superficie():
    e = ev.error_area(1.0, 0.8)
    assert e["error_ha"] == pytest.approx(-0.2) and e["error_pct"] == pytest.approx(-20.0)
    r = ev.resumen_areas([ev.error_area(1.0, 0.8), ev.error_area(2.0, 2.6), ev.error_area(None, 1.0)])
    assert r["escenas"] == 2 and r["sesgo_ha"] == pytest.approx(0.2) and r["mae_ha"] == pytest.approx(0.4)
    assert ev.error_area(0.0, 0.5)["error_pct"] is None


def test_protocolo_impide_que_test_intervenga_en_la_seleccion():
    p = Protocolo()
    p.usar("val", "seleccion de checkpoint")
    with pytest.raises(RuntimeError, match="antes de congelar"):
        p.usar("test", "prueba temporal")
    p.usar("val", "seleccion de umbral")
    p.congelar({"checkpoint": "last.pt", "imgsz": 1024, "umbral": 0.31})
    p.usar("test", "prueba temporal")
    with pytest.raises(RuntimeError, match="ya se aplicó una vez"):
        p.usar("test", "segunda pasada")
    with pytest.raises(RuntimeError, match="nueva selección"):
        p.usar("val", "seleccion de umbral")
    with pytest.raises(RuntimeError):
        p.congelar({"umbral": 0.5})
    r = p.resumen()
    assert r["evaluaciones_test"] == 1 and r["test_intervino_en_seleccion"] is False
    assert [e["split"] for e in r["eventos"]] == ["val", "val", "test"]


def test_umbral_operativo_retrocede_un_margen_sin_cambiar_la_mascara_de_validacion():
    from types import SimpleNamespace

    from santa_isabel.flujo import elegir_umbral

    def elemento(puntuaciones):
        ref = np.array([[1, 1, 0, 0, 0]], bool)
        return [{"nombre": "e", "referencia": ref, "puntuacion": np.array([puntuaciones], np.float32),
                 "observable": np.ones_like(ref), "ambiguos": None,
                 "prediccion": SimpleNamespace(umbral_propuestas=0.01)}]
    el = elemento([0.9, 0.6, 0.3, 0.0, 0.0])
    u = elegir_umbral(el, "observable")
    assert u["umbral_f1"] == pytest.approx(0.6) and u["umbral"] == pytest.approx(0.595) and u["f1"] == 1.0
    assert np.array_equal(el[0]["puntuacion"] >= np.float32(u["umbral"]), el[0]["puntuacion"] >= np.float32(0.6))
    # Si la siguiente puntuación está más cerca que el margen, se retrocede solo hasta la mitad.
    el = elemento([0.9, 0.6, 0.598, 0.0, 0.0])
    u = elegir_umbral(el, "observable")
    assert u["umbral"] == pytest.approx(0.599, abs=1e-6) and u["margen_numerico"] == pytest.approx(0.001, abs=1e-6)
    assert int((el[0]["puntuacion"] >= np.float32(u["umbral"])).sum()) == 2
    sin_propuestas = elegir_umbral(elemento([0.0] * 5), "observable")
    assert sin_propuestas["umbral"] == 0.25 and sin_propuestas["f1"] is None


def test_seleccion_por_iou_y_desempate_por_mae():
    cand = [{"iou": 0.40, "mae_ha": 1.0}, {"iou": 0.55, "mae_ha": 3.0}, {"iou": 0.55, "mae_ha": 2.0},
            {"iou": None, "mae_ha": 0.1}]
    assert mejor_candidato(cand) == 2
