"""Hitos A y F · La aplicación: abre sin modelo y explica qué necesita; con el paquete real analiza el ejemplo.

Usa el probador de Streamlit (AppTest), que ejecuta app.py sin navegador.
"""
import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
APP = str(RAIZ / "app.py")


@pytest.fixture(autouse=True)
def cache_limpia():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def textos(elementos):
    return " ".join(str(getattr(e, "value", "")) for e in elementos)


def paquete_publicado():
    activo = RAIZ / "models" / "ACTIVO"
    if not activo.is_file() or not (RAIZ / "models" / activo.read_text(encoding="utf-8").strip()).is_dir():
        pytest.skip("no hay un paquete publicado en models/ (ejecute el notebook)")
    if len(list((RAIZ / "examples").glob("*/metadata.json"))) < 2:
        pytest.skip("no hay dos escenas de ejemplo en examples/")


def test_sin_modelo_abre_y_explica_que_necesita(monkeypatch):
    monkeypatch.setenv("SANTA_ISABEL_PAQUETE", "paquete_que_no_existe")
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert "Nieve y hielo visible" in at.title[0].value
    assert [t.label for t in at.tabs] == ["Analizar una imagen", "Comparar dos fechas", "Modelo y evaluación"]
    errores = textos(at.error)
    assert "no hay un modelo ajustado publicado" in errores and "no" in errores
    cuerpo = textos(at.markdown)
    assert "models/<id>/" in cuerpo and "models/ACTIVO" in cuerpo and "Cómo resolverlo" in cuerpo
    assert "Camilo Andrés Castañeda Galindo" in cuerpo and "Maestría en Inteligencia Artificial · 2026" in cuerpo
    assert "pendientes" in textos(at.warning)
    assert not at.metric                       # nunca presenta resultados de nieve sin pesos ajustados
    assert not [b for b in at.button if b.label in ("Analizar", "Comparar")]


@pytest.mark.paquete_real
def test_demo_de_dos_fechas_con_el_paquete_real():
    paquete_publicado()
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception and not at.error
    at.button(key="btn_ejemplo_dos").click().run()
    assert not at.exception
    at.button(key="btn_comparar").click().run()
    assert not at.exception and not at.error
    metricas = {m.label: m.value for m in at.metric}
    for rotulo in ("Pérdida bruta", "Pérdida / inicial", "Ganancia", "Cambio neto", "Cambio neto / inicial",
                   "Área común observable"):
        assert rotulo in metricas, metricas
    assert metricas["Pérdida bruta"].endswith(" ha") and metricas["Área común observable"] != "N/D"
    # El resultado de la app coincide con el del notebook para la misma corrida (mismo paquete y ejemplos).
    activo = (RAIZ / "models" / "ACTIVO").read_text(encoding="utf-8").strip()
    met = json.loads((RAIZ / "models" / activo / "metricas.json").read_text(encoding="utf-8"))
    demo = RAIZ / "reports" / met["run_id"] / "demo_comparacion.csv"
    if demo.is_file():
        import csv
        with open(demo, newline="", encoding="utf-8-sig") as f:
            fila = next(csv.DictReader(f))
        en_app = float(metricas["Pérdida bruta"].replace(" ha", "").replace("\u202f", "").replace(",", "."))
        assert en_app == pytest.approx(float(fila["perdida_bruta_ha"]), abs=0.06)
    assert any("Copernicus Sentinel-2" in c.value for c in at.caption)
    assert [b.label for b in at.get("download_button")] == ["Descargar comparación (CSV)",
                                                            "Descargar mapa de cambio (GeoTIFF)"]


@pytest.mark.paquete_real
def test_dos_sesiones_aisladas_y_analisis_de_una_escena():
    paquete_publicado()
    sesion_a = AppTest.from_file(APP, default_timeout=300).run()
    sesion_b = AppTest.from_file(APP, default_timeout=300).run()
    sesion_a.button(key="btn_ejemplo_una").click().run()
    sesion_a.button(key="btn_analizar_una").click().run()
    assert not sesion_a.exception and not sesion_a.error
    metricas = {m.label: m.value for m in sesion_a.metric}
    assert metricas["Superficie"].endswith(" ha") and metricas["Cobertura observable"].endswith(" ha")
    assert "una" in sesion_a.session_state and sesion_a.session_state["una"]["analisis"] is not None
    sesion_b.run()                              # la otra sesión no ve la escena ni el resultado
    assert "una" not in sesion_b.session_state and not sesion_b.metric
    assert "Aún no hay ninguna escena" in textos(sesion_b.info)
    assert sesion_a.session_state["una"]["escena"].tiene_qa
