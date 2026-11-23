"""Captura evidencias de la app en ejecución a tamaño de portátil (1366 × 768) con Playwright sobre Edge.

Recorre el guion de la demostración: portada, una escena, dos fechas y «Modelo y evaluación». Las capturas se
guardan en reports/<run_id>/capturas/. Requiere `pip install playwright` (usa el Edge instalado, sin descargar
navegadores) y la app arrancada:

    streamlit run app.py --server.port 8765
    python tools/capturar_app.py http://127.0.0.1:8765 reports/<run_id>/capturas
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def capturar(url, destino, canal="msedge"):
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    resumen = {}
    with sync_playwright() as p:
        navegador = p.chromium.launch(channel=canal, headless=True)
        pagina = navegador.new_page(viewport={"width": 1366, "height": 768}, device_scale_factor=1)
        pagina.goto(url, wait_until="domcontentloaded")
        pagina.get_by_role("heading", name="Nieve y hielo visible").wait_for(timeout=120_000)
        pagina.get_by_role("button", name="Cargar ejemplo del Santa Isabel", exact=True).wait_for(timeout=60_000)
        pagina.wait_for_timeout(800)
        pagina.screenshot(path=str(destino / "01_portada.png"))

        # Una escena
        pagina.get_by_role("button", name="Cargar ejemplo del Santa Isabel", exact=True).click()
        pagina.get_by_role("button", name="Analizar", exact=True).click()
        pagina.get_by_text("Predicción superpuesta").first.wait_for(timeout=120_000)
        pagina.wait_for_timeout(1500)
        pagina.get_by_text("Predicción superpuesta").first.evaluate("e => e.scrollIntoView({block: 'center'})")
        pagina.wait_for_timeout(800)
        pagina.screenshot(path=str(destino / "02_una_escena.png"))
        resumen["una_escena"] = [t.replace("\n", " = ") for t in
                                 pagina.locator('[data-testid="stMetric"]').all_inner_texts()]

        # Dos fechas
        pagina.get_by_role("tab", name="Comparar dos fechas").click()
        pagina.get_by_role("button", name="Cargar ejemplo del Santa Isabel (dos fechas)").click()
        pagina.get_by_role("button", name="Comparar", exact=True).click()
        pagina.get_by_text("Mapa de cambio sobre el soporte común").first.wait_for(timeout=180_000)
        pagina.wait_for_timeout(2000)
        pagina.get_by_role("heading", name="pérdida observada de cobertura").evaluate("e => e.scrollIntoView({block: 'start'})")
        pagina.wait_for_timeout(800)
        pagina.screenshot(path=str(destino / "03_dos_fechas_mapa_de_cambio.png"))
        panel = pagina.get_by_role("tabpanel", name="Comparar dos fechas")
        resumen["dos_fechas"] = [t.replace("\n", " = ") for t in
                                 panel.locator('[data-testid="stMetric"]').all_inner_texts()]
        pagina.get_by_text("Inicial · original").first.evaluate("e => e.scrollIntoView({block: 'end'})")
        pagina.wait_for_timeout(800)
        pagina.screenshot(path=str(destino / "04_dos_fechas_escenas.png"))

        # Modelo y evaluación
        pagina.get_by_role("tab", name="Modelo y evaluación").click()
        pagina.get_by_text("Paquete:").first.wait_for(timeout=60_000)
        pagina.get_by_text("Paquete:").first.scroll_into_view_if_needed()
        pagina.wait_for_timeout(1500)
        pagina.screenshot(path=str(destino / "05_modelo_y_evaluacion.png"))
        navegador.close()
    return resumen


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    for clave, valores in capturar(sys.argv[1], sys.argv[2]).items():
        print(clave, "→", valores)
