"""Prueba de servicio (hito F): arranque con PORT configurable, health check, dos sesiones aisladas, subida real
y demostración de dos fechas, con tiempos y memoria. Solo necesita requirements-app.txt.

    PORT=10000 python tools/prueba_servicio.py            # Linux / Render
    $env:PORT=10000; python tools/prueba_servicio.py       # Windows

Imprime un JSON con las mediciones. Sale con código distinto de cero si algo no se cumple.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))


def memoria_mib(pico=True):
    import psutil
    info = psutil.Process().memory_info()
    if pico and hasattr(info, "peak_wset"):
        return round(info.peak_wset / 2**20, 1)
    if pico:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    return round(info.rss / 2**20, 1)


def esperar(url, segundos=120):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < segundos:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return time.perf_counter() - t0, r.read().decode()
        except Exception:
            time.sleep(0.25)
    raise RuntimeError(f"{url} no respondió 200 en {segundos} s")


def prueba_servidor(puerto) -> dict:
    """Arranca el servicio con el mismo comando que render.yaml y comprueba el health check y la portada."""
    import psutil
    orden = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port",
             str(puerto), "--server.headless", "true"]
    t0 = time.perf_counter()
    proceso = subprocess.Popen(orden, cwd=RAIZ, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        _, cuerpo = esperar(f"http://127.0.0.1:{puerto}/_stcore/health")
        arranque = time.perf_counter() - t0
        _, portada = esperar(f"http://127.0.0.1:{puerto}/")
        hijo = psutil.Process(proceso.pid)
        rss = sum(p.memory_info().rss for p in [hijo, *hijo.children(recursive=True)]) / 2**20
        return {"puerto": puerto, "health": cuerpo.strip(), "arranque_en_frio_s": round(arranque, 2),
                "portada_http_200": "<title>" in portada.lower() or "streamlit" in portada.lower(),
                "rss_servidor_en_reposo_mib": round(rss, 1)}
    finally:
        proceso.terminate()
        try:
            proceso.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proceso.kill()


def valor(texto) -> float:
    return float(texto.replace("\u202f", "").split()[0].replace(",", ".").replace("+", "")) if texto != "N/D" else None


def prueba_sesiones() -> dict:
    """Dos sesiones aisladas, subida real de una escena y demostración de dos fechas, con el probador de
    Streamlit (ejecuta app.py con el mismo código y modelo que el servicio)."""
    from streamlit.testing.v1 import AppTest
    app = str(RAIZ / "app.py")
    demo = json.loads((RAIZ / "examples" / "demo.json").read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    a = AppTest.from_file(app, default_timeout=600).run()
    primera_carga = time.perf_counter() - t0
    assert not a.exception and not a.error, [e.value for e in a.error]
    b = AppTest.from_file(app, default_timeout=600).run()

    # Sesión A: subida real de los archivos de una escena (paquete del generador, archivos sueltos).
    carpeta = RAIZ / "examples" / demo["par"][0]
    archivos = [carpeta / "metadata.json", *sorted((carpeta / "rgb").glob("*.png")), carpeta / "calidad.tif",
                carpeta / "soporte_png.tif"]
    a.file_uploader(key="carga_una").set_value([(f.name, f.read_bytes(), "application/octet-stream")
                                                for f in archivos])
    a.run()
    a.button(key="btn_leer_una").click().run()
    assert not a.exception and not a.error, [e.value for e in a.error]
    t0 = time.perf_counter()
    a.button(key="btn_analizar_una").click().run()
    una = time.perf_counter() - t0
    assert not a.exception and not a.error, [e.value for e in a.error]
    m_a = {m.label: m.value for m in a.metric}
    assert a.session_state["una"]["origen"] == "carga" and a.session_state["una"]["escena"].tiene_qa

    # Sesión B: no ve nada de A; ejecuta la demostración de dos fechas.
    b.run()
    assert "una" not in b.session_state and not b.metric
    assert a.session_state["tmp"] != b.session_state["tmp"] if "tmp" in b.session_state else True
    b.button(key="btn_ejemplo_dos").click().run()
    t0 = time.perf_counter()
    b.button(key="btn_comparar").click().run()
    dos = time.perf_counter() - t0
    assert not b.exception and not b.error, [e.value for e in b.error]
    m_b = {m.label: m.value for m in b.metric}
    assert "dos" not in a.session_state                      # A tampoco ve la comparación de B
    return {"primera_carga_con_modelo_s": round(primera_carga, 2), "latencia_una_escena_s": round(una, 2),
            "latencia_dos_escenas_s": round(dos, 2), "sesiones_aisladas": True,
            "subida_real": {"archivos": [f.name for f in archivos], "superficie_ha": valor(m_a["Superficie"]),
                            "cobertura_observable_ha": valor(m_a["Cobertura observable"])},
            "dos_fechas": {k: valor(v) for k, v in m_b.items()},
            "temporales_por_sesion": a.session_state["tmp"] != b.session_state.get("tmp", "")}


if __name__ == "__main__":
    import platform
    import santa_isabel  # noqa: F401
    puerto = int(os.environ.get("PORT", "10000"))
    salida = {"plataforma": platform.platform(), "python": platform.python_version(),
              "paquete": (RAIZ / "models" / "ACTIVO").read_text(encoding="utf-8").strip(),
              "backend": os.environ.get("SANTA_ISABEL_BACKEND") or "automático"}
    salida["servidor"] = prueba_servidor(puerto)
    salida["sesiones"] = prueba_sesiones()
    salida["torch_cargado"] = "torch" in sys.modules
    salida["memoria_pico_proceso_de_prueba_mib"] = memoria_mib()
    print(json.dumps(salida, ensure_ascii=False, indent=1))
