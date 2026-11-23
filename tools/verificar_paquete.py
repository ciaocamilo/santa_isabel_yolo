"""Reabre un paquete publicado en un proceso limpio y en CPU, con el mismo cargador que usa Streamlit.

Para cada backend instalado analiza las escenas de ejemplo y, si hay dos, las compara. Imprime un JSON con
áreas, tiempos y memoria pico del proceso. Sirve como comprobación de inferencia en CPU y para medir el
consumo del servicio (también en Linux).

    python tools/verificar_paquete.py [carpeta_paquete] [--ejemplos carpeta] [--backends onnx,ultralytics]
"""
import argparse
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import santa_isabel  # noqa: E402,F401

from santa_isabel import cargar_config  # noqa: E402
from santa_isabel.escenas import leer_escena  # noqa: E402
from santa_isabel.flujo import analizar_escena, comparar_fechas  # noqa: E402
from santa_isabel.paquete import backends_instalados, cargar_paquete, leer_contrato, resolver_activo  # noqa: E402


def memoria_pico_mib():
    try:
        import psutil
        info = psutil.Process().memory_info()
        return round(getattr(info, "peak_wset", info.rss) / 2**20, 1)
    except ImportError:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def verificar(ruta, carpeta_ejemplos, backends=None) -> dict:
    cfg = cargar_config(raiz=RAIZ)
    contrato = leer_contrato(ruta)
    instalados = backends_instalados()
    backends = backends or [b for b in contrato["backends"] if instalados.get(b)]
    demo = {}
    if (carpeta_ejemplos / "demo.json").is_file():
        demo = json.loads((carpeta_ejemplos / "demo.json").read_text(encoding="utf-8"))
    claves = demo.get("par") or sorted(p.parent.name for p in carpeta_ejemplos.glob("*/metadata.json"))
    salida = {"paquete": contrato["id"], "python": sys.version.split()[0], "plataforma": sys.platform,
              "torch_importado_al_inicio": "torch" in sys.modules, "backends": {}}
    for nombre in backends:
        t0 = time.perf_counter()
        modelo = cargar_paquete(ruta, backend=nombre, dispositivo="cpu")
        carga = time.perf_counter() - t0
        analisis, escenas = [], []
        for clave in claves:
            t = time.perf_counter()
            a = analizar_escena(leer_escena(carpeta_ejemplos / clave), modelo, cfg)
            analisis.append(a)
            escenas.append({"clave": clave, "fecha": f"{a.escena.fecha:%Y-%m-%d}", "estado": a.medicion.estado,
                            "area_nieve_ha": a.medicion.area_nieve_ha, "pixeles_nieve": a.medicion.pixeles_nieve,
                            "parches": a.prediccion.instancias_operativas,
                            "propuestas": len(a.prediccion.instancias),
                            "segundos_lectura_e_inferencia": round(time.perf_counter() - t, 3),
                            "segundos_inferencia": round(a.prediccion.segundos, 3)})
        datos = {"dispositivo": str(modelo.backend.dispositivo), "segundos_carga_modelo": round(carga, 2),
                 "escenas": escenas, "torch_cargado": "torch" in sys.modules}
        if len(analisis) >= 2:
            t = time.perf_counter()
            c = comparar_fechas(analisis[0], analisis[1], cfg)
            datos["comparacion"] = {k: v for k, v in c.como_fila().items()
                                    if k.endswith(("_ha", "_pct_del_inicial", "_pct_del_aoi")) or k == "estado"}
            datos["comparacion"]["segundos"] = round(time.perf_counter() - t, 3)
        datos["memoria_pico_mib"] = memoria_pico_mib()
        salida["backends"][nombre] = datos
    return salida


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("paquete", nargs="?", help="carpeta del paquete (por defecto, models/ACTIVO)")
    p.add_argument("--ejemplos", default=str(RAIZ / "examples"))
    p.add_argument("--backends", default="")
    a = p.parse_args()
    ruta = Path(a.paquete) if a.paquete else resolver_activo(RAIZ / "models")
    print(json.dumps(verificar(ruta, Path(a.ejemplos), [b for b in a.backends.split(",") if b] or None),
                     ensure_ascii=False, indent=1))
