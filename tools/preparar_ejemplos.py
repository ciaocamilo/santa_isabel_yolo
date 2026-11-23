"""Copia a examples/ escenas reales de la caché del generador mensual (solo lectura sobre la caché).

Cada ejemplo incluye el PNG RGB original, calidad.tif, soporte_png.tif y un metadata.json recortado con
fecha, malla, receta, fuente, crédito y hashes. Es lo único que necesita la app en Render para la demo.

    python tools/preparar_ejemplos.py 2024-08 2026-08          # meses del manifiesto (primera = inicial)
"""
import json
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from santa_isabel import cargar_config  # noqa: E402
from santa_isabel.dataset import auditar_dataset, resolver_cache  # noqa: E402
from santa_isabel.escenas import leer_escena, sha256_archivo  # noqa: E402

CAMPOS = ("estado", "nombre_escena", "clave", "perfil", "nivel", "id_asset", "system_index", "coleccion",
          "metodo_qa", "coleccion_qa", "satelite", "tesela", "orbita_relativa", "baseline", "nube_granulo_pct",
          "adquisicion_utc", "grupo_temporal", "malla", "orden_canales", "atribucion")


def preparar(meses, destino=None):
    cfg = cargar_config()
    aud = auditar_dataset(cfg, verificar_hashes=False)
    destino = Path(destino) if destino else RAIZ / "examples"
    destino.mkdir(parents=True, exist_ok=True)
    claves = []
    for mes in meses:
        fila = next((f for f in aud.filas if f["mes"] == mes), None)
        if fila is None:
            raise SystemExit(f"El mes {mes} no está en el manifiesto de {aud.version}.")
        origen = resolver_cache(cfg, fila, aud.resumen.get("id_aoi"))
        if origen is None:
            raise SystemExit(f"No hay coincidencia única en la caché del productor para {fila['clave_escena']} "
                             "(configure dataset.raiz_cache en configs/local.toml).")
        meta = json.loads((origen / "metadata.json").read_text(encoding="utf-8"))
        clave = meta["clave"]
        carpeta = destino / clave
        shutil.rmtree(carpeta, ignore_errors=True)
        (carpeta / "rgb").mkdir(parents=True)
        rel_rgb = meta["rutas"]["rgb"]
        copiados = {rel_rgb: origen / rel_rgb, "calidad.tif": origen / "calidad.tif",
                    "soporte_png.tif": origen / "soporte_png.tif"}
        for rel, src in copiados.items():
            shutil.copyfile(src, carpeta / rel)
        nuevo = {k: meta[k] for k in CAMPOS if k in meta}
        nuevo["receta"] = {"version": meta["receta"]["version"], "huella": meta["receta"]["huella"],
                           "png_rgb": meta["receta"]["detalle"]["png"]["rgb"],
                           "calidad": meta["receta"]["detalle"]["calidad"]}
        nuevo["porcentajes"] = {"local_10m": meta["porcentajes"]["local_10m"]}
        nuevo["rutas"] = {"rgb": rel_rgb, "calidad": "calidad.tif", "soporte_png": "soporte_png.tif"}
        nuevo["archivos"] = {rel: {"sha256": sha256_archivo(carpeta / rel), "bytes": (carpeta / rel).stat().st_size}
                             for rel in copiados}
        for rel in copiados:
            if rel in meta["archivos"] and meta["archivos"][rel]["sha256"] != nuevo["archivos"][rel]["sha256"]:
                raise SystemExit(f"{clave}: el hash de {rel} no coincide con el del productor.")
        nuevo["origen_ejemplo"] = {
            "dataset": aud.version, "mes": fila["mes"], "split": fila["split"],
            "etiqueta_del_dataset": f"{fila['origen']} (sin revisión humana)",
            "copiado_de": "caché del generador mensual (SDD 03), receta " + meta["receta"]["huella"],
            "licencia": "Datos Copernicus Sentinel-2: uso libre, completo y abierto con atribución "
                        "(Reglamento UE 1159/2013)."}
        (carpeta / "metadata.json").write_text(json.dumps(nuevo, ensure_ascii=False, indent=1), encoding="utf-8")
        escena = leer_escena(carpeta)
        if not escena.tiene_qa or not escena.receta["verificada"]:
            raise SystemExit(f"{clave}: el ejemplo copiado no se lee con QA y hash verificados.")
        claves.append(clave)
        print(f"{clave}: {fila['split']} · {escena.fecha:%Y-%m-%d} · {escena.ancho}×{escena.alto} · observable "
              f"{100 * escena.observable.mean():.1f} % · {sum(v['bytes'] for v in nuevo['archivos'].values()) / 1e3:.0f} kB")
    if len(claves) >= 2:
        (destino / "demo.json").write_text(json.dumps({
            "par": claves[:2], "dataset": aud.version,
            "nota": "Par de la demostración de dos fechas (inicial, final). Escenas reales Sentinel-2 L2A del "
                    "Nevado Santa Isabel descargadas con Google Earth Engine por el generador mensual."},
            ensure_ascii=False, indent=1), encoding="utf-8")
    return claves


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    preparar(sys.argv[1:])
