"""Deja el entorno de entrenamiento listo de forma reproducible (ejecutar con el Python de yolo_gee).

1. Escribe en site-packages un archivo .pth que fija MKL_THREADING_LAYER=SEQUENTIAL al arrancar cualquier
   proceso de este intérprete. Motivo: NumPy de conda-forge usa MKL con el OpenMP de LLVM (libomp.dll) y el wheel
   pip de torch trae el de Intel (libiomp5md.dll); juntos abortan con «OMP: Error #15». Con la capa secuencial,
   MKL no carga ningún OpenMP. Debe fijarse antes de importar NumPy, y Streamlit o Jupyter lo importan antes
   que el código del proyecto: por eso se fija en el entorno y no solo en el paquete.
2. Registra el kernel «Python (yolo_gee)» con las variables PROJ/GDAL del propio entorno (un kernel no ejecuta
   los scripts de activación de conda, y una PROJ_LIB global de otra instalación rompe los códigos EPSG).

    python tools/configurar_entorno.py
"""
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

NOMBRE_KERNEL = "yolo_gee"
PTH = "santa_isabel_mkl.pth"


def main():
    prefijo = Path(sys.prefix)
    sitio = Path(sysconfig.get_paths()["purelib"])
    (sitio / PTH).write_text('import os; os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")\n',
                             encoding="ascii")
    print(f"Escrito {sitio / PTH}")
    subprocess.run([sys.executable, "-m", "ipykernel", "install", "--user", "--name", NOMBRE_KERNEL,
                    "--display-name", "Python (yolo_gee)"], check=True)
    datos = subprocess.run([sys.executable, "-m", "jupyter", "kernelspec", "list", "--json"], check=True,
                           capture_output=True, text=True).stdout
    carpeta = Path(json.loads(datos)["kernelspecs"][NOMBRE_KERNEL]["resource_dir"])
    spec = json.loads((carpeta / "kernel.json").read_text(encoding="utf-8"))
    entorno = {"MKL_THREADING_LAYER": "SEQUENTIAL"}
    for var, subdirs in {"PROJ_DATA": ("Library/share/proj", "share/proj"),
                         "GDAL_DATA": ("Library/share/gdal", "share/gdal")}.items():
        ruta = next((prefijo / s for s in subdirs if (prefijo / s).is_dir()), None)
        if ruta:
            entorno[var] = str(ruta)
            if var == "PROJ_DATA":
                entorno["PROJ_LIB"] = str(ruta)
    spec["env"] = {**spec.get("env", {}), **entorno}
    (carpeta / "kernel.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    print(f"Kernel registrado en {carpeta} con env = {entorno}")
    prueba = subprocess.run([sys.executable, "-c",
                             "import os, numpy; print(os.environ.get('MKL_THREADING_LAYER'))"],
                            capture_output=True, text=True, env={k: v for k, v in os.environ.items()
                                                                 if k != "MKL_THREADING_LAYER"})
    print("Comprobación en un proceso nuevo: MKL_THREADING_LAYER =", prueba.stdout.strip())


if __name__ == "__main__":
    main()
