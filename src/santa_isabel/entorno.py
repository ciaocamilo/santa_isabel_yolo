"""Ajuste de PROJ/GDAL antes de importar rasterio.

En Windows es habitual que una variable global PROJ_LIB (p. ej. la de PostgreSQL/PostGIS) apunte a un proj.db
de otra versión; un kernel de Jupyter no ejecuta los scripts de activación de conda, y entonces rasterio no
reconoce ningún código EPSG. Se fijan las rutas del propio entorno (conda) o las que trae el wheel de rasterio.
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import warnings
from pathlib import Path

_HECHO = False


def _primer_directorio(candidatos, archivo=None):
    for c in candidatos:
        if c.is_dir() and (archivo is None or (c / archivo).is_file()):
            return c
    return None


def preparar_entorno() -> dict:
    global _HECHO
    prefijo = Path(sys.prefix)
    paquete = None
    try:
        spec = importlib.util.find_spec("rasterio")
        paquete = Path(spec.origin).parent if spec and spec.origin else None
    except Exception:
        pass
    proj = _primer_directorio([prefijo / "Library/share/proj", prefijo / "share/proj",
                               *([paquete / "proj_data"] if paquete else [])], "proj.db")
    gdal = _primer_directorio([prefijo / "Library/share/gdal", prefijo / "share/gdal",
                               *([paquete / "gdal_data"] if paquete else [])])
    if proj:
        os.environ["PROJ_DATA"] = os.environ["PROJ_LIB"] = str(proj)
    if gdal:
        os.environ["GDAL_DATA"] = str(gdal)
    # NumPy de conda-forge usa MKL con OpenMP de LLVM (libomp.dll) y el wheel pip de torch trae el de Intel
    # (libiomp5md.dll): juntos abortan con «OMP: Error #15». Con la capa secuencial MKL no carga OpenMP.
    # Solo tiene efecto si se fija antes de importar numpy: importe santa_isabel en primer lugar.
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    if os.name == "nt" and not _HECHO:
        dll = [str(prefijo / d) for d in ("Library/bin",) if (prefijo / d).is_dir()]
        if dll:
            os.environ["PATH"] = os.pathsep.join(dll + [p for p in os.environ.get("PATH", "").split(os.pathsep)
                                                        if p not in dll])
            with contextlib.suppress(OSError):
                os.add_dll_directory(dll[0])
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"google\..*")
    _HECHO = True
    return {"PROJ_DATA": os.environ.get("PROJ_DATA"), "GDAL_DATA": os.environ.get("GDAL_DATA"),
            "MKL_THREADING_LAYER": os.environ.get("MKL_THREADING_LAYER")}
