"""Raíz del proyecto y configuración TOML, independientes del directorio de trabajo."""
from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path

MARCADOR = Path("configs") / "experimento.toml"


def raiz_proyecto(inicio=None) -> Path:
    """Carpeta que contiene configs/experimento.toml (SANTA_ISABEL_RAIZ tiene prioridad)."""
    env = os.environ.get("SANTA_ISABEL_RAIZ")
    if env:
        raiz = Path(env).expanduser().resolve()
        if not (raiz / MARCADOR).is_file():
            raise FileNotFoundError(f"SANTA_ISABEL_RAIZ={raiz} no contiene {MARCADOR.as_posix()}")
        return raiz
    candidatos = ([Path(inicio).resolve()] if inicio else []) + [Path(__file__).resolve(), Path.cwd().resolve()]
    for c in candidatos:
        for p in (c, *c.parents):
            if (p / MARCADOR).is_file():
                return p
    raise FileNotFoundError(
        f"No se encontró {MARCADOR.as_posix()} subiendo desde {candidatos[-1]}. Ejecute dentro del proyecto "
        "santa_isabel_yolo o defina la variable SANTA_ISABEL_RAIZ.")


def _mezclar(base: dict, extra: dict) -> dict:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _mezclar(base[k], v)
        else:
            base[k] = v
    return base


def cargar_config(ruta=None, raiz=None) -> dict:
    """Lee experimento.toml, aplica configs/local.toml si existe y las variables de entorno admitidas."""
    raiz = Path(raiz).resolve() if raiz else raiz_proyecto(ruta)
    ruta = Path(ruta) if ruta else raiz / MARCADOR
    with open(ruta, "rb") as f:
        cfg = tomllib.load(f)
    local = raiz / "configs" / "local.toml"
    if local.is_file():
        with open(local, "rb") as f:
            _mezclar(cfg, tomllib.load(f))
    if os.environ.get("SANTA_ISABEL_CACHE"):
        cfg.setdefault("dataset", {})["raiz_cache"] = os.environ["SANTA_ISABEL_CACHE"]
    if os.environ.get("SANTA_ISABEL_DATASET_VERSION"):
        cfg.setdefault("dataset", {})["version"] = os.environ["SANTA_ISABEL_DATASET_VERSION"]
    if os.environ.get("SANTA_ISABEL_PAQUETE"):
        cfg.setdefault("app", {})["paquete"] = os.environ["SANTA_ISABEL_PAQUETE"]
    cfg["_raiz"] = str(raiz)
    cfg["_origen"] = str(ruta)
    return cfg


def resolver(cfg: dict, relativa) -> Path:
    """Ruta absoluta: las relativas cuelgan de la raíz del proyecto."""
    p = Path(str(relativa)).expanduser()
    return p if p.is_absolute() else Path(cfg["_raiz"]) / p


def sin_privados(cfg: dict) -> dict:
    """Copia serializable de la configuración (sin claves internas)."""
    return {k: copy.deepcopy(v) for k, v in cfg.items() if not k.startswith("_")}
