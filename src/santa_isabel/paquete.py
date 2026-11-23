"""Paquete inmutable del modelo: publicación desde el notebook y carga validada para notebook y app.

Estructura de models/<id>/:
    modelo.json            contrato (clase, tarea, variante, receta, entrada, posproceso, backends, dataset…)
    pesos/best.pt          pesos ajustados (backend de referencia, Ultralytics)
    pesos/best.onnx        exportación para el servicio CPU (ONNX Runtime), si existe
    metricas.json, roc_*.csv, *.png   evidencia de la corrida
    SHA256SUMS             hash de cada archivo del paquete
"""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .escenas import sha256_archivo

ESQUEMA = "1.0"
TAREA = "segment"
CLASE = {"id": 0, "nombre": "nieve_hielo_visible"}
VARIANTES = ("rgb", "espectral")
ARCHIVO_ACTIVO = "ACTIVO"


class ErrorPaquete(RuntimeError):
    """Paquete no utilizable. `como_resolver` es el texto que la interfaz muestra al usuario."""

    def __init__(self, mensaje, como_resolver=""):
        super().__init__(mensaje)
        self.como_resolver = como_resolver

    def __str__(self):
        base = super().__str__()
        return f"{base} Cómo resolverlo: {self.como_resolver}" if self.como_resolver else base


@dataclass
class Modelo:
    ruta: Path
    contrato: dict
    backend: object
    metricas: dict = field(default_factory=dict)
    candado: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def id(self) -> str:
        return self.contrato["id"]

    @property
    def variante(self) -> str:
        return self.contrato["variante"]

    @property
    def umbral_operativo(self) -> float:
        return float(self.contrato["posproceso"]["umbral_operativo"])


def backends_instalados() -> dict:
    import importlib.util
    return {"ultralytics": importlib.util.find_spec("ultralytics") is not None
            and importlib.util.find_spec("torch") is not None,
            "onnx": importlib.util.find_spec("onnxruntime") is not None}


def resolver_activo(raiz_models, nombre=None) -> Path:
    """Carpeta del paquete activo: nombre explícito, o el contenido de models/ACTIVO."""
    raiz_models = Path(raiz_models)
    if not nombre:
        puntero = raiz_models / ARCHIVO_ACTIVO
        if not puntero.is_file():
            raise ErrorPaquete(
                f"No hay un modelo publicado: falta {puntero}.",
                "ejecute notebooks/01_entrenar_evaluar.ipynb hasta la celda de exportación; publica el paquete en "
                "models/<id>/ y escribe models/ACTIVO. La aplicación nunca presenta resultados de nieve con pesos "
                "genéricos.")
        nombre = puntero.read_text(encoding="utf-8").strip()
    ruta = Path(nombre) if Path(nombre).is_absolute() else raiz_models / nombre
    if not ruta.is_dir():
        raise ErrorPaquete(f"No existe la carpeta del paquete {ruta}.",
                           "revise models/ACTIVO o la variable SANTA_ISABEL_PAQUETE, o vuelva a exportar el "
                           "paquete desde el notebook.")
    return ruta


def leer_contrato(ruta) -> dict:
    """Lee y valida modelo.json (sin cargar pesos)."""
    ruta = Path(ruta)
    archivo = ruta / "modelo.json"
    if not archivo.is_file():
        raise ErrorPaquete(f"{ruta} no contiene modelo.json.",
                           "la carpeta no es un paquete publicado por el notebook; vuelva a exportarlo.")
    try:
        c = json.loads(archivo.read_text(encoding="utf-8"))
    except Exception as e:
        raise ErrorPaquete(f"modelo.json no es un JSON válido ({e}).", "vuelva a exportar el paquete.") from e
    if c.get("esquema") != ESQUEMA:
        raise ErrorPaquete(f"Versión de esquema {c.get('esquema')!r} no admitida (esta versión lee {ESQUEMA!r}).",
                           "actualice el código de la aplicación o vuelva a exportar el paquete con este código.")
    if c.get("tarea") != TAREA:
        raise ErrorPaquete(f"El paquete declara la tarea {c.get('tarea')!r}; se requiere {TAREA!r}.",
                           "use un paquete de segmentación entrenado con el notebook de este proyecto.")
    if c.get("clase") != CLASE:
        raise ErrorPaquete(f"El paquete declara la clase {c.get('clase')}; se requiere {CLASE}.",
                           "los pesos genéricos (p. ej. COCO) no detectan nieve: use el paquete ajustado.")
    if c.get("variante") not in VARIANTES:
        raise ErrorPaquete(f"Variante {c.get('variante')!r} desconocida.", "vuelva a exportar el paquete.")
    for seccion, campos in {"entrada": ("imgsz",), "posproceso": ("umbral_propuestas", "umbral_operativo",
                                                                  "max_det"),
                            "dataset": ("version",), "referencia": ("origen",)}.items():
        faltan = [k for k in campos if k not in (c.get(seccion) or {})]
        if faltan:
            raise ErrorPaquete(f"modelo.json: en «{seccion}» faltan {', '.join(faltan)}.",
                               "vuelva a exportar el paquete desde el notebook.")
    p = c["posproceso"]
    if not (0 < float(p["umbral_propuestas"]) <= float(p["umbral_operativo"]) <= 1):
        raise ErrorPaquete("modelo.json: se requiere 0 < umbral de propuestas ≤ umbral operativo ≤ 1.",
                           "vuelva a exportar el paquete desde el notebook.")
    if not c.get("backends"):
        raise ErrorPaquete("modelo.json no declara ningún backend.", "vuelva a exportar el paquete.")
    return c


def verificar_hashes(ruta, contrato) -> int:
    """Comprueba el SHA-256 de cada archivo declarado en el contrato. Devuelve cuántos verificó."""
    ruta = Path(ruta)
    n = 0
    for rel, esperado in (contrato.get("archivos") or {}).items():
        archivo = ruta / rel
        if not archivo.is_file():
            raise ErrorPaquete(f"Falta el archivo {rel} del paquete {ruta.name}.",
                               "el paquete está incompleto: vuelva a copiarlo o a exportarlo (si los pesos se "
                               "publican como activo de GitHub Release, descárguelos de la URL fija del README).")
        leido = sha256_archivo(archivo)
        if leido != esperado:
            raise ErrorPaquete(
                f"Hash erróneo en {rel}: esperado {esperado[:12]}…, leído {leido[:12]}….",
                "el archivo se alteró o se descargó incompleto; restaure el paquete publicado. Un paquete es "
                "inmutable: para cambiar pesos o umbral, exporte uno nuevo.")
        n += 1
    return n


def cargar_paquete(ruta, backend=None, dispositivo="cpu") -> Modelo:
    """Valida esquema, tarea, clase, variante, hashes y backend, y deja el modelo listo para `predecir`.

    `backend`: 'ultralytics', 'onnx' o None (el de servicio declarado si está instalado; si no, el otro).
    """
    ruta = Path(ruta)
    contrato = leer_contrato(ruta)
    verificar_hashes(ruta, contrato)
    disponibles, instalados = contrato["backends"], backends_instalados()
    if backend is None:
        orden = [contrato.get("backend_servicio"), contrato.get("backend_referencia"), *disponibles]
        backend = next((b for b in orden if b in disponibles and instalados.get(b)), None)
        if backend is None:
            raise ErrorPaquete(
                f"Ningún backend del paquete ({', '.join(disponibles)}) está instalado en este entorno.",
                "instale las dependencias con «python -m pip install -r requirements-app.txt» (ONNX Runtime) o "
                "use el entorno yolo_gee (Ultralytics).")
    if backend not in disponibles:
        raise ErrorPaquete(f"El paquete no incluye el backend {backend!r} (tiene: {', '.join(disponibles)}).",
                           "elija uno de los backends del paquete o vuelva a exportarlo con ese backend.")
    if not instalados.get(backend):
        raise ErrorPaquete(f"El backend {backend!r} no está instalado en este entorno.",
                           "instale onnxruntime (requirements-app.txt) o use el entorno yolo_gee para Ultralytics.")
    archivo = ruta / disponibles[backend]["archivo"]
    from . import inferencia
    try:
        motor = (inferencia.BackendUltralytics(archivo, dispositivo) if backend == "ultralytics"
                 else inferencia.BackendOnnx(archivo))
    except Exception as e:
        raise ErrorPaquete(f"No se pudieron cargar los pesos {archivo.name} con {backend}: {e}",
                           "compruebe las versiones registradas en modelo.json frente a las instaladas.") from e
    if backend == "onnx" and motor.tam_fijo and motor.tam_fijo != int(contrato["entrada"]["imgsz"]):
        raise ErrorPaquete(f"El ONNX tiene entrada {motor.tam_fijo} y el contrato declara imgsz "
                           f"{contrato['entrada']['imgsz']}.", "vuelva a exportar el paquete.")
    metricas = {}
    if (ruta / "metricas.json").is_file():
        metricas = json.loads((ruta / "metricas.json").read_text(encoding="utf-8"))
    return Modelo(ruta=ruta, contrato=contrato, backend=motor, metricas=metricas)


def modelo_en_memoria(ruta_pesos, cfg, variante="rgb", imgsz=None, umbral_operativo=None, backend="ultralytics",
                      dispositivo="cpu", identificador="provisional") -> Modelo:
    """Modelo provisional (sin publicar) para seleccionar checkpoint, resolución y umbral en validación.

    Usa exactamente el mismo backend y posprocesado que un paquete publicado.
    """
    from . import inferencia
    inf, ent = cfg["inferencia"], cfg["entrenamiento"]
    contrato = {
        "esquema": ESQUEMA, "id": identificador, "tarea": TAREA, "clase": CLASE, "variante": variante,
        "entrada": {"imgsz": int(imgsz or ent["imgsz"]), "tesela_px": int(inf["tesela_px"]),
                    "solape_px": int(inf["solape_px"])},
        "posproceso": {"umbral_propuestas": float(inf["umbral_propuestas"]),
                       "umbral_operativo": float(umbral_operativo or inf["umbral_operativo_inicial"]),
                       "iou_nms": float(inf["iou_nms"]), "max_det": int(inf["max_det"])},
        "backends": {backend: {"archivo": str(ruta_pesos)}}, "provisional": True}
    motor = (inferencia.BackendUltralytics(ruta_pesos, dispositivo) if backend == "ultralytics"
             else inferencia.BackendOnnx(ruta_pesos))
    return Modelo(ruta=Path(ruta_pesos).parent, contrato=contrato, backend=motor)


def publicar_paquete(raiz_models, contrato: dict, archivos: dict, activar=True) -> Path:
    """Crea models/<id>/ con los archivos indicados ({ruta relativa: origen}), sus hashes y modelo.json.

    Un paquete publicado es inmutable: si el identificador ya existe se rechaza.
    """
    raiz_models = Path(raiz_models)
    destino = raiz_models / contrato["id"]
    if destino.exists():
        raise ErrorPaquete(f"Ya existe el paquete {destino.name}: los paquetes son inmutables.",
                           "publique con un identificador nuevo (otra corrida).")
    tmp = raiz_models / f"_tmp_{contrato['id']}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        hashes = {}
        for rel, origen in archivos.items():
            objetivo = tmp / rel
            objetivo.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(origen, (bytes, bytearray)):
                objetivo.write_bytes(origen)
            else:
                shutil.copyfile(origen, objetivo)
            hashes[rel] = sha256_archivo(objetivo)
        contrato = dict(contrato, esquema=ESQUEMA, tarea=TAREA, clase=CLASE, archivos=hashes,
                        creado_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        for nombre, datos in contrato["backends"].items():
            datos["sha256"] = hashes[datos["archivo"]]
        # Se escriben bytes (LF) para que el archivo sea idéntico en cualquier sistema y coincida con su hash.
        texto = json.dumps(contrato, ensure_ascii=False, indent=1).encode("utf-8")
        (tmp / "modelo.json").write_bytes(texto)
        sumas = dict(hashes, **{"modelo.json": hashlib.sha256(texto).hexdigest()})
        (tmp / "SHA256SUMS").write_bytes("".join(f"{h}  {rel}\n" for rel, h in sorted(sumas.items()))
                                         .encode("utf-8"))
        leer_contrato(tmp)
        verificar_hashes(tmp, contrato)
        tmp.rename(destino)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if activar:
        (raiz_models / ARCHIVO_ACTIVO).write_text(contrato["id"] + "\n", encoding="utf-8")
    return destino
