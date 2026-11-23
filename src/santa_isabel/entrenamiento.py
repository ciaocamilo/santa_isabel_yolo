"""Entrenamiento con Ultralytics en la GPU (solo entorno de entrenamiento; la app no importa este módulo)."""
from __future__ import annotations

import contextlib
import json
import os
import platform
import sys
import time
from pathlib import Path

from .config import resolver

AUMENTOS = ("mosaic", "mixup", "copy_paste", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
            "fliplr", "flipud")


def diagnostico_gpu(exigir=True) -> dict:
    """Intérprete, versiones, CUDA, GPU y VRAM; ejecuta una operación en cuda:0 y sincroniza.

    Si CUDA no está disponible y `exigir` es verdadero, detiene con el diagnóstico de kernel, wheel y driver.
    """
    import torch
    import torchvision
    import ultralytics
    info = {"ejecutable": sys.executable, "python": platform.python_version(), "so": platform.platform(),
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "ultralytics": ultralytics.__version__, "cuda_compilada": torch.version.cuda,
            "cuda_disponible": bool(torch.cuda.is_available())}
    if not info["cuda_disponible"]:
        if exigir:
            raise RuntimeError(
                "CUDA no está disponible en este kernel.\n"
                f"  Intérprete: {sys.executable}\n  torch: {torch.__version__} (CUDA compilada: "
                f"{torch.version.cuda})\n"
                "  Compruebe: (1) que el kernel es «Python (yolo_gee)»; (2) que el wheel es +cuXXX y no +cpu "
                "(pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132); "
                "(3) el driver con nvidia-smi.")
        return info
    p = torch.cuda.get_device_properties(0)
    x = torch.randn(1024, 1024, device="cuda:0")
    y = (x @ x).sum()
    torch.cuda.synchronize()
    libre, total = torch.cuda.mem_get_info(0)
    info.update(gpu=p.name, vram_total_mib=total // 2**20, vram_libre_mib=libre // 2**20,
                cudnn=torch.backends.cudnn.version(), operacion_cuda=str(y.device), operacion_ok=bool(y == y))
    return info


@contextlib.contextmanager
def _en_directorio(carpeta):
    """Las descargas auxiliares de Ultralytics (pesos base, comprobación AMP) caen en esta carpeta."""
    previo = Path.cwd()
    Path(carpeta).mkdir(parents=True, exist_ok=True)
    os.chdir(carpeta)
    try:
        yield
    finally:
        os.chdir(previo)


def carpeta_base(cfg) -> Path:
    return resolver(cfg, "runs") / "_base"


def comprobar_lote(cfg, yaml_local, split="train", imgsz=None, lote=None) -> dict:
    """Carga un lote real con el cargador de segmentación de Ultralytics (imágenes, máscaras e instancias)."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import DEFAULT_CFG
    e = cfg["entrenamiento"]
    imgsz, lote = int(imgsz or e["imgsz"]), int(lote or e["lote"])
    with _en_directorio(carpeta_base(cfg)):
        datos = check_det_dataset(str(yaml_local))
        args = get_cfg(DEFAULT_CFG, {"task": "segment", "imgsz": imgsz, "workers": 0, "cache": False,
                                     "plots": False, "mask_ratio": int(e["mask_ratio"]),
                                     "overlap_mask": bool(e["overlap_mask"])})
        ds = build_yolo_dataset(args, datos[split], lote, datos, mode="val", rect=False, stride=32)
        n_inst = sum(len(lb.get("segments") or []) for lb in ds.labels)
        muestra = ds.collate_fn([ds[i] for i in range(min(lote, len(ds)))])
    if "masks" not in muestra:
        raise RuntimeError(f"El lote de {split} no contiene máscaras de segmentación.")
    return {"split": split, "imagenes": len(ds), "instancias": n_inst, "lote_img": list(muestra["img"].shape),
            "lote_masks": list(muestra["masks"].shape), "dtype_img": str(muestra["img"].dtype),
            "instancias_en_lote": int(len(muestra["cls"]))}


def argumentos(cfg, yaml_local, carpeta_corrida, nombre, epocas=None, paciencia=None, extra=None) -> dict:
    e = cfg["entrenamiento"]
    args = {"data": str(yaml_local), "task": "segment", "epochs": int(epocas or e["epocas"]),
            "patience": int(e["paciencia"] if paciencia is None else paciencia), "batch": int(e["lote"]), "imgsz": int(e["imgsz"]),
            "device": int(e["device"]), "workers": int(e["workers"]), "seed": int(cfg["proyecto"]["semilla"]),
            "deterministic": True, "mask_ratio": int(e["mask_ratio"]),
            "overlap_mask": bool(e["overlap_mask"]), "project": str(Path(carpeta_corrida).resolve()),
            "name": nombre, "exist_ok": True, "plots": True, "val": True, "cache": False, "verbose": False}
    args.update({k: float(e[k]) for k in AUMENTOS})
    args.update(extra or {})
    return args


def entrenar(cfg, yaml_local, carpeta_corrida, nombre, epocas=None, paciencia=None, extra=None,
             silencioso=False) -> dict:
    """Afina el modelo base preentrenado con device=0 explícito y registra tiempos y pico real de memoria.

    `silencioso` reduce el registro de Ultralytics a avisos (las curvas quedan en results.csv)."""
    import logging
    import torch
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    diagnostico_gpu(exigir=True)
    args = argumentos(cfg, yaml_local, carpeta_corrida, nombre, epocas, paciencia, extra)
    base = carpeta_base(cfg)
    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.time()
    nivel = LOGGER.level
    try:
        if silencioso:
            LOGGER.setLevel(logging.WARNING)
        with _en_directorio(base):
            modelo = YOLO(cfg["entrenamiento"]["modelo_base"])
            modelo.train(**args)
    finally:
        LOGGER.setLevel(nivel)
    torch.cuda.synchronize()
    carpeta = Path(modelo.trainer.save_dir)
    registro = {
        "nombre": nombre, "carpeta": str(carpeta), "best": str(carpeta / "weights" / "best.pt"),
        "last": str(carpeta / "weights" / "last.pt"), "segundos": round(time.time() - t0, 1),
        "dispositivo": str(modelo.trainer.device), "gpu": torch.cuda.get_device_name(0),
        "pico_asignado_mib": torch.cuda.max_memory_allocated(0) // 2**20,
        "pico_reservado_mib": torch.cuda.max_memory_reserved(0) // 2**20,
        "epocas_pedidas": args["epochs"], "epocas_ejecutadas": int(modelo.trainer.epoch) + 1,
        "mejor_epoca_fitness": getattr(modelo.trainer.stopper, "best_epoch", None),
        "criterio_best": "fitness interno de Ultralytics sobre validación (combinación ponderada de mAP de cajas "
                         "y máscaras; ver results.csv)",
        "argumentos": {k: v for k, v in vars(modelo.trainer.args).items()
                       if isinstance(v, (int, float, str, bool, type(None)))},
        "modelo_base": cfg["entrenamiento"]["modelo_base"]}
    (carpeta / "registro_entrenamiento.json").write_text(json.dumps(registro, ensure_ascii=False, indent=1),
                                                         encoding="utf-8")
    return registro


def checkpoints_candidatos(carpetas: dict, destino) -> list:
    """Copias finales de los checkpoints a comparar: best.pt y last.pt tal cual, y los periódicos (epochN.pt,
    con optimizador) convertidos al mismo formato final. Así lo que se evalúa es exactamente lo que se empaqueta."""
    import logging
    import shutil
    from ultralytics.utils import LOGGER
    from ultralytics.utils.torch_utils import strip_optimizer
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    salida, nivel = [], LOGGER.level
    LOGGER.setLevel(logging.WARNING)
    try:
        for corrida, carpeta in carpetas.items():
            pesos = sorted((Path(carpeta) / "weights").glob("*.pt"), key=lambda p: (len(p.stem), p.stem))
            for pt in pesos:
                copia = destino / f"{corrida}__{pt.name}"
                if pt.name in ("best.pt", "last.pt"):
                    shutil.copyfile(pt, copia)
                else:
                    strip_optimizer(pt, str(copia))
                salida.append({"checkpoint": f"{corrida}/{pt.name}", "ruta": copia})
    finally:
        LOGGER.setLevel(nivel)
    return salida


def validar_instancias(cfg, pesos, yaml_local, split, carpeta_corrida, nombre, imgsz=None) -> dict:
    """mAP50 y mAP50–95 de cajas y máscaras de instancia con el validador de Ultralytics."""
    from ultralytics import YOLO
    e = cfg["entrenamiento"]
    with _en_directorio(carpeta_base(cfg)):
        m = YOLO(str(pesos)).val(data=str(yaml_local), split=split, imgsz=int(imgsz or e["imgsz"]), batch=1,
                                 device=int(e["device"]), workers=0, plots=False, verbose=False,
                                 project=str(Path(carpeta_corrida).resolve()), name=nombre, exist_ok=True)
    return {"split": split, "imgsz": int(imgsz or e["imgsz"]),
            "caja_map50": float(m.box.map50), "caja_map50_95": float(m.box.map),
            "mascara_map50": float(m.seg.map50), "mascara_map50_95": float(m.seg.map)}


def exportar_onnx(cfg, pesos, imgsz, destino) -> dict:
    """Exporta a ONNX (forma fija imgsz×imgsz) desde el entorno de entrenamiento.

    El modelo ajustado predice con la rama con NMS (end2end = False). Se exporta esa misma rama con el NMS
    embebido (`nms=True`) y los umbrales del contrato, de modo que la salida (1, max_det, 6 + 32) contiene las
    mismas detecciones que la predicción de referencia con los pesos .pt.
    """
    import shutil
    from ultralytics import YOLO
    inf = cfg["inferencia"]
    with _en_directorio(carpeta_base(cfg)):
        ruta = YOLO(str(pesos)).export(format="onnx", imgsz=int(imgsz), dynamic=False, simplify=True,
                                       device="cpu", nms=True, conf=float(inf["umbral_propuestas"]),
                                       iou=float(inf["iou_nms"]), max_det=int(inf["max_det"]))
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ruta, destino)
    import onnx
    import onnxruntime
    modelo = onnx.load(str(destino))
    return {"archivo": str(destino), "imgsz": int(imgsz), "opset": int(modelo.opset_import[0].version),
            "onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__,
            "salidas": [o.name for o in modelo.graph.output]}
