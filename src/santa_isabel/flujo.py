"""Flujos compartidos por el notebook y la app: análisis de una escena, comparación de dos fechas y
evaluación de un split con el mismo posprocesado que usa el servicio."""
from __future__ import annotations

import numpy as np

from . import areas, dataset, evaluacion
from .escenas import Escena, leer_escena
from .inferencia import Prediccion, predecir, preparar_entrada

SOPORTES = {
    "observable": "píxel observable (calidad = 2 y soporte del productor)",
    "imagen": "concordancia con pseudoetiquetas sobre toda la imagen, sin control de observabilidad",
    "observable_sin_ambiguos": "píxel observable excluyendo componentes NDSI ambiguos (referencia incierta)",
}


def analizar_escena(escena: Escena, modelo, cfg=None, umbral_operativo=None) -> areas.Analisis:
    """leer_escena → preparar_entrada → predecir → medir. Es el único camino de inferencia de notebook y app."""
    imagen = preparar_entrada(escena, modelo.contrato)
    prediccion = predecir(imagen, modelo, umbral_operativo)
    return areas.Analisis(escena, prediccion, areas.medir(escena, prediccion, cfg))


def comparar_fechas(analisis_a, analisis_b, cfg=None) -> areas.Cambio:
    """Las dos escenas deben haberse analizado con el mismo paquete y la misma receta."""
    return areas.comparar(analisis_a, analisis_b, cfg)


class Protocolo:
    """Registro del uso de los splits para garantizar que test no interviene en la selección.

    Validación puede usarse cuantas veces haga falta. Test solo puede evaluarse una vez y después de
    `congelar` la configuración (checkpoint, resolución y umbral); tras usar test no se admite otra selección.
    """

    def __init__(self):
        self.eventos = []
        self.congelado = None

    def usar(self, split, proposito=""):
        if split == "test":
            if self.congelado is None:
                raise RuntimeError("Protocolo: test no puede evaluarse antes de congelar la configuración "
                                   "elegida con validación.")
            if any(e["split"] == "test" for e in self.eventos):
                raise RuntimeError("Protocolo: la configuración congelada ya se aplicó una vez al test.")
        elif self.usos_test:
            if proposito.startswith("seleccion"):
                raise RuntimeError("Protocolo: no se admite una nueva selección después de evaluar el test.")
        self.eventos.append({"split": split, "proposito": proposito})

    def congelar(self, configuracion: dict):
        if self.usos_test:
            raise RuntimeError("Protocolo: no se puede congelar una configuración después de evaluar el test.")
        if self.congelado is not None:
            raise RuntimeError("Protocolo: la configuración ya estaba congelada.")
        self.congelado = dict(configuracion)

    @property
    def usos_test(self) -> int:
        return sum(e["split"] == "test" for e in self.eventos)

    def resumen(self) -> dict:
        return {"eventos": list(self.eventos), "configuracion_congelada": self.congelado,
                "evaluaciones_test": self.usos_test,
                "test_intervino_en_seleccion": False if self.congelado is not None else None}


# --- Evaluación de un split ------------------------------------------------------------------------------------------
def inferir_split(cfg, auditoria, modelo, split, con_cache=True, protocolo=None, proposito="") -> list:
    """Una inferencia por imagen del split. Guarda puntuación, referencia rasterizada y, si la caché del
    productor está disponible, observabilidad, componentes ambiguos y NDSI."""
    if protocolo is not None:
        protocolo.usar(split, proposito)
    ancho, alto = int(cfg["dataset"]["ancho"]), int(cfg["dataset"]["alto"])
    variante = modelo.contrato["variante"]
    elementos = []
    for fila in (f for f in auditoria.filas if f["split"] == split):
        carpeta = dataset.resolver_cache(cfg, fila, auditoria.resumen.get("id_aoi")) if con_cache else None
        if carpeta is not None:
            escena = leer_escena(carpeta)
            capas = dataset.capas_cache(carpeta, fila.get("reglas_huella"))
        else:
            from PIL import Image
            with Image.open(fila[f"_imagen_{variante}"]) as im:
                rgb = np.asarray(im.convert("RGB"))
            escena = Escena(clave=fila["clave_escena"], tipo="imagen_rgb", alto=alto, ancho=ancho,
                            fecha=fila.get("_fecha"), perfil=fila["perfil"], rgb8=rgb,
                            extension=np.ones((alto, ancho), bool))
            capas = None
        if variante == "rgb":
            imagen = preparar_entrada(escena, modelo.contrato)
        else:
            from PIL import Image
            with Image.open(fila["_imagen_espectral"]) as im:
                imagen = np.ascontiguousarray(np.asarray(im.convert("RGB")))
        pred = predecir(imagen, modelo)
        elementos.append({
            "fila": fila, "nombre": fila["nombre"], "mes": fila["mes"], "perfil": fila["perfil"],
            "origen": fila["origen"], "escena": escena, "prediccion": pred, "puntuacion": pred.puntuacion,
            "referencia": dataset.mascara_referencia(fila, ancho, alto),
            "observable": None if capas is None else capas["observable"],
            "ambiguos": None if capas is None else capas["ambiguos"],
            "ndsi": None if capas is None else capas["ndsi"],
            "soporte": None if capas is None else capas["soporte"],
            "umbral_ndsi": None if capas is None else capas.get("umbral_ndsi"),
            "ha_por_celda": escena.georef.ha_por_celda if escena.georef else None,
            "instancias_propuestas": len(pred.instancias), "tope_alcanzado": pred.tope_alcanzado})
    return elementos


def soporte_valido(elemento, soporte) -> np.ndarray | None:
    if soporte == "imagen":
        return np.ones(elemento["referencia"].shape, bool)
    if elemento["observable"] is None:
        return None
    if soporte == "observable":
        return elemento["observable"]
    if soporte == "observable_sin_ambiguos":
        if elemento["ambiguos"] is None:
            return None
        return elemento["observable"] & ~elemento["ambiguos"]
    raise ValueError(f"Soporte desconocido: {soporte!r}")


def acumulador(elementos, soporte="observable", clave="puntuacion") -> evaluacion.AcumuladorROC:
    acu = evaluacion.AcumuladorROC()
    for e in elementos:
        valido = soporte_valido(e, soporte)
        if valido is None:
            raise ValueError(f"{e['nombre']}: sin QA no se puede evaluar sobre «{soporte}» (use 'imagen').")
        acu.agregar(e["referencia"], e[clave], valido)
    return acu


def elegir_umbral(elementos, soporte="observable", minimo=None, por_defecto=0.25, margen=0.005) -> dict:
    """Umbral operativo que maximiza F1 sobre el conjunto (validación); empates → umbral mayor.

    El máximo de F1 se alcanza en una puntuación observada s, que es la confianza exacta de una instancia.
    Entre backends (GPU, CPU, ONNX) las confianzas difieren del orden de 1e-3, así que usar s como umbral
    haría que esa instancia entrara o no según el backend. El umbral operativo se retrocede `margen` (o la
    mitad de la distancia a la siguiente puntuación inferior, si es menor): la máscara y el F1 en validación
    son idénticos a los de s. Se devuelven ambos valores.
    """
    minimo = elementos[0]["prediccion"].umbral_propuestas if minimo is None else minimo
    acu = acumulador(elementos, soporte)
    # Solo son candidatos los valores ≥ umbral de propuestas (los demás píxeles tienen puntuación 0).
    mejor = acu.mejor_umbral_f1(minimo=float(np.nextafter(np.float32(minimo), np.float32(0))))
    if mejor is None:
        return {"umbral": float(por_defecto), "umbral_f1": None, "f1": None, "soporte": soporte,
                "motivo": "sin propuestas o sin positivos en validación: se conserva el umbral inicial"}
    s_f1 = mejor["umbral"]
    puntuaciones = acu.tabla()[0]
    inferiores = puntuaciones[puntuaciones < np.float32(s_f1)]
    s_inferior = max(float(inferiores.max()) if len(inferiores) else 0.0, 0.0)
    retroceso = min(float(margen), (s_f1 - s_inferior) / 2.0)
    operativo = max(s_f1 - retroceso, float(minimo))
    return dict(mejor, umbral=float(operativo), umbral_f1=float(s_f1), margen_numerico=float(s_f1 - operativo),
                soporte=soporte, motivo="máximo F1 en validación (empates → umbral mayor), con un retroceso "
                                        "numérico que no cambia la máscara de validación")


def metricas_split(elementos, umbral, soporte="observable", clave="puntuacion") -> dict:
    """Métricas por escena y acumuladas sobre todos los píxeles válidos, con ROC/AUC y errores de superficie."""
    por_escena, errores = [], []
    for e in elementos:
        valido = soporte_valido(e, soporte)
        if valido is None:
            raise ValueError(f"{e['nombre']}: sin QA no se puede evaluar sobre «{soporte}» (use 'imagen').")
        m = evaluacion.evaluar_escena(e["referencia"], e[clave], umbral, valido, e["ha_por_celda"])
        nd = m.pop("nd")
        fila = {"nombre": e["nombre"], "mes": e["mes"], "perfil": e["perfil"], "referencia": e["origen"],
                "soporte": soporte, **m, "nd": "; ".join(f"{k}: {v}" for k, v in nd.items()),
                "instancias_propuestas": e["instancias_propuestas"], "tope_alcanzado": e["tope_alcanzado"]}
        por_escena.append(fila)
        errores.append(fila)
    acu = acumulador(elementos, soporte, clave)
    c = acu.confusion_en(umbral)
    agregado = evaluacion.metricas_binarias(c["tp"], c["fp"], c["fn"], c["tn"])
    roc = acu.curva_roc()
    agregado.update(umbral=float(umbral), soporte=soporte, descripcion_soporte=SOPORTES[soporte],
                    n_escenas=len(elementos), auc=roc["auc"], roc_motivo=roc["motivo"],
                    **evaluacion.resumen_areas(errores))
    return {"por_escena": por_escena, "agregado": agregado, "roc": roc, "pr": acu.precision_recall(),
            "acumulador": acu}


def puntuacion_ndsi(elemento) -> np.ndarray:
    """NDSI como puntuación del comparador (−1 donde no hay índice válido)."""
    return np.where(np.isfinite(elemento["ndsi"]) & elemento["soporte"], elemento["ndsi"], -1.0).astype(np.float32)


def comparador_ndsi(elementos, soporte="observable") -> dict | None:
    """Método de referencia del productor sin reglas por componente: NDSI ≥ umbral sobre el soporte.

    Las pseudoetiquetas se generan con ese mismo índice: esta concordancia es circular y no demuestra
    superioridad de ningún método.
    """
    if any(e["ndsi"] is None for e in elementos):
        return None
    umbral = elementos[0]["umbral_ndsi"] or 0.4
    copia = [dict(e, puntuacion_ndsi=puntuacion_ndsi(e)) for e in elementos]
    return metricas_split(copia, umbral, soporte, clave="puntuacion_ndsi")


def resumen_seleccion(elementos, umbral, soporte="observable") -> dict:
    """IoU de cobertura y MAE de área para comparar checkpoints o estrategias de resolución en validación."""
    m = metricas_split(elementos, umbral, soporte)["agregado"]
    return {"iou": m["iou"], "f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
            "mae_ha": m["mae_ha"], "sesgo_ha": m["sesgo_ha"], "auc": m["auc"], "umbral": umbral}


def mejor_candidato(candidatos) -> int:
    """Índice del mejor candidato: mayor IoU de cobertura en validación; desempate por menor MAE de área."""
    def clave(k):
        c = candidatos[k]
        return (-(c["iou"] if c["iou"] is not None else -1.0),
                c["mae_ha"] if c["mae_ha"] is not None else float("inf"), k)
    return min(range(len(candidatos)), key=clave)
