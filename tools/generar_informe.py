"""Genera reports/<run_id>/informe.md (y, si hay pandoc y Edge/Chrome, informe.html e informe.pdf).

Todas las cifras, tablas y figuras salen de los archivos de la corrida real (metricas.json y CSV que escribe el
notebook). Los apartados marcados «[Completar]» son del autor del informe. No se fabrica ningún dato: si falta
un archivo, el apartado indica «pendiente» y el motivo.

    python tools/generar_informe.py [run_id]          # por defecto, la corrida del paquete activo
"""
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def num(v, d=2, suf="", signo=False):
    if v is None or v == "":
        return "N/D"
    v = float(v)
    t = f"{v:+,.{d}f}" if signo else f"{v:,.{d}f}"
    return t.replace(",", "\u202f").replace(".", ",") + suf


def leer_csv(ruta):
    if not ruta.is_file():
        return []
    with open(ruta, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def tabla(encabezados, filas):
    lineas = ["| " + " | ".join(encabezados) + " |", "|" + "|".join("---" if k == 0 else "---:" for k in
                                                                    range(len(encabezados))) + "|"]
    lineas += ["| " + " | ".join(str(c) for c in f) + " |" for f in filas]
    return "\n".join(lineas)


def figura(rep, nombre, pie):
    if (rep / nombre).is_file():
        return f"![{pie}]({nombre})\n\n*{pie}*\n"
    return f"> Figura pendiente: `{nombre}` no existe en esta corrida.\n"


def fila_metricas(rotulo, m):
    return [rotulo, num(m.get("precision"), 3), num(m.get("recall"), 3), num(m.get("f1"), 3), num(m.get("iou"), 3),
            num(m.get("auc"), 3), num(m.get("exactitud"), 4), m.get("tp"), m.get("fp"), m.get("fn"), m.get("tn"),
            num(m.get("mae_ha")), num(m.get("sesgo_ha"), 2, signo=True)]


def construir(run_id) -> Path:
    rep = RAIZ / "reports" / run_id
    met_ruta = rep / "metricas.json"
    if not met_ruta.is_file():
        raise SystemExit(f"No existe {met_ruta}: ejecute primero el notebook (la plantilla solo se llena con una "
                         "corrida real).")
    met = json.loads(met_ruta.read_text(encoding="utf-8"))
    ds, sel, ref = met["dataset"], met["seleccion"], met["referencia"]
    ent = met["entrenamiento"]
    principal = sel["soporte"]
    splits = leer_csv(rep / "tabla_splits.csv")
    refs = leer_csv(rep / "tabla_referencias.csv")
    demo = leer_csv(rep / "demo_comparacion.csv")
    demo_esc = leer_csv(rep / "demo_escenas.csv")
    ckpt = leer_csv(rep / "seleccion_checkpoints.csv")
    backends = leer_csv(rep / "verificacion_backends.csv")
    parches = met.get("recall_parches_test") or []
    ndsi = met.get("comparador_ndsi_test") or {}
    fechas = sorted(r["fecha_utc"][:10] for r in refs)
    capturas = sorted((rep / "capturas").glob("*.png")) if (rep / "capturas").is_dir() else []
    t = []
    t.append(f"""---
title: "Cobertura de nieve y hielo visible en el Nevado Santa Isabel con YOLO Segmentation"
subtitle: "Informe del proyecto final · Visión por computador"
author: "[Completar: nombre del estudiante]"
date: "[Completar: fecha de entrega]"
lang: es
---

> Borrador generado desde la corrida **`{run_id}`** (paquete `{met['paquete']}`). Todas las cifras remiten a
> `reports/{run_id}/`. Los apartados «[Completar]» son del autor.

# 1. Contexto y área de aplicación

**Pregunta de monitoreo.** ¿Cuánta cobertura de nieve y hielo visible hay en una escena Sentinel-2 del Nevado
Santa Isabel (Cordillera Central de Colombia) y dónde cambió entre dos fechas?

El área de interés es el rectángulo WGS84 [-75,397; 4,773; -75,348; 4,827], analizado en la malla EPSG:32618 de
10 m del generador mensual (545 × 598 píxeles; 1 píxel = 0,01 ha). Se analiza todo el AOI, incluidos los parches
separados de la cumbre principal.

**Nieve/hielo visible no es pérdida glaciar.** La clase reúne nieve e hielo visibles. Una reducción entre dos
adquisiciones puede incluir nieve estacional, sombras y errores del modelo; por eso se habla de *pérdida
observada de cobertura*. Dos fechas por sí solas no permiten medir balance de masa ni atribuir una tendencia
glaciar.

[Completar: motivación y relevancia del caso.]

# 2. Descripción del dataset

- **Origen.** Sentinel-2 (Copernicus) descargado con Google Earth Engine por el generador mensual (SDD 03):
  una adquisición como máximo por mes entre 2016 y 2026, elegida por visibilidad de las cumbres.
- **Versión usada.** `{ds['version']}`, huella de contenido `{ds['huella_contenido']}`, ejecución
  `{ds['id_ejecucion']}`, SHA-256 del manifiesto `{ds['sha256_manifiesto'][:16]}…`.
- **Fechas efectivamente disponibles.** {len(fechas)} escenas entre {fechas[0] if fechas else '—'} y
  {fechas[-1] if fechas else '—'} ({sum(int(s['meses_ausentes']) for s in splits)} de 132 meses planificados
  quedaron sin escena exportada, sobre todo por nubosidad).
- **Perfiles.** L1C (reflectancia TOA, con Cloud Score+) antes de la disponibilidad de L2A y L2A (reflectancia de
  superficie, con SCL) después.
- **Variantes.** `rgb` (B4/B3/B2 en 8 bits, reflectancia 0–0,60 → 0–255) y `espectral` (NDSI/NDVI/B4), con las
  mismas claves, particiones y etiquetas. Este informe usa **{met['paquete'].split('-')[1]}**.
- **Procedencia de las etiquetas.** {', '.join(ref['origen'])}: pseudoetiquetas automáticas por reglas NDSI
  (umbral 0,4 y reglas por componente), **sin revisión humana**. Los componentes ambiguos (p. ej. en contacto con
  nubes) quedan como fondo.
- **Particiones.** Cronológicas, definidas por el generador sobre el calendario (92/19/21 meses planificados).
""")
    t.append(tabla(["Split", "Meses planificados", "Escenas", "Positivas", "Negativas", "Instancias", "L1C", "L2A",
                    "Años"],
                   [[s["split"], s["meses_planificados"], s["escenas"], s["positivas"], s["negativas"],
                     s["instancias"], s["L1C"], s["L2A"], s["años"]] for s in splits]))
    amb = [float(r["ambiguos_ha"]) for r in refs if r.get("ambiguos_ha")]
    etiq = [float(r["referencia_ha"]) for r in refs]
    t.append(f"""
*Tabla 1. Escenas útiles reales por partición (`tabla_splits.csv`).*

Limitaciones del conjunto: validación con muy poco soporte, ninguna escena totalmente negativa y referencia
automática. En total hay {num(sum(etiq), 1)} ha etiquetadas y {num(sum(amb), 1)} ha en componentes NDSI ambiguos
que las etiquetas dejan como fondo (`tabla_referencias.csv`).

{figura(rep, 'fig_ejemplos_etiquetas.png', 'Figura 1. Imagen y etiqueta: referencia en magenta, componentes ambiguos omitidos en rojo, no observable en gris.')}
{figura(rep, 'fig_histogramas.png', 'Figura 2. Área y tamaño de los parches etiquetados por partición.')}

# 3. Metodología

**Flujo.** Adquisición Sentinel-2 (GEE) → malla común de 10 m → pseudoetiquetado NDSI → particiones cronológicas
→ ajuste de YOLO26n-seg → unión de máscaras de instancia → hectáreas (área planimétrica: píxeles × 0,01 ha).

**Modelo y entrenamiento.** Se afina `{ent['ajustada']['modelo_base']}` preentrenado con una clase
(`nieve_hielo_visible`), semilla 42, imgsz {ent['ajustada']['argumentos'].get('imgsz')}, lote
{ent['ajustada']['argumentos'].get('batch')}, sin mosaic, mixup ni jitter HSV (solo aumentos geométricos).
Dispositivo: **{ent['ajustada']['gpu']}** (`{ent['ajustada']['dispositivo']}`).
""")
    t.append(tabla(["Corrida", "Épocas pedidas", "Épocas ejecutadas", "nbs", "Paciencia", "Tiempo (s)",
                    "Pico asignado (MiB)", "Pico reservado (MiB)"],
                   [[r["nombre"], r["epocas_pedidas"], r["epocas_ejecutadas"], r["argumentos"].get("nbs"),
                     r["argumentos"].get("patience"), num(r["segundos"], 1), r["pico_asignado_mib"],
                     r["pico_reservado_mib"]] for r in (ent["integracion"], ent["inicial"], ent["ajustada"])]))
    ious = [float(c["iou"]) for c in ckpt if c.get("iou") not in ("", None)]
    t.append(f"""
*Tabla 2. Corridas en la GPU (`tabla_entrenamiento.csv`).*

La configuración inicial propuesta (50 épocas, paciencia 10) se detuvo en la época
{ent['inicial']['epocas_ejecutadas']}: con lote 2, Ultralytics acumula gradientes hasta un lote nominal de 64
(un paso del optimizador cada ~3,5 épocas) y la parada temprana se guía por el mAP de instancia de una validación
de 7 instancias. La corrida ajustada usa un paso por lote y sin parada temprana.

{figura(rep, 'fig_curvas_entrenamiento.png', 'Figura 3. Curvas de entrenamiento de las corridas inicial y ajustada.')}

**Selección (solo validación).** Checkpoint y resolución por mayor IoU de cobertura (desempate: menor MAE de
área); umbral operativo por máximo F1. Elegido: `{sel['checkpoint']}`, imgsz {sel['imgsz']}, umbral operativo
{num(sel['umbral'], 4)}. Entre los {len(ckpt)} checkpoints comparados el IoU de validación varió de
{num(min(ious), 3) if ious else 'N/D'} a {num(max(ious), 3) if ious else 'N/D'}: con 4 escenas la selección es
inestable.

**Puntuación por píxel y ROC.** Una inferencia por imagen con umbral de propuestas 0,01; la puntuación de cada
píxel es el máximo de las confianzas de las instancias que lo cubren (0 fuera). Es un ranking con empates, no una
probabilidad calibrada. La ROC recorre el umbral sobre esa puntuación; el punto operativo es el umbral elegido.

**Protocolo.** La configuración se congeló antes de tocar la prueba, que se evaluó
{met['protocolo']['evaluaciones_test']} vez.

# 4. Resultados

Clasificación binaria por píxel (positivo = nieve/hielo visible) contra pseudoetiquetas NDSI. Soporte principal:
**{ref['soporte_principal']}**.
""")
    filas = []
    for split, rotulo in (("val", "Validación"), ("test", "Prueba temporal")):
        for sop, txt in (("observable", "píxel observable"), ("observable_sin_ambiguos", "observable sin ambiguos"),
                         ("imagen", "toda la imagen, sin control de observabilidad")):
            if sop in met[split]:
                filas.append(fila_metricas(f"{rotulo} · {txt}", met[split][sop]))
    t.append(tabla(["Conjunto y soporte", "Precisión", "Recall", "F1", "IoU", "AUC", "Exactitud", "TP", "FP", "FN",
                    "TN", "MAE área (ha)", "Sesgo (ha)"], filas))
    inst = met["test"].get("instancias") or {}
    t.append(f"""
*Tabla 3. Métricas acumuladas sobre todos los píxeles válidos (`metricas.json`). Precisión = TP/(TP+FP), distinta
de la exactitud global.*

Máscaras de instancia en prueba (validador de Ultralytics): mAP50 {num(inst.get('mascara_map50'), 3)}, mAP50–95
{num(inst.get('mascara_map50_95'), 3)}.

{figura(rep, 'fig_roc.png', 'Figura 4. Curva ROC por píxel con el punto operativo marcado.')}
{figura(rep, 'fig_matriz_test.png', 'Figura 5. Matriz de confusión de la prueba temporal.')}
{figura(rep, 'fig_pr.png', 'Figura 6. Curva precisión–recall (clases muy desbalanceadas).')}
""")
    t.append(tabla(["Fecha", "Perfil", "Precisión", "Recall", "F1", "IoU", "AUC", "Ref. (ha)", "Pred. (ha)",
                    "Error (ha)"],
                   [[e["mes"], e["perfil"], num(e.get("precision"), 3), num(e.get("recall"), 3), num(e.get("f1"), 3),
                     num(e.get("iou"), 3), num(e.get("auc"), 3), num(e.get("area_ref_ha")), num(e.get("area_pred_ha")),
                     num(e.get("error_ha"), 2, signo=True)] for e in met["test_por_escena"]]))
    t.append(f"""
*Tabla 4. Prueba temporal por escena, soporte {principal} (`metricas_test_por_escena_{principal}.csv`). Referencia:
{', '.join(ref['origen'])}.*

{figura(rep, 'fig_test_predicciones.png', 'Figura 7. Prueba temporal: original, predicción (cian) frente a referencia (magenta) y mapa de aciertos y errores.')}
""")
    if parches:
        t.append(tabla(["Tamaño del parche", "Parches", "Recuperados", "Recall de parches"],
                       [[p["tramo"], p["parches"], p["recuperados"], num(p["recall_de_parches"], 2)]
                        for p in parches]))
        t.append("\n*Tabla 5. Parches de la referencia recuperados (≥ 50 % de sus píxeles) según su tamaño.*\n")
    if ndsi:
        t.append(tabla(["Método y soporte", "Precisión", "Recall", "F1", "IoU", "AUC", "MAE área (ha)"],
                       [[f"YOLO · {s}", *[num(met["test"][s].get(k), 3) for k in ("precision", "recall", "f1", "iou",
                                                                                 "auc")],
                         num(met["test"][s].get("mae_ha"))] for s in ndsi] +
                       [[f"NDSI ≥ 0,4 · {s}", *[num(m.get(k), 3) for k in ("precision", "recall", "f1", "iou", "auc")],
                         num(m.get("mae_ha"))] for s, m in ndsi.items()]))
        t.append("\n*Tabla 6. Comparador NDSI en las mismas escenas. Es una concordancia **circular**: las "
                 "pseudoetiquetas se generan con ese índice; no demuestra superioridad de ningún método.*\n")
    if demo:
        d = demo[0]
        t.append(f"""
## Comparación de dos fechas

Escenas reales {d['fecha_inicial_utc'][:10]} y {d['fecha_final_utc'][:10]} (Sentinel-2 L2A), analizadas con el
mismo paquete. Soporte común observable V: {num(d['area_comun_observable_ha'], 1)} ha
({num(d['cobertura_comun_pct_del_aoi'], 1)} % del AOI). Solo se informa sobre V; no se extrapola a lo oculto.
""")
        t.append(tabla(["Magnitud", "Valor"], [
            ["Nieve/hielo visible inicial comparable A₀", num(d["area_inicial_comparable_ha"], 2, " ha")],
            ["Nieve/hielo visible final comparable A₁", num(d["area_final_comparable_ha"], 2, " ha")],
            ["Persistencia", num(d["persistencia_ha"], 2, " ha")],
            ["Pérdida bruta localizada L", num(d["perdida_bruta_ha"], 2, " ha")],
            ["Pérdida respecto al inicial (100·L/A₀)", num(d["perdida_pct_del_inicial"], 1, " %")],
            ["Ganancia G", num(d["ganancia_ha"], 2, " ha")],
            ["Cambio neto Δ = A₁ − A₀", num(d["cambio_neto_ha"], 2, " ha", signo=True)],
            ["Cambio neto respecto al inicial (100·Δ/A₀)", num(d["cambio_neto_pct_del_inicial"], 1, " %", signo=True)],
            ["Nieve inicial fuera del soporte común", num(d.get("nieve_inicial_fuera_del_area_comun_ha"), 2, " ha")]]))
        t.append(f"""
*Tabla 7. Demostración de dos fechas (`demo_comparacion.csv`). La pérdida bruta responde cuánto desapareció donde
antes había nieve; el cambio neto incorpora la nieve aparecida en otros lugares.*

{figura(rep, 'fig_demo_dos_fechas.png', 'Figura 8. Predicción en cada fecha y mapa de cambio: rojo pérdida, verde ganancia, azul persistencia, gris sin observación común.')}
""")
        if d.get("avisos"):
            t.append("Avisos de la comparación: " + d["avisos"] + "\n")
    if capturas:
        t.append("## Capturas de la aplicación\n")
        for c in capturas:
            t.append(f"![{c.stem}](capturas/{c.name})\n\n*Captura de la aplicación Streamlit: {c.stem}.*\n")
    else:
        t.append("\n> Capturas de la aplicación: pendientes (guárdelas en `capturas/` y vuelva a generar).\n")
    if backends:
        t.append("## Equivalencia entre backends (misma escena, mismo paquete)\n")
        t.append(tabla(["Backend", "Escena", "Área (ha)", "Píxeles", "Parches", "Inferencia (s)", "Memoria pico (MiB)"],
                       [[b["backend"], b["clave"][:8], num(b["area_nieve_ha"]), b["pixeles_nieve"], b["parches"],
                         num(b["segundos_inferencia"], 2), b.get("memoria_pico_mib") or "—"] for b in backends]))
        t.append("\n*Tabla 8. El paquete reabierto en un proceso limpio en CPU reproduce el resultado de la GPU "
                 "(`verificacion_backends.csv`).*\n")
    m = met["test"].get(principal, {})
    mv = met["val"].get(principal, {})
    t.append(f"""
# 5. Conclusiones

Hechos medidos en esta corrida (interpretación a cargo del autor):

- En la prueba temporal, sobre {principal}, el modelo obtuvo precisión {num(m.get('precision'), 3)}, recall
  {num(m.get('recall'), 3)}, F1 {num(m.get('f1'), 3)}, IoU {num(m.get('iou'), 3)} y AUC {num(m.get('auc'), 3)},
  con MAE de área de {num(m.get('mae_ha'))} ha por escena y sesgo {num(m.get('sesgo_ha'), 2, signo=True)} ha. En
  validación el F1 fue {num(mv.get('f1'), 3)}: la diferencia entre ambos indica cuánto dependió la selección de
  un conjunto de validación de 4 escenas.
- La referencia es automática. Estas cifras miden concordancia con pseudoetiquetas NDSI, no exactitud
  glaciológica; al excluir los componentes ambiguos el resultado cambia (Tabla 3), lo que muestra el peso de esa
  incertidumbre.
- Nubes y estacionalidad: la medición solo cuenta píxeles observables y la comparación se limita al soporte
  común; una diferencia entre dos fechas puede deberse a nieve estacional.
- Tamaño muestral: {ds['escenas']['train']} escenas de entrenamiento, {ds['escenas']['val']} de validación y
  {ds['escenas']['test']} de prueba, todas del mismo nevado y sin escenas negativas.

[Completar: aciertos y fallos observados en las figuras 7 y 8, y conclusión que permiten las dos fechas.]

# Fuentes y reproducibilidad

- Datos: contiene datos modificados de Copernicus Sentinel-2 procesados mediante Google Earth Engine.
- Código, configuración y paquete: repositorio `santa_isabel_yolo`; corrida `{run_id}`; paquete
  `models/{met['paquete']}`.
- Software: Ultralytics YOLO (AGPL-3.0), PyTorch, ONNX Runtime, Rasterio, Streamlit.
""")
    destino = rep / "informe.md"
    destino.write_text("\n".join(t), encoding="utf-8")
    return destino


def renderizar(md: Path):
    """HTML autocontenido con pandoc y PDF con Edge o Chrome sin interfaz."""
    html, pdf = md.with_suffix(".html"), md.with_suffix(".pdf")
    pandoc = shutil.which("pandoc")
    if not pandoc:
        print("pandoc no está instalado: abra informe.md en un editor Markdown y exporte a PDF.")
        return
    css = md.parent / "_informe.css"
    css.write_text("body{font-family:Segoe UI,Arial,sans-serif;max-width:900px;margin:auto;font-size:11pt;"
                   "line-height:1.45}table{border-collapse:collapse;font-size:9pt;margin:8px 0}"
                   "td,th{border:1px solid #999;padding:3px 6px}img{max-width:100%}"
                   "blockquote{color:#555;border-left:3px solid #bbb;padding-left:10px;margin-left:0}"
                   "h1{font-size:15pt;border-bottom:1px solid #ccc}h2{font-size:12.5pt}", encoding="utf-8")
    subprocess.run([pandoc, md.name, "-o", html.name, "--standalone", "--embed-resources", "--css", css.name,
                    "--from", "markdown+pipe_tables"], cwd=md.parent, check=True)
    css.unlink()
    print("Escrito", html)
    navegadores = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                   r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                   r"C:\Program Files\Google\Chrome\Application\chrome.exe", shutil.which("chromium") or "",
                   shutil.which("google-chrome") or ""]
    nav = next((n for n in navegadores if n and Path(n).is_file()), None)
    if not nav:
        print("No se encontró Edge ni Chrome: abra informe.html e imprima a PDF.")
        return
    subprocess.run([nav, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    html.as_uri()], check=True, capture_output=True, timeout=180)
    print("Escrito", pdf if pdf.is_file() else "(el navegador no generó el PDF: imprima informe.html a PDF)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run = sys.argv[1]
    else:
        activo = (RAIZ / "models" / "ACTIVO").read_text(encoding="utf-8").strip()
        run = json.loads((RAIZ / "models" / activo / "metricas.json").read_text(encoding="utf-8"))["run_id"]
    md = construir(run)
    print("Escrito", md)
    renderizar(md)
