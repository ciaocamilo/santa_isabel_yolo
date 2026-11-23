"""Aplicación Streamlit · Cobertura de nieve y hielo visible en el Nevado Santa Isabel (SDD 04 §7).

Consume el paquete publicado por el notebook y usa las mismas funciones de lectura, inferencia y medición.
Abre y analiza archivos locales sin autenticarse en Earth Engine. Sin un paquete ajustado no presenta
resultados de nieve.

    streamlit run app.py
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

import santa_isabel  # noqa: E402,F401  (primero: fija PROJ/GDAL antes de importar rasterio)

import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402

import streamlit as st  # noqa: E402

from santa_isabel import areas, visual  # noqa: E402
from santa_isabel.config import cargar_config, resolver  # noqa: E402
from santa_isabel.escenas import ErrorEscena, indice_ejemplos, iso_utc, leer_escena  # noqa: E402
from santa_isabel.flujo import analizar_escena, comparar_fechas  # noqa: E402
from santa_isabel.paquete import ErrorPaquete, cargar_paquete, resolver_activo  # noqa: E402
from santa_isabel.visual import numero  # noqa: E402

EXTENSIONES = ["png", "jpg", "jpeg", "tif", "tiff", "zip", "json"]
BASE_TEMPORAL = Path(tempfile.gettempdir()) / "santa_isabel_app"

st.set_page_config(page_title="Nieve y hielo visible · Nevado Santa Isabel", page_icon="🏔️", layout="wide",
                   initial_sidebar_state="collapsed")


# --- Recursos compartidos por proceso ---------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def configuracion():
    return cargar_config(raiz=RAIZ)


@st.cache_resource(show_spinner="Cargando el modelo publicado…")
def modelo_activo():
    """Carga el paquete una sola vez por proceso. Devuelve (modelo, None) o (None, ErrorPaquete)."""
    cfg = configuracion()
    try:
        ruta = resolver_activo(resolver(cfg, "models"), cfg.get("app", {}).get("paquete") or None)
        return cargar_paquete(ruta, backend=os.environ.get("SANTA_ISABEL_BACKEND") or None), None
    except ErrorPaquete as e:
        return None, e


@st.cache_resource(show_spinner=False)
def ejemplos():
    """Escenas de ejemplo incluidas en el repositorio y el par de la demostración."""
    cfg = configuracion()
    carpeta = resolver(cfg, cfg.get("app", {}).get("ejemplos", "examples"))
    demo = {}
    if (carpeta / "demo.json").is_file():
        demo = json.loads((carpeta / "demo.json").read_text(encoding="utf-8"))
    escenas = []
    for meta in sorted(carpeta.glob("*/metadata.json")):
        datos = json.loads(meta.read_text(encoding="utf-8"))
        escenas.append({"clave": datos["clave"], "fecha": datos["adquisicion_utc"], "carpeta": meta.parent,
                        "fuente": datos.get("id_asset", ""), "atribucion": datos.get("atribucion", "")})
    escenas.sort(key=lambda e: e["fecha"])
    return {"carpeta": carpeta, "escenas": escenas, "demo": demo, "indice": indice_ejemplos(carpeta)}


def limpiar_temporales_antiguos(horas=6):
    if not BASE_TEMPORAL.is_dir():
        return
    limite = time.time() - horas * 3600
    for d in BASE_TEMPORAL.iterdir():
        try:
            if d.is_dir() and d.stat().st_mtime < limite:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


# --- Estado por sesión ------------------------------------------------------------------------------------------------
def carpeta_sesion() -> Path:
    """Directorio temporal propio de la sesión: archivos y resultados no se comparten entre usuarios."""
    if "tmp" not in st.session_state:
        BASE_TEMPORAL.mkdir(parents=True, exist_ok=True)
        limpiar_temporales_antiguos()
        st.session_state["tmp"] = tempfile.mkdtemp(prefix="sesion_", dir=BASE_TEMPORAL)
    return Path(st.session_state["tmp"])


def guardar_cargas(archivos, ranura) -> list:
    """Escribe las cargas en una subcarpeta nueva de la sesión (borra la anterior de esa ranura)."""
    lim = configuracion()["limites"]
    for f in archivos:
        if f.size > lim["max_bytes"]:
            raise ErrorEscena(f"{f.name} pesa {f.size / 1e6:.1f} MB; el máximo admitido es "
                              f"{lim['max_bytes'] / 1e6:.0f} MB. Recorte la escena al AOI del Santa Isabel.")
    base = carpeta_sesion() / ranura
    shutil.rmtree(base, ignore_errors=True)
    destino = base / uuid.uuid4().hex[:8]
    destino.mkdir(parents=True)
    rutas = []
    for f in archivos:
        ruta = destino / Path(f.name).name
        ruta.write_bytes(f.getbuffer())
        rutas.append(ruta)
    return rutas


def opciones_de_carga(ranura):
    """Datos que el archivo puede no traer: nivel de procesamiento, fecha y mapeo de bandas."""
    with st.expander("Opciones de lectura (nivel, fecha, bandas)", expanded=st.session_state.get(f"pide_{ranura}",
                                                                                              False)):
        c1, c2 = st.columns(2)
        perfil = c1.selectbox("Nivel de procesamiento", ["Automático", "L2A", "L1C"], key=f"perfil_{ranura}",
                              help="Necesario para GeoTIFF en valores digitales (reflectancia = DN × 0,0001).")
        con_fecha = c2.checkbox("Indicar la fecha de adquisición", key=f"confecha_{ranura}",
                                help="Solo si el archivo no la trae en sus metadatos o en su nombre.")
        fecha = c2.date_input("Fecha (UTC)", key=f"fecha_{ranura}", format="YYYY-MM-DD") if con_fecha else None
        texto = st.text_input("Mapeo de bandas del GeoTIFF (solo si no trae nombres)", key=f"mapeo_{ranura}",
                              placeholder="B2=1, B3=2, B4=3, B8=4, B11=5")
    mapeo = None
    if texto.strip():
        try:
            mapeo = {k.strip().upper(): int(v) for k, v in (par.split("=") for par in texto.split(","))}
        except ValueError:
            st.error("Mapeo de bandas no reconocido. Use el formato B2=1, B3=2, B4=3, B8=4, B11=5.")
            mapeo = None
    return (None if perfil == "Automático" else perfil), fecha, mapeo


def leer_carga(archivos, ranura, perfil, fecha, mapeo):
    """Lee una escena cargada; muestra el error en español y qué dato falta."""
    try:
        rutas = guardar_cargas(archivos, ranura)
        escena = leer_escena(rutas, perfil, fecha=fecha, mapeo_bandas=mapeo, indice=ejemplos()["indice"],
                             limites=configuracion()["limites"], dir_temporal=carpeta_sesion() / ranura)
        st.session_state[f"pide_{ranura}"] = False
        return escena
    except ErrorEscena as e:
        st.session_state[f"pide_{ranura}"] = e.requiere is not None
        st.error(f"No se pudo leer la escena: {e}")
        if e.requiere:
            st.info({"mapeo_bandas": "Abra «Opciones de lectura» e indique el mapeo de bandas.",
                     "perfil": "Abra «Opciones de lectura» y seleccione L2A o L1C.",
                     "fecha": "Abra «Opciones de lectura» e indique la fecha de adquisición."}[e.requiere])
        return None


# --- Presentación -----------------------------------------------------------------------------------------------------
def ficha_escena(escena):
    datos = [f"**Fecha (UTC):** {iso_utc(escena.fecha) or 'no indicada'}",
             f"**Fuente:** {escena.fuente or escena.clave}",
             f"**Tamaño:** {escena.ancho} × {escena.alto} px"
             + (f" · {escena.georef.crs} · celda {escena.georef.ha_por_celda:g} ha" if escena.georef
                else " · sin georreferencia"),
             f"**Calidad (QA):** {escena.procedencia_observable}"]
    st.markdown("  \n".join(datos))
    for aviso in escena.avisos:
        st.warning(aviso, icon="⚠️")


def imagen_base(escena):
    return visual.realzar(escena.rgb8) if st.session_state.get("realce", True) else escena.rgb8


def nota_realce() -> str:
    return " (brillo realzado para ver)" if st.session_state.get("realce", True) else ""


def panel_como_interpretar():
    with st.expander("Cómo interpretar"):
        st.markdown(
            "- La clase reúne **nieve e hielo visibles**. Una reducción entre dos fechas puede incluir nieve "
            "estacional, sombras y errores del modelo: por eso se habla de **pérdida observada de cobertura**.\n"
            "- Dos fechas por sí solas **no miden balance de masa** ni permiten atribuir una tendencia glaciar.\n"
            "- Las hectáreas son **superficie planimétrica estimada** sobre píxeles observables (sin nube ni "
            "sombra según la máscara de calidad). Lo no observable no se cuenta ni se extrapola.\n"
            "- **Pérdida bruta (%)** responde cuánto desapareció donde antes había nieve. **Cambio neto (%)** "
            "incorpora además la nieve que apareció en otros lugares.\n"
            "- La referencia con la que se entrenó y evaluó el modelo es **automática** (reglas NDSI del "
            "generador mensual, sin revisión humana): las métricas son concordancia con esas pseudoetiquetas.")


def metrica(columna, titulo, valor, ayuda=None):
    columna.metric(titulo, valor, help=ayuda)


def mostrar_analisis(analisis, modelo, clave_descarga):
    """Original, superposición, cifras y descargas de una escena."""
    esc, pred, med = analisis.escena, analisis.prediccion, analisis.medicion
    opacidad = st.session_state.get("opacidad", 0.45)
    c1, c2, c3 = st.columns([1, 1, 1.1])
    c1.image(imagen_base(esc), caption=f"Original{nota_realce()} · {iso_utc(esc.fecha) or 'sin fecha'}",
             width="stretch")
    c2.image(visual.superponer(imagen_base(esc), pred.mascara, esc.observable, opacidad),
             caption="Predicción superpuesta", width="stretch")
    with c3:
        st.markdown(f"**{'Área estimada' if med.estado == 'cuantitativa' else 'Área'} de nieve/hielo visible**")
        a, b = st.columns(2)
        metrica(a, "Superficie", numero(med.area_nieve_ha, 2, " ha"), med.etiqueta_area)
        metrica(b, "% del AOI", numero(med.pct_nieve_aoi, 2, " %"),
                "Nieve/hielo visible respecto al área total del AOI.")
        a, b = st.columns(2)
        metrica(a, "Cobertura observable", numero(med.area_observable_ha, 1, " ha"),
                "Superficie del AOI con observación válida (denominador de la medición).")
        metrica(b, "Observable / AOI", numero(med.pct_observable_aoi, 1, " %"))
        st.caption(f"{med.etiqueta_area.capitalize()} · modelo `{modelo.id}` · umbral operativo "
                   f"{numero(pred.umbral_operativo, 3)} · {pred.instancias_operativas} parche(s) · "
                   f"{pred.backend} en {pred.dispositivo} · {numero(pred.segundos, 2)} s")
        for motivo in med.motivos:
            st.info(motivo)
        if pred.tope_alcanzado:
            st.warning("Se alcanzó el tope de detecciones: algún parche pudo quedar recortado.")
    st.markdown(visual.leyenda_html(visual.LEYENDA_ESCENA), unsafe_allow_html=True)
    d1, d2, _ = st.columns([1, 1, 2])
    fila = {"modelo": modelo.id, **med.como_fila(), "parches": pred.instancias_operativas,
            "atribucion": esc.atribucion}
    d1.download_button("Descargar resumen (CSV)", areas.filas_a_csv([fila]), f"resumen_{esc.clave}.csv",
                       "text/csv", key=f"csv_{clave_descarga}")
    if med.malla is not None and med.mascara is not None:
        d2.download_button("Descargar máscara (GeoTIFF)", areas.exportar_mascara_geotiff(med, esc),
                           f"mascara_{esc.clave}.tif", "image/tiff", key=f"tif_{clave_descarga}")
    else:
        d2.caption("Sin georreferencia: no hay máscara georreferenciada que descargar.")


def mostrar_cambio(cambio, a, b, modelo):
    """Mapa de cambio, tabla A/B sobre el mismo soporte y descargas."""
    if cambio.estado != "ok":
        st.error("Comparación no disponible: " + " ".join(cambio.motivos))
        for aviso in cambio.avisos:
            st.warning(aviso, icon="⚠️")
        return
    ini, fin = (a, b) if not cambio.invertidas else (b, a)
    if cambio.invertidas:
        st.info("Las escenas se cargaron en orden inverso: se muestran ordenadas cronológicamente.")
    st.markdown(f"#### {cambio.fecha_inicial:%Y-%m-%d} → {cambio.fecha_final:%Y-%m-%d} · pérdida observada de "
                "cobertura")
    opacidad = st.session_state.get("opacidad", 0.45)
    izq, der = st.columns([0.9, 1.1])
    with izq:
        fondo = imagen_base(fin.escena) if fin.escena.georef.misma_malla(cambio.malla) else None
        st.image(visual.mapa_cambio_rgb(cambio.mapa, fondo), width="stretch",
                 caption="Mapa de cambio sobre el soporte común observable (fondo: escena final atenuada)")
        st.markdown(visual.leyenda_html(visual.leyenda_cambio()), unsafe_allow_html=True)
    with der:
        m1, m2 = st.columns(2)
        metrica(m1, "Pérdida bruta", numero(cambio.perdida_ha, 2, " ha"),
                "Superficie con nieve en la fecha inicial y sin nieve en la final.")
        metrica(m2, "Pérdida / inicial", numero(cambio.perdida_pct, 1, " %"),
                "Cuánto desapareció donde antes había nieve (100·L/A₀).")
        m1, m2 = st.columns(2)
        metrica(m1, "Ganancia", numero(cambio.ganancia_ha, 2, " ha"),
                "Superficie sin nieve en la fecha inicial y con nieve en la final.")
        metrica(m2, "Cambio neto", numero(cambio.cambio_neto_ha, 2, " ha", signo=True),
                "Δ = final − inicial = ganancia − pérdida.")
        m1, m2 = st.columns(2)
        metrica(m1, "Cambio neto / inicial", numero(cambio.cambio_neto_pct, 1, " %", signo=True),
                "Incorpora la nieve aparecida en otros lugares (100·Δ/A₀).")
        metrica(m2, "Área común observable", numero(cambio.area_comun_ha, 1, " ha"),
                f"{numero(cambio.cobertura_comun_pct, 1)} % del AOI. Solo se informa sobre este soporte.")
        st.dataframe(
            {"Magnitud (sobre el soporte común)": ["Nieve/hielo visible comparable (ha)",
                                                   "% del área común observable", "Persistencia (ha)",
                                                   "Nivel de procesamiento", "Área propia de la escena (ha)",
                                                   "% del AOI en la escena"],
             f"Inicial {cambio.fecha_inicial:%Y-%m-%d}": [
                 numero(cambio.area_inicial_ha), numero(cambio.pct_inicial_en_comun), numero(cambio.persistencia_ha),
                 ini.escena.perfil or "—", numero(ini.medicion.area_nieve_ha), numero(ini.medicion.pct_nieve_aoi)],
             f"Final {cambio.fecha_final:%Y-%m-%d}": [
                 numero(cambio.area_final_ha), numero(cambio.pct_final_en_comun), numero(cambio.persistencia_ha),
                 fin.escena.perfil or "—", numero(fin.medicion.area_nieve_ha), numero(fin.medicion.pct_nieve_aoi)]},
            hide_index=True, width="stretch")
        if cambio.inicial_fuera_comun_ha:
            st.caption(f"Nieve inicial observable fuera del soporte común (sin dato en la otra fecha): "
                       f"{numero(cambio.inicial_fuera_comun_ha)} ha "
                       f"({numero(cambio.inicial_fuera_comun_pct, 1)} %). No se extrapola.")
        for motivo in cambio.motivos:
            st.info(motivo)
        for aviso in cambio.avisos:
            st.warning(aviso, icon="⚠️")
    c1, c2, c3, c4 = st.columns(4)
    for col, an, rotulo in ((c1, ini, "Inicial"), (c3, fin, "Final")):
        col.image(imagen_base(an.escena), caption=f"{rotulo} · original · {an.escena.fecha:%Y-%m-%d}",
                  width="stretch")
    for col, an, rotulo in ((c2, ini, "Inicial"), (c4, fin, "Final")):
        col.image(visual.superponer(imagen_base(an.escena), an.prediccion.mascara, an.escena.observable, opacidad),
                  caption=f"{rotulo} · predicción · {numero(an.medicion.area_nieve_ha)} ha", width="stretch")
    st.markdown(visual.leyenda_html(visual.LEYENDA_ESCENA), unsafe_allow_html=True)
    d1, d2, _ = st.columns([1, 1, 2])
    fila = {"modelo": modelo.id, **cambio.como_fila(),
            "area_propia_inicial_ha": ini.medicion.area_nieve_ha, "area_propia_final_ha": fin.medicion.area_nieve_ha,
            "atribucion_inicial": ini.escena.atribucion, "atribucion_final": fin.escena.atribucion}
    nombre = f"{cambio.fecha_inicial:%Y%m%d}_{cambio.fecha_final:%Y%m%d}"
    d1.download_button("Descargar comparación (CSV)", areas.filas_a_csv([fila]), f"comparacion_{nombre}.csv",
                       "text/csv", key="csv_cambio")
    d2.download_button("Descargar mapa de cambio (GeoTIFF)", areas.exportar_cambio_geotiff(cambio),
                       f"cambio_{nombre}.tif", "image/tiff", key="tif_cambio")
    st.caption("GeoTIFF categórico: " + "; ".join(f"{k} = {v}" for k, v in areas.CATEGORIAS.items()) + ".")


def analizar_con_progreso(escenas, modelo):
    """Ejecuta la inferencia mostrando el avance. Devuelve la lista de análisis o None si falla."""
    cfg = configuracion()
    barra = st.progress(0.0, text="Preparando la escena…")
    resultados = []
    try:
        for k, escena in enumerate(escenas):
            barra.progress(k / len(escenas), text=f"Analizando {escena.clave} ({k + 1} de {len(escenas)})…")
            resultados.append(analizar_escena(escena, modelo, cfg))
        barra.progress(1.0, text="Análisis terminado.")
    except ErrorEscena as e:
        barra.empty()
        st.error(f"No se pudo analizar la escena: {e}")
        return None
    except Exception as e:  # error inesperado: se informa sin dejar la sesión rota
        barra.empty()
        st.error(f"El análisis falló de forma inesperada ({type(e).__name__}: {e}). Vuelva a intentarlo o reduzca "
                 "el tamaño de la imagen.")
        return None
    barra.empty()
    return resultados


# --- Pestañas ---------------------------------------------------------------------------------------------------------
def pestana_una(modelo):
    st.markdown("Cargue una escena Sentinel-2 del Santa Isabel y pulse **Analizar**. Formatos: paquete de escena "
                "del generador (ZIP o `metadata.json` + PNG + `calidad.tif` + `soporte_png.tif`), GeoTIFF RGB o "
                "multibanda, PNG o JPG RGB de 8 bits.")
    izq, der = st.columns([3, 2])
    archivos = izq.file_uploader("Archivos de la escena", type=EXTENSIONES, accept_multiple_files=True, key="carga_una")
    lista = ejemplos()["escenas"]
    with der:
        st.markdown("**o bien**")
        if lista:
            elegido = st.selectbox("Ejemplo incluido", [e["clave"] for e in lista], key="ejemplo_una",
                                   format_func=lambda c: next(f"{e['fecha'][:10]} · {c[:8]}…" for e in lista
                                                              if e["clave"] == c))
            if st.button("Cargar ejemplo del Santa Isabel", key="btn_ejemplo_una", width="stretch"):
                carpeta = next(e["carpeta"] for e in lista if e["clave"] == elegido)
                st.session_state["una"] = {"escena": leer_escena(carpeta), "analisis": None, "origen": "ejemplo"}
        else:
            st.caption("No hay ejemplos en `examples/`.")
    perfil, fecha, mapeo = opciones_de_carga("una")
    if archivos and st.button("Leer la escena cargada", key="btn_leer_una"):
        escena = leer_carga(archivos, "una", perfil, fecha, mapeo)
        if escena is not None:
            st.session_state["una"] = {"escena": escena, "analisis": None, "origen": "carga"}
    estado = st.session_state.get("una")
    if not estado:
        st.info("Aún no hay ninguna escena. Cargue un archivo o use el ejemplo.")
        return
    escena = estado["escena"]
    st.divider()
    c1, c2 = st.columns([1, 2])
    c1.image(imagen_base(escena), caption="Vista previa", width="stretch")
    with c2:
        ficha_escena(escena)
        if st.button("Analizar", type="primary", key="btn_analizar_una"):
            res = analizar_con_progreso([escena], modelo)
            estado["analisis"] = res[0] if res else None
    if estado.get("analisis") is not None:
        st.divider()
        mostrar_analisis(estado["analisis"], modelo, "una")
        if escena.atribucion:
            st.caption(escena.atribucion)


def pestana_dos(modelo):
    st.markdown("Cargue dos adquisiciones del mismo lugar y pulse **Comparar**. Para cuantificar la pérdida se "
                "exige georreferencia y máscara de calidad en ambas; se analizan con el mismo modelo y receta.")
    demo = ejemplos()
    par = demo["demo"].get("par") or [e["clave"] for e in demo["escenas"]][:2]
    carpetas = {e["clave"]: e["carpeta"] for e in demo["escenas"]}
    if len(par) == 2 and all(p in carpetas for p in par):
        if st.button("Cargar ejemplo del Santa Isabel (dos fechas)", key="btn_ejemplo_dos"):
            st.session_state["dos"] = {"a": leer_escena(carpetas[par[0]]), "b": leer_escena(carpetas[par[1]]),
                                       "resultado": None}
    c1, c2 = st.columns(2)
    with c1:
        arch_a = st.file_uploader("Escena A", type=EXTENSIONES, accept_multiple_files=True, key="carga_a")
        opc_a = opciones_de_carga("a")
    with c2:
        arch_b = st.file_uploader("Escena B", type=EXTENSIONES, accept_multiple_files=True, key="carga_b")
        opc_b = opciones_de_carga("b")
    if arch_a and arch_b and st.button("Leer las dos escenas cargadas", key="btn_leer_dos"):
        a, b = leer_carga(arch_a, "a", *opc_a), leer_carga(arch_b, "b", *opc_b)
        if a is not None and b is not None:
            st.session_state["dos"] = {"a": a, "b": b, "resultado": None}
    estado = st.session_state.get("dos")
    if not estado:
        st.info("Aún no hay escenas. Cargue dos archivos o use el ejemplo de dos fechas.")
        return
    st.divider()
    c1, c2 = st.columns(2)
    for col, k, rotulo in ((c1, "a", "Escena A"), (c2, "b", "Escena B")):
        with col:
            st.markdown(f"**{rotulo}**")
            p1, p2 = st.columns([1, 2])
            p1.image(imagen_base(estado[k]), width="stretch")
            with p2:
                ficha_escena(estado[k])
    fa, fb = estado["a"].fecha, estado["b"].fecha
    if fa is not None and fb is not None and fa.date() == fb.date():
        st.error(f"Las dos escenas son de la misma fecha ({fa:%Y-%m-%d}): cargue dos adquisiciones distintas.")
        return
    if st.button("Comparar", type="primary", key="btn_comparar"):
        res = analizar_con_progreso([estado["a"], estado["b"]], modelo)
        estado["resultado"] = None if res is None else {
            "a": res[0], "b": res[1], "cambio": comparar_fechas(res[0], res[1], configuracion())}
    if estado.get("resultado"):
        st.divider()
        r = estado["resultado"]
        mostrar_cambio(r["cambio"], r["a"], r["b"], modelo)
        creditos = {e.atribucion for e in (estado["a"], estado["b"]) if e.atribucion}
        for credito in sorted(creditos):
            st.caption(credito)


def fila_metricas(titulo, m):
    return {"Conjunto": titulo, "Precisión": numero(m.get("precision"), 3), "Recall": numero(m.get("recall"), 3),
            "F1": numero(m.get("f1"), 3), "IoU": numero(m.get("iou"), 3), "AUC ROC": numero(m.get("auc"), 3),
            "MAE área (ha)": numero(m.get("mae_ha"), 2), "Sesgo área (ha)": numero(m.get("sesgo_ha"), 2, signo=True),
            "TP": m.get("tp"), "FP": m.get("fp"), "FN": m.get("fn"), "TN": m.get("tn")}


def pestana_modelo(modelo, error):
    if modelo is None:
        st.warning("No hay un modelo publicado: las métricas están pendientes.")
        st.markdown(f"**Estado:** {error}")
        return
    c, met = modelo.contrato, modelo.metricas
    ds, ref = c["dataset"], c["referencia"]
    escenas_ds = ds.get("escenas") or {}
    st.markdown(
        f"**Paquete:** `{c['id']}` · variante **{c['variante']}** · {c['backends'][modelo.backend.nombre]['archivo']} "
        f"({modelo.backend.nombre}) · creado {c.get('creado_utc', '—')}  \n"
        f"**Dataset:** {ds['version']} (huella `{ds.get('huella_contenido', '—')}`) · "
        f"{escenas_ds.get('train', '—')} escenas de entrenamiento, {escenas_ds.get('val', '—')} de validación y "
        f"{escenas_ds.get('test', '—')} de prueba · selección en **{ds.get('seleccion', 'val')}**, prueba en "
        f"**{ds.get('prueba', 'test')}**  \n"
        f"**Referencia:** {ref['origen']} · revisión humana: {'sí' if ref.get('revision_humana') else 'no'}  \n"
        f"**Entrada:** imgsz {c['entrada']['imgsz']} · umbral de propuestas "
        f"{numero(c['posproceso']['umbral_propuestas'], 3)} · **umbral operativo "
        f"{numero(c['posproceso']['umbral_operativo'], 3)}** (máximo F1 en validación)")
    st.warning(ref.get("aviso", "Referencia automática: las métricas miden concordancia con pseudoetiquetas."),
               icon="⚠️")
    if not met:
        st.info("El paquete no incluye métricas: evaluación pendiente.")
        return
    filas = []
    for split, rotulo in (("val", "Validación"), ("test", "Prueba temporal")):
        for soporte, texto in (("observable", "píxel observable"),
                               ("observable_sin_ambiguos", "observable sin componentes ambiguos"),
                               ("imagen", "toda la imagen, sin control de observabilidad")):
            m = ((met.get(split) or {}).get(soporte) or {})
            if m:
                filas.append(fila_metricas(f"{rotulo} · {texto}", m))
    if filas:
        st.dataframe(filas, hide_index=True, width="stretch")
        st.caption("Clasificación binaria por píxel (positivo = nieve/hielo visible) contra pseudoetiquetas NDSI. "
                   "La precisión es TP/(TP+FP), distinta de la exactitud global. La puntuación de la ROC es el "
                   "máximo de las confianzas de las instancias que cubren el píxel: un ranking con empates, no "
                   "una probabilidad calibrada.")
    else:
        st.info("Métricas pendientes: " + str(met.get("estado", "sin evaluación registrada")))
    figuras = [(modelo.ruta / n, t) for n, t in (("roc_test.png", "ROC · prueba temporal"),
                                                  ("roc_val.png", "ROC · validación")) if (modelo.ruta / n).is_file()]
    if figuras:
        for col, (ruta, titulo) in zip(st.columns(len(figuras) + 1), figuras):
            col.image(str(ruta), caption=titulo, width="stretch")
    inst = (met.get("test") or {}).get("instancias") or {}
    if inst:
        st.caption(f"Máscaras de instancia en prueba: mAP50 {numero(inst.get('mascara_map50'), 3)} · mAP50–95 "
                   f"{numero(inst.get('mascara_map50_95'), 3)} (validador de Ultralytics).")
    estado = met.get("bitacora") or {}
    if estado:
        st.markdown("**Bitácora de verificación:** " + " · ".join(f"{k}: {v}" for k, v in estado.items()))


def pagina():
    st.title("Nieve y hielo visible · Nevado Santa Isabel")
    st.caption("¿Cuánta cobertura de nieve y hielo visible hay en una escena Sentinel-2 y dónde cambió entre dos "
               "fechas? · AOI [-75,397; 4,773; -75,348; 4,827] (WGS84) · malla EPSG:32618 de 10 m")
    with st.sidebar:
        st.header("Visualización")
        st.slider("Opacidad de la superposición", 0.0, 1.0, 0.45, 0.05, key="opacidad")
        st.checkbox("Realzar brillo (solo visualización)", value=True, key="realce",
                    help="La inferencia siempre usa la imagen sin realzar.")
        st.divider()
        lim = configuracion()["limites"]
        st.caption(f"Límites de carga: {lim['max_bytes'] / 1e6:.0f} MB y {lim['max_pixeles'] / 1e6:.0f} Mpx por "
                   "imagen. Solo se aceptan archivos de escena, nunca pesos de modelo.")
    modelo, error = modelo_activo()
    panel_como_interpretar()
    una, dos, info = st.tabs(["Analizar una imagen", "Comparar dos fechas", "Modelo y evaluación"])
    if modelo is None:
        for pestana in (una, dos):
            with pestana:
                st.error("La aplicación está lista, pero **no hay un modelo ajustado publicado**, así que no "
                         "puede presentar resultados de nieve.")
                st.markdown(f"**Qué falta:** {error}")
                st.markdown("**Qué necesita la aplicación:**\n"
                            "1. Un paquete en `models/<id>/` con `modelo.json` y los pesos ajustados (lo publica "
                            "la celda de exportación del notebook).\n"
                            "2. El puntero `models/ACTIVO` (o la variable `SANTA_ISABEL_PAQUETE`).\n"
                            "3. Para la demostración, dos escenas reales en `examples/` con su máscara de calidad.")
    else:
        with una:
            pestana_una(modelo)
        with dos:
            pestana_dos(modelo)
    with info:
        pestana_modelo(modelo, error)
    pie_de_pagina()


def pie_de_pagina():
    st.divider()
    st.markdown("<div style='text-align:center;color:#6b7280;font-size:0.85rem;padding-bottom:0.5rem'>"
                "Creado por <strong>Ing. Camilo Andrés Castañeda Galindo</strong> · "
                "Estudiante de Maestría en Inteligencia Artificial · 2026</div>", unsafe_allow_html=True)


pagina()
