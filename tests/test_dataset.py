"""Hito B · Contrato con el generador: auditoría con una versión sintética y con v0003 real (si está extraído)."""
import csv
import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from santa_isabel import cargar_config, dataset as ds
from santa_isabel.escenas import cargar_reflectancia, codificar_espectral, codificar_rgb, leer_escena

ANCHO, ALTO = 32, 24
POLIGONO = "0 0.25000000 0.25000000 0.75000000 0.25000000 0.75000000 0.75000000 0.25000000 0.75000000\n"


def sha(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def crear_version(raiz, escenas=None, version="v9001"):
    """Versión mínima con la misma estructura que publica el generador mensual."""
    escenas = escenas or [("2016-01", "train", "L1C", POLIGONO), ("2016-07", "train", "L1C", POLIGONO * 2),
                          ("2024-01", "val", "L2A", POLIGONO), ("2025-07", "test", "L2A", "")]
    carpeta = raiz / "data" / "datasets" / version
    filas, conteos = [], {s: {"imagenes": 0, "positivas": 0, "negativas": 0, "instancias": 0} for s in ds.SPLITS}
    rng = np.random.default_rng(1)
    for mes, split, perfil, txt in escenas:
        dia = f"{mes.replace('-', '')}15"
        clave = f"{dia}_T18NVL_{dia}T152649_{dia}T152651" + ("_l1c" if perfil == "L1C" else "")
        nombre = f"{mes}_{clave}"
        hashes = {}
        for v in ("rgb", "espectral"):
            (carpeta / v / "images" / split).mkdir(parents=True, exist_ok=True)
            (carpeta / v / "labels" / split).mkdir(parents=True, exist_ok=True)
            img = carpeta / v / "images" / split / f"{nombre}.png"
            Image.fromarray(rng.integers(0, 255, (ALTO, ANCHO, 3), dtype=np.uint8)).save(img)
            hashes[v] = sha(img.read_bytes())
            (carpeta / v / "labels" / split / f"{nombre}.txt").write_bytes(txt.encode())
        n = txt.count("\n")
        coleccion = "S2_SR_HARMONIZED" if perfil == "L2A" else "S2_HARMONIZED"
        filas.append({
            "mes": mes, "split": split, "nombre": nombre, "perfil": perfil, "clave_escena": clave,
            "id_asset": f"COPERNICUS/{coleccion}/{clave}", "adquisicion_utc": f"{mes}-15T15:26:49Z",
            "grupo_temporal": f"aoi_{dia}", "decision": "positiva" if n else "negativa", "instancias": n,
            "area_pseudoetiquetada_m2": 100.0 * n, "origen": "automatico_ndsi", "revision_humana": "False",
            "provisional": "False", "seleccion_incompleta": "False",
            "imagen_rgb": f"rgb/images/{split}/{nombre}.png", "imagen_espectral": f"espectral/images/{split}/{nombre}.png",
            "etiqueta": f"<variante>/labels/{split}/{nombre}.txt", "sha256_rgb": hashes["rgb"],
            "sha256_espectral": hashes["espectral"], "sha256_txt": sha(txt.encode()),
            "receta_huella": "abc", "reglas_huella": "def"})
        c = conteos[split]
        c["imagenes"] += 1
        c["positivas" if n else "negativas"] += 1
        c["instancias"] += n
    with open(carpeta / "manifiesto.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    for v in ("rgb", "espectral"):
        (carpeta / v / "data.yaml").write_text(
            f"# Dataset de prueba\npath: C:/equipo/del/productor/salida/datasets/{version}/{v}\n"
            "train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: nieve_hielo_visible\n",
            encoding="utf-8")
    (carpeta / "configuracion.toml").write_text("[etiquetas.L2A]\nambiguos_bloquean = false\n", encoding="utf-8")
    resumen = {"version": version, "clases": {"0": "nieve_hielo_visible"}, "huella_contenido": "f00d",
               "estado_producto": {"rgb": "entrenable", "espectral": "entrenable"}, "conteos": conteos,
               "cortes": {"corte_val": "2023-09", "corte_test": "2025-04", "meses_plan": ["2016-01", "2026-12", 132]}}
    (carpeta / "resumen.json").write_text(json.dumps(resumen), encoding="utf-8")
    cfg = {"_raiz": str(raiz), "dataset": {"version": version, "raiz": "data/datasets", "variantes": ["rgb", "espectral"],
                                           "clase_id": 0, "clase_nombre": "nieve_hielo_visible", "ancho": ANCHO,
                                           "alto": ALTO, "raiz_cache": ""}}
    return cfg, carpeta


def test_version_sintetica_valida(tmp_path):
    cfg, carpeta = crear_version(tmp_path)
    aud = ds.auditar_dataset(cfg)
    assert aud.ok, aud.bloqueos
    assert aud.hashes_verificados == 4 * 4 and len(aud.instancias) == 4
    tabla = {t["split"]: t for t in aud.tabla_splits()}
    assert (tabla["train"]["escenas"], tabla["train"]["instancias"], tabla["train"]["meses_planificados"]) == (2, 3, 92)
    assert tabla["val"]["meses_planificados"] == 19 and tabla["test"]["meses_planificados"] == 21
    assert tabla["test"]["negativas"] == 1 and tabla["train"]["meses_ausentes"] == 90
    assert aud.filas[3]["_poligonos"] == []                    # TXT vacío explícito = negativo
    assert int(ds.mascara_referencia(aud.filas[0], ANCHO, ALTO).sum()) == 16 * 12
    assert any("ambiguos_bloquean" in a for a in aud.avisos) and any("validación pequeño" in a for a in aud.avisos)
    assert aud.identidad()["huella_contenido"] == "f00d" and set(ds.EQUIVALENCIAS) >= set(aud.encabezados)


def test_yaml_locales_solo_cambian_path(tmp_path):
    cfg, carpeta = crear_version(tmp_path)
    original = (carpeta / "rgb" / "data.yaml").read_text(encoding="utf-8")
    rutas = ds.yaml_locales(cfg, tmp_path / "runs" / "r1")
    assert set(rutas) == {"rgb", "espectral"} and rutas["rgb"].parent == tmp_path / "runs" / "r1" / "data_yaml"
    assert (carpeta / "rgb" / "data.yaml").read_text(encoding="utf-8") == original      # el original no se edita
    antes, despues = original.splitlines(), rutas["rgb"].read_text(encoding="utf-8").splitlines()
    distintas = [(a, d) for a, d in zip(antes, despues) if a != d]
    assert len(antes) == len(despues) and len(distintas) == 1 and distintas[0][0].startswith("path: C:/equipo")
    import yaml
    datos = yaml.safe_load(rutas["rgb"].read_text(encoding="utf-8"))
    assert datos["path"] == (carpeta / "rgb").resolve().as_posix() and datos["names"] == {0: "nieve_hielo_visible"}
    assert ds.ruta_etiqueta({"etiqueta": "<variante>/labels/val/x.txt"}, "espectral") == "espectral/labels/val/x.txt"


@pytest.mark.parametrize("sabotaje,patron", [
    ("txt_ausente", "falta la etiqueta"), ("hash_imagen", "hash de la imagen"), ("hash_txt", "hash de la etiqueta"),
    ("png_corrupto", "corrupta"), ("huerfano", "no están en el manifiesto"), ("clase", "clase 3"),
    ("txt_distinto", "difieren entre variantes"), ("sin_yaml", "Falta el archivo"), ("version", "declara la versión"),
])
def test_fallas_de_integridad_bloquean(tmp_path, sabotaje, patron):
    cfg, carpeta = crear_version(tmp_path)
    nombre = "2016-01_20160115_T18NVL_20160115T152649_20160115T152651_l1c"
    if sabotaje == "txt_ausente":
        (carpeta / "rgb" / "labels" / "train" / f"{nombre}.txt").unlink()
    elif sabotaje == "hash_imagen":
        Image.new("RGB", (ANCHO, ALTO), (1, 2, 3)).save(carpeta / "rgb" / "images" / "train" / f"{nombre}.png")
    elif sabotaje == "hash_txt":
        for v in ("rgb", "espectral"):
            (carpeta / v / "labels" / "train" / f"{nombre}.txt").write_text(POLIGONO.replace("0.25", "0.26"))
    elif sabotaje == "png_corrupto":
        (carpeta / "espectral" / "images" / "train" / f"{nombre}.png").write_bytes(b"\x89PNG roto")
    elif sabotaje == "huerfano":
        Image.new("RGB", (ANCHO, ALTO)).save(carpeta / "rgb" / "images" / "val" / "intruso.png")
    elif sabotaje == "clase":
        for v in ("rgb", "espectral"):
            (carpeta / v / "labels" / "train" / f"{nombre}.txt").write_text(POLIGONO.replace("0 ", "3 ", 1))
    elif sabotaje == "txt_distinto":
        (carpeta / "espectral" / "labels" / "train" / f"{nombre}.txt").write_text(POLIGONO * 3)
    elif sabotaje == "sin_yaml":
        (carpeta / "espectral" / "data.yaml").unlink()
    elif sabotaje == "version":
        (carpeta / "resumen.json").write_text((carpeta / "resumen.json").read_text().replace("v9001", "v0001"))
    aud = ds.auditar_dataset(cfg)
    assert not aud.ok and any(patron in b for b in aud.bloqueos), aud.bloqueos


def test_fuga_de_fechas_y_duplicados_bloquean(tmp_path):
    cfg, carpeta = crear_version(tmp_path)
    ruta = carpeta / "manifiesto.csv"
    with open(ruta, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    filas[2]["grupo_temporal"] = filas[0]["grupo_temporal"]      # la misma fecha en train y en val
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    aud = ds.auditar_dataset(cfg, verificar_hashes=False)
    assert any("fuga temporal" in b for b in aud.bloqueos)
    filas[2]["grupo_temporal"] = "aoi_20240115"
    filas[2]["split"] = "train"                                  # un mes de validación colocado en train
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    aud = ds.auditar_dataset(cfg, verificar_hashes=False)
    assert any("no respeta los cortes" in b for b in aud.bloqueos)
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in filas[0] if k != "sha256_txt"], extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)
    aud = ds.auditar_dataset(cfg)
    assert aud.bloqueos == ["manifiesto.csv: faltan las columnas sha256_txt"]


def test_sin_positivos_en_train_solo_auditoria(tmp_path):
    cfg, _ = crear_version(tmp_path, [("2016-01", "train", "L1C", ""), ("2024-01", "val", "L2A", POLIGONO)])
    aud = ds.auditar_dataset(cfg)
    assert any("solo pueden ejecutarse auditoría y fixtures" in b for b in aud.bloqueos)
    assert any("No hay escenas de test" in a for a in aud.avisos)
    cfg["dataset"]["version"] = "v0404"
    assert "No existe la carpeta" in ds.auditar_dataset(cfg).bloqueos[0]


# --- Datos reales ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real():
    cfg = cargar_config()
    if not (ds.carpeta_version(cfg) / "manifiesto.csv").is_file():
        pytest.skip("dataset real no extraído en data/datasets/")
    return cfg, ds.auditar_dataset(cfg)


@pytest.mark.datos_reales
def test_v0003_real_pasa_la_auditoria(real):
    cfg, aud = real
    assert aud.ok, aud.bloqueos
    if aud.version == "v0003":
        tabla = {t["split"]: (t["escenas"], t["instancias"], t["meses_planificados"]) for t in aud.tabla_splits()}
        assert tabla == {"train": (18, 273, 92), "val": (4, 7, 19), "test": (6, 41, 21)}
        assert aud.hashes_verificados == 112 and all(f["decision"] == "positiva" for f in aud.filas)
        assert aud.identidad()["huella_contenido"] == "ef44e9dcf906fd08"
    grupos = {}
    for f in aud.filas:
        grupos.setdefault(f["grupo_temporal"], set()).add(f["split"])
    assert all(len(s) == 1 for s in grupos.values())            # sin fuga de fechas entre splits
    for f in aud.filas:                                          # mismas claves y etiquetas en ambas variantes
        assert f["_etiqueta_rgb"].read_bytes() == f["_etiqueta_espectral"].read_bytes()


@pytest.mark.cache_productor
def test_png_publicados_equivalen_a_la_cache_del_productor(real):
    cfg, aud = real
    if ds.raiz_cache(cfg) is None:
        pytest.skip("caché del productor no configurada (configs/local.toml o SANTA_ISABEL_CACHE)")
    resueltas = 0
    for f in aud.filas[::3]:
        carpeta = ds.resolver_cache(cfg, f, aud.resumen.get("id_aoi"))
        assert carpeta is not None, f"sin coincidencia única en la caché para {f['clave_escena']}"
        escena = leer_escena(carpeta)
        publicado = np.asarray(Image.open(f["_imagen_rgb"]).convert("RGB"))
        assert np.array_equal(escena.rgb8, publicado)
        refl = cargar_reflectancia(escena)
        assert np.array_equal(codificar_rgb(refl), publicado)
        assert np.array_equal(codificar_espectral(refl), np.asarray(Image.open(f["_imagen_espectral"]).convert("RGB")))
        capas = ds.capas_cache(carpeta, f["reglas_huella"])
        assert np.array_equal(capas["observable"], escena.observable)
        referencia = ds.mascara_referencia(f, escena.ancho, escena.alto)
        assert referencia.sum() * 100.0 == pytest.approx(float(f["area_pseudoetiquetada_m2"]), rel=0.02)
        assert not (referencia & capas["ambiguos"]).any()         # lo ambiguo quedó como fondo en la etiqueta
        resueltas += 1
    assert resueltas >= 1
