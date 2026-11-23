"""Cobertura de nieve y hielo visible en el Nevado Santa Isabel (SDD 04).

Interfaces compartidas por el notebook y la aplicación Streamlit:

    leer_escena(archivos, perfil) -> Escena
    preparar_entrada(escena, contrato) -> imagen
    predecir(imagen, paquete) -> Prediccion
    medir(escena, prediccion) -> Medicion
    comparar(anterior, posterior, config) -> Cambio
    cargar_paquete(ruta) -> Modelo
"""
from .entorno import preparar_entorno

preparar_entorno()  # antes de que cualquier módulo importe rasterio

from .config import cargar_config, raiz_proyecto, resolver  # noqa: E402

__version__ = "0.1.0"
__all__ = ["cargar_config", "raiz_proyecto", "resolver", "__version__"]
