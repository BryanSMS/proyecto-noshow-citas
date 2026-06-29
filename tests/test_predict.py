"""
Pruebas básicas del sistema predictivo.
Estas pruebas se ejecutan automáticamente en GitHub Actions
(pestaña "Actions" del repositorio) cada vez que se sube código nuevo.
Sirven como evidencia formal de que el modelo carga y predice
correctamente (Capítulo VII, sección 7.1.3).
"""
 
import os
import joblib
import numpy as np
 
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
 
 
def test_el_modelo_carga_correctamente():
    """Verifica que el archivo del modelo existe y se puede cargar."""
    modelo = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    assert modelo is not None
 
 
def test_el_scaler_carga_correctamente():
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    assert scaler is not None
 
 
def test_columnas_coinciden_con_el_entrenamiento():
    columnas = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    assert len(columnas) == 11
    assert "LeadTime" in columnas
    assert "Age" in columnas
 
 
def test_prediccion_devuelve_probabilidad_valida():
    """Verifica que la salida del modelo sea una probabilidad entre 0 y 1."""
    modelo = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    columnas = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
 
    # Paciente ficticio de prueba (vector de ceros del tamaño correcto)
    entrada_ejemplo = np.zeros((1, len(columnas)))
    probabilidad = modelo.predict_proba(entrada_ejemplo)[0][1]
 
    assert 0.0 <= probabilidad <= 1.0
 