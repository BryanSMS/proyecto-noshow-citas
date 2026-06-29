"""
App de predicción de inasistencias (No-show) a citas médicas.
Modelo: XGBoost entrenado sobre el dataset de Vitória, Brasil.
Proyecto: Aprendizaje Estadístico - UPAO 2026
"""

import streamlit as st
import pandas as pd
import joblib
from datetime import date

# ------------------------------------------------------------------
# 1. CONFIGURACIÓN DE LA PÁGINA
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Predicción de Inasistencias a Citas Médicas",
    page_icon="🏥",
    layout="centered"
)

# ------------------------------------------------------------------
# 2. CARGA DE MODELO Y OBJETOS DE PREPROCESAMIENTO (con caché)
# ------------------------------------------------------------------
@st.cache_resource
def cargar_artefactos():
    modelo = joblib.load("models/xgboost_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    le_barrios = joblib.load("models/label_encoder_barrio.pkl")
    columnas = joblib.load("models/feature_columns.pkl")
    return modelo, scaler, le_barrios, columnas

modelo, scaler, le_barrios, columnas_modelo = cargar_artefactos()

# ------------------------------------------------------------------
# 3. ENCABEZADO
# ------------------------------------------------------------------
st.title("🏥 Predicción de Inasistencia a Citas Médicas")
st.caption(
    "Modelo XGBoost entrenado sobre el dataset *Medical Appointment No Shows* "
    "(Vitória, Brasil). Proyecto de Aprendizaje Estadístico — UPAO 2026."
)
st.divider()

# ------------------------------------------------------------------
# 4. FORMULARIO DE ENTRADA
# ------------------------------------------------------------------
st.subheader("Datos del paciente y de la cita")

col1, col2 = st.columns(2)

with col1:
    genero = st.selectbox("Género", ["Femenino", "Masculino"])
    edad = st.number_input("Edad", min_value=0, max_value=115, value=35)
    barrio = st.selectbox("Barrio / Distrito", sorted(le_barrios.classes_))
    dia_agendamiento = st.date_input("Fecha en que se agenda la cita", value=date.today())

with col2:
    beca_social = st.checkbox("Recibe ayuda social (Scholarship)")
    hipertension = st.checkbox("Hipertensión")
    diabetes = st.checkbox("Diabetes")
    alcoholismo = st.checkbox("Antecedente de alcoholismo")
    discapacidad = st.slider("Nivel de discapacidad", 0, 4, 0)
    sms = st.checkbox("Recibirá recordatorio por SMS", value=True)
    dia_cita = st.date_input("Fecha de la cita médica", value=date.today())

st.divider()
predecir = st.button("🔍 Calcular riesgo de inasistencia", type="primary", use_container_width=True)

# ------------------------------------------------------------------
# 5. PREPROCESAMIENTO + PREDICCIÓN
# ------------------------------------------------------------------
if predecir:

    if dia_cita < dia_agendamiento:
        st.error("La fecha de la cita no puede ser anterior a la fecha de agendamiento.")
        st.stop()

    # --- Ingeniería de características temporales (igual que en el notebook) ---
    lead_time = (dia_cita - dia_agendamiento).days
    day_of_week = dia_cita.weekday()  # 0 = Lunes ... 6 = Domingo

    # --- Encoding igual que en el notebook ---
    gender_enc = 1 if genero == "Femenino" else 0
    barrio_enc = le_barrios.transform([barrio])[0]

    # --- Escalado de Age y LeadTime (mismo orden usado al entrenar el scaler) ---
    age_leadtime_escalado = scaler.transform([[edad, lead_time]])
    age_esc, leadtime_esc = age_leadtime_escalado[0]

    # --- Construcción del vector de entrada respetando el orden de entrenamiento ---
    entrada = pd.DataFrame([{
        "Gender": gender_enc,
        "Age": age_esc,
        "Barrio": barrio_enc,
        "Scholarship": int(beca_social),
        "Hipertension": int(hipertension),
        "Diabetes": int(diabetes),
        "Alcoholism": int(alcoholismo),
        "Discapacidad": discapacidad,
        "SMS_received": int(sms),
        "LeadTime": leadtime_esc,
        "DayOfWeek": day_of_week
    }])[columnas_modelo]  # reordena exactamente como X_train

    # --- Predicción ---
    probabilidad = float(modelo.predict_proba(entrada)[0][1])

    # --- Clasificación de riesgo (alineado al RF01 de tu documento) ---
    if probabilidad >= 0.5:
        nivel, color = "ALTO", "🔴"
    elif probabilidad >= 0.3:
        nivel, color = "MEDIO", "🟡"
    else:
        nivel, color = "BAJO", "🟢"

    # ------------------------------------------------------------------
    # 6. RESULTADO
    # ------------------------------------------------------------------
    st.subheader("Resultado de la predicción")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Probabilidad de inasistencia", f"{probabilidad*100:.1f}%")
    with c2:
        st.metric("Nivel de riesgo", f"{color} {nivel}")

    st.progress(min(probabilidad, 1.0))

    if nivel == "ALTO":
        st.warning(
            "⚠️ Se recomienda priorizar a este paciente en la campaña de "
            "confirmación telefónica o considerar overbooking en este bloque horario."
        )
    elif nivel == "MEDIO":
        st.info("ℹ️ Se recomienda enviar un recordatorio adicional por SMS o llamada.")
    else:
        st.success("✅ Bajo riesgo de inasistencia. No se requiere intervención adicional.")

    with st.expander("Ver variables enviadas al modelo"):
        st.dataframe(entrada)

st.divider()
st.caption(
    "Modelo: XGBoost · Recall (Test): 0.7504 · ROC-AUC (Test): 0.7305 · "
    "Dataset: Kaggle - Medical Appointment No Shows (Arroba, 2016)"
)