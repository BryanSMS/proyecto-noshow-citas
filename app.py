"""
Sistema Predictivo de Inasistencias a Citas Médicas
Modelo: XGBoost — dataset Medical Appointment No Shows (Vitória, Brasil)
Proyecto: Aprendizaje Estadístico — UPAO 2026

RF01: Predicción individual de riesgo (pestaña "Paciente individual")
RF02: Dashboard de agenda diaria con alertas de color (pestaña "Agenda del día")
"""

import mysql.connector
import streamlit as st
import pandas as pd
import joblib
from datetime import date

# ──────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sistema Predictivo — Inasistencias a Citas Médicas",
    page_icon="🏥",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────
# ESTILOS GLOBALES
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

.app-header {
    background: linear-gradient(135deg, #1a3a5c 0%, #2d6a9f 100%);
    border-radius: 12px;
    padding: 1.4rem 2rem;
    margin-bottom: 1.5rem;
    color: white;
}
.app-header h1 { margin: 0; font-size: 1.7rem; font-weight: 700; letter-spacing: -0.3px; }
.app-header p  { margin: 0.3rem 0 0; font-size: 0.82rem; opacity: 0.8; }

.summary-card {
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
    font-weight: 700;
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
}
.card-alto  { background: #fee2e2; color: #991b1b; border-left: 5px solid #ef4444; }
.card-medio { background: #fef9c3; color: #854d0e; border-left: 5px solid #eab308; }
.card-bajo  { background: #dcfce7; color: #166534; border-left: 5px solid #22c55e; }
.card-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em;
              text-transform: uppercase; margin-top: 0.2rem; }

.badge {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-alto  { background: #fca5a5; color: #7f1d1d; }
.badge-medio { background: #fde68a; color: #78350f; }
.badge-bajo  { background: #86efac; color: #14532d; }

.form-section {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.form-section h4 {
    margin: 0 0 0.8rem;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #64748b;
    font-weight: 600;
}

.result-box { border-radius: 12px; padding: 1.4rem 1.8rem; margin-top: 1rem; }
.result-alto  { background: #fee2e2; border-left: 6px solid #ef4444; }
.result-medio { background: #fef9c3; border-left: 6px solid #eab308; }
.result-bajo  { background: #dcfce7; border-left: 6px solid #22c55e; }
.result-title { font-size: 1.05rem; font-weight: 700; margin: 0 0 0.3rem; }
.result-pct   { font-size: 2.2rem; font-weight: 800; margin: 0; }
.result-msg   { font-size: 0.83rem; margin: 0.6rem 0 0; opacity: 0.85; }

.prog-bar-bg {
    background: #e2e8f0; border-radius: 999px; height: 10px;
    margin: 0.8rem 0 0.3rem; overflow: hidden;
}
.prog-bar-fill { height: 100%; border-radius: 999px; transition: width 0.4s ease; }

.footer {
    font-size: 0.73rem; color: #94a3b8;
    text-align: center; margin-top: 2.5rem; padding-top: 1rem;
    border-top: 1px solid #e2e8f0;
}

/* Oculta el icono de enlace (#) que Streamlit agrega automáticamente
   a los encabezados — no aporta nada útil dentro de la app. */
[data-testid="stHeaderActionElements"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# CARGA DE ARTEFACTOS
# ──────────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_artefactos():
    modelo    = joblib.load("models/xgboost_model.pkl")
    scaler    = joblib.load("models/scaler.pkl")
    le_barrio = joblib.load("models/label_encoder_barrio.pkl")
    columnas  = joblib.load("models/feature_columns.pkl")
    return modelo, scaler, le_barrio, columnas

modelo, scaler, le_barrios, columnas_modelo = cargar_artefactos()

# ──────────────────────────────────────────────────────────────────
# FUNCIÓN COMPARTIDA DE PREPROCESAMIENTO + PREDICCIÓN
# ──────────────────────────────────────────────────────────────────
def predecir_pacientes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["FechaAgendamiento"] = pd.to_datetime(df["FechaAgendamiento"])
    df["FechaCita"]         = pd.to_datetime(df["FechaCita"])
    df["LeadTime"]          = (df["FechaCita"] - df["FechaAgendamiento"]).dt.days
    df["DayOfWeek"]         = df["FechaCita"].dt.dayofweek
    df["Gender"]            = df["Genero"].map({"Femenino": 1, "Masculino": 0})
    df["Barrio_enc"]        = le_barrios.transform(df["Barrio"])
    esc                     = scaler.transform(df[["Edad", "LeadTime"]].rename(columns={"Edad": "Age"}))
    df["Age_esc"]           = esc[:, 0]
    df["LT_esc"]            = esc[:, 1]

    entrada = pd.DataFrame({
        "Gender":       df["Gender"],
        "Age":          df["Age_esc"],
        "Barrio":       df["Barrio_enc"],
        "Scholarship":  df["Scholarship"],
        "Hipertension": df["Hipertension"],
        "Diabetes":     df["Diabetes"],
        "Alcoholism":   df["Alcoholism"],
        "Discapacidad": df["Discapacidad"],
        "SMS_received": df["SMS_received"],
        "LeadTime":     df["LT_esc"],
        "DayOfWeek":    df["DayOfWeek"],
    })[columnas_modelo]

    probs              = modelo.predict_proba(entrada)[:, 1]
    df["Probabilidad"] = [float(p) for p in probs]
    df["NivelRiesgo"]  = pd.cut(
        df["Probabilidad"],
        bins=[-0.01, 0.30, 0.50, 1.0],
        labels=["BAJO", "MEDIO", "ALTO"]
    ).astype(str)

    for col in columnas_modelo:
        df[f"_modelo_{col}"] = entrada[col].values

    return df

# ──────────────────────────────────────────────────────────────────
# RF05 — AUDITORÍA Y REGISTRO DE RECLASIFICACIÓN MANUAL
# Persistido en MySQL (Aiven), no en archivo ni en session_state,
# para que sobreviva recargas, cierres del navegador y sea visible
# para todo el personal que abra la app (local o en la nube).
# ──────────────────────────────────────────────────────────────────

def conectar_bd():
    return mysql.connector.connect(
        host=st.secrets["mysql"]["host"],
        port=st.secrets["mysql"]["port"],
        user=st.secrets["mysql"]["user"],
        password=st.secrets["mysql"]["password"],
        database=st.secrets["mysql"]["database"],
        ssl_disabled=False,
    )

def cargar_ediciones() -> pd.DataFrame:
    try:
        conn = conectar_bd()
        df = pd.read_sql("SELECT * FROM ediciones_manuales", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(
            f"No se pudo conectar a la base de datos: {e}\n\n"
            "Si usas el plan gratuito de Aiven, el servicio puede haberse "
            "apagado por inactividad — revisa tu panel de Aiven y reactívalo."
        )
        return pd.DataFrame(columns=[
            "id", "Timestamp", "PacienteID", "Nombre",
            "EstadoAnterior", "EstadoNuevo", "Observacion"
        ])

def estado_vigente_por_paciente(ediciones: pd.DataFrame) -> dict:
    """Devuelve, por cada PacienteID, el ÚLTIMO estado registrado
    (el más reciente por Timestamp), que es el que debe regir hoy."""
    if ediciones.empty:
        return {}
    ediciones = ediciones.copy()
    ediciones["Timestamp"] = pd.to_datetime(ediciones["Timestamp"])
    ultimo_por_paciente = ediciones.sort_values("Timestamp").groupby("PacienteID").tail(1)
    return dict(zip(ultimo_por_paciente["PacienteID"], ultimo_por_paciente["EstadoNuevo"]))

def guardar_edicion(paciente_id, nombre, estado_anterior, estado_nuevo, observacion):
    try:
        conn = conectar_bd()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO ediciones_manuales "
            "(Timestamp, PacienteID, Nombre, EstadoAnterior, EstadoNuevo, Observacion) "
            "VALUES (NOW(), %s, %s, %s, %s, %s)",
            (paciente_id, nombre, estado_anterior, estado_nuevo, observacion)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"No se pudo guardar el cambio en la base de datos: {e}")
        return False

def reiniciar_ediciones():
    try:
        conn = conectar_bd()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM ediciones_manuales")
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"No se pudo reiniciar el historial: {e}")
        return False

# ──────────────────────────────────────────────────────────────────
# HEADER GLOBAL
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>🏥 Sistema Predictivo de Inasistencias a Citas Médicas</h1>
  <p>Modelo XGBoost · Dataset Medical Appointment No Shows (Vitória, Brasil) · UPAO 2026</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["👤  Paciente individual", "📋  Agenda del día", "🛏️  Overbooking"])

# ══════════════════════════════════════════════════════════════════
# PESTAÑA 1 — RF01
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("##### Ingresa los datos del paciente y la cita para calcular el riesgo de inasistencia.")
    col_izq, col_der = st.columns([1, 1], gap="large")

    with col_izq:
        with st.container(border=True):
            st.markdown("**Datos del paciente**")
            genero = st.selectbox("Género", ["Femenino", "Masculino"], key="ind_genero")
            edad   = st.number_input("Edad", min_value=0, max_value=115, value=35, key="ind_edad")
            barrio = st.selectbox("Barrio / Distrito", sorted(le_barrios.classes_), key="ind_barrio")
            beca   = st.checkbox("Recibe beca social (Scholarship)", key="ind_beca")

        with st.container(border=True):
            st.markdown("**Condiciones de salud**")
            c1, c2 = st.columns(2)
            with c1:
                hiper   = st.checkbox("Hipertensión", key="ind_hiper")
                alcohol = st.checkbox("Alcoholismo",  key="ind_alcohol")
            with c2:
                diabetes = st.checkbox("Diabetes", key="ind_diabetes")
            discap = st.slider("Nivel de discapacidad", 0, 4, 0, key="ind_discap")

    with col_der:
        with st.container(border=True):
            st.markdown("**Datos de la cita**")
            dia_agenda = st.date_input("Fecha de agendamiento",   value=date.today(), key="ind_agenda")
            dia_cita   = st.date_input("Fecha de la cita médica", value=date.today(), key="ind_cita")
            sms        = st.checkbox("Recibirá recordatorio SMS", value=True, key="ind_sms")

            if dia_cita - dia_agenda > pd.Timedelta(days=179):
                st.caption(
                    "⚠️ Esta espera supera los 179 días que el modelo conoció en su "
                    "entrenamiento; la predicción puede no ser confiable."
                )

        calcular = st.button("Calcular riesgo", type="primary",
                             use_container_width=True, key="btn_ind")

        if calcular:
            if dia_cita < dia_agenda:
                st.error("La fecha de la cita no puede ser anterior a la del agendamiento.")
                st.stop()

            fila = pd.DataFrame([{
                "Genero": genero, "Edad": edad, "Barrio": barrio,
                "FechaAgendamiento": dia_agenda.isoformat(),
                "FechaCita": dia_cita.isoformat(),
                "Scholarship": int(beca), "Hipertension": int(hiper),
                "Diabetes": int(diabetes), "Alcoholism": int(alcohol),
                "Discapacidad": discap, "SMS_received": int(sms),
            }])
            res   = predecir_pacientes(fila)
            prob  = res["Probabilidad"].iloc[0]
            nivel = res["NivelRiesgo"].iloc[0]
            pct   = f"{prob * 100:.1f}%"

            clase  = {"ALTO": "result-alto",  "MEDIO": "result-medio",  "BAJO": "result-bajo"}[nivel]
            icono  = {"ALTO": "🔴",            "MEDIO": "🟡",            "BAJO": "🟢"}[nivel]
            color  = {"ALTO": "#ef4444",       "MEDIO": "#eab308",       "BAJO": "#22c55e"}[nivel]
            msg    = {
                "ALTO":  "Se recomienda priorizar confirmación telefónica o considerar overbooking.",
                "MEDIO": "Se recomienda enviar un recordatorio adicional por SMS o llamada.",
                "BAJO":  "Bajo riesgo de inasistencia. No se requiere intervención adicional.",
            }[nivel]

            st.markdown(f"""
            <div class="result-box {clase}">
              <p class="result-title">{icono} Nivel de riesgo: <strong>{nivel}</strong></p>
              <p class="result-pct">{pct}</p>
              <div class="prog-bar-bg">
                <div class="prog-bar-fill" style="width:{min(prob*100,100):.1f}%; background:{color};"></div>
              </div>
              <p class="result-msg">{msg}</p>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Ver vector enviado al modelo"):
                cols_debug = [f"_modelo_{c}" for c in columnas_modelo]
                tabla_debug = res[cols_debug].copy()
                tabla_debug.columns = columnas_modelo
                st.dataframe(tabla_debug, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PESTAÑA 2 — RF02
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("##### Carga la agenda del día en formato CSV para ver el riesgo de todos los pacientes de un vistazo.")

    with st.expander("¿Qué formato debe tener el CSV?", expanded=False):
        st.markdown("""
El archivo debe tener exactamente estas columnas (sensibles a mayúsculas):

| Columna | Tipo | Ejemplo |
|---|---|---|
| `PacienteID` | texto | PAC-1001 |
| `Nombre` | texto | Maria Silva |
| `Genero` | texto | Femenino / Masculino |
| `Edad` | entero | 45 |
| `Barrio` | texto | JARDIM CAMBURI |
| `FechaAgendamiento` | fecha YYYY-MM-DD | 2026-06-20 |
| `FechaCita` | fecha YYYY-MM-DD | 2026-06-29 |
| `HoraCita` | texto | 09:30 |
| `Scholarship` | 0 / 1 | 0 |
| `Hipertension` | 0 / 1 | 1 |
| `Diabetes` | 0 / 1 | 0 |
| `Alcoholism` | 0 / 1 | 0 |
| `Discapacidad` | 0 – 4 | 0 |
| `SMS_received` | 0 / 1 | 1 |
| `Telefono` | texto (opcional) | 999-123-456 |

Si no subes ningún archivo, se carga automáticamente `data/agenda_ejemplo.csv` del repositorio.
        """)

    archivo = st.file_uploader(
        "Sube un CSV propio (opcional — si no subes nada, se usa la agenda de ejemplo)",
        type=["csv"], key="uploader_agenda"
    )

    import os
    ruta_ejemplo = os.path.join("data", "agenda_ejemplo.csv")

    if archivo is not None:
        fuente = archivo
        st.caption(f"📄 Mostrando: **{archivo.name}** (archivo subido por ti)")
    elif os.path.exists(ruta_ejemplo):
        fuente = ruta_ejemplo
        st.caption("📄 Mostrando: **data/agenda_ejemplo.csv** (agenda de ejemplo del repositorio)")
    else:
        fuente = None
        st.warning(
            "No se encontró `data/agenda_ejemplo.csv` en el repositorio y no subiste "
            "ningún archivo. Sube un CSV para continuar."
        )

    if fuente is not None:
        try:
            df_raw = pd.read_csv(fuente)

            cols_req = ["PacienteID","Nombre","Genero","Edad","Barrio",
                        "FechaAgendamiento","FechaCita","HoraCita",
                        "Scholarship","Hipertension","Diabetes",
                        "Alcoholism","Discapacidad","SMS_received"]
            faltantes = [c for c in cols_req if c not in df_raw.columns]
            if faltantes:
                st.error(f"Columnas faltantes en el CSV: {faltantes}")
                st.stop()

            if "Telefono" not in df_raw.columns:
                df_raw["Telefono"] = "No registrado"

            barrios_invalidos = set(df_raw["Barrio"]) - set(le_barrios.classes_)
            if barrios_invalidos:
                st.warning(f"Barrios no reconocidos (se omitirán): {barrios_invalidos}")
                df_raw = df_raw[~df_raw["Barrio"].isin(barrios_invalidos)].reset_index(drop=True)

            if df_raw.empty:
                st.error("No quedan registros válidos.")
                st.stop()

            with st.spinner("Calculando riesgo para cada paciente..."):
                df_pred = predecir_pacientes(df_raw)

                ediciones = cargar_ediciones()
                estados   = estado_vigente_por_paciente(ediciones)
                df_pred["EstadoManual"] = df_pred["PacienteID"].map(estados).fillna("Sin cambios")

                st.session_state["df_pred_agenda"] = df_pred

            df_activos = df_pred[~df_pred["EstadoManual"].isin(["Cancelada", "Reprogramada"])]

            n_alto  = (df_activos["NivelRiesgo"] == "ALTO").sum()
            n_medio = (df_activos["NivelRiesgo"] == "MEDIO").sum()
            n_bajo  = (df_activos["NivelRiesgo"] == "BAJO").sum()
            n_total = len(df_pred)
            n_inactivos = len(df_pred) - len(df_activos)

            st.markdown(
                f"**{n_total} pacientes · {date.today().strftime('%d/%m/%Y')}**"
                + (f" &nbsp;·&nbsp; {n_inactivos} cancelados/reprogramados (excluidos del cálculo)"
                   if n_inactivos > 0 else "")
            )
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown(f'<div class="summary-card card-alto">{n_alto}<div class="card-label">🔴 Alto riesgo</div></div>', unsafe_allow_html=True)
            with mc2:
                st.markdown(f'<div class="summary-card card-medio">{n_medio}<div class="card-label">🟡 Riesgo medio</div></div>', unsafe_allow_html=True)
            with mc3:
                st.markdown(f'<div class="summary-card card-bajo">{n_bajo}<div class="card-label">🟢 Bajo riesgo</div></div>', unsafe_allow_html=True)

            st.markdown("---")

            filtro = st.radio(
                "Filtrar por nivel",
                ["Todos", "🔴 Solo ALTO", "🟡 Solo MEDIO", "🟢 Solo BAJO"],
                horizontal=True, key="filtro_riesgo"
            )
            mapa = {"Todos": None, "🔴 Solo ALTO": "ALTO", "🟡 Solo MEDIO": "MEDIO", "🟢 Solo BAJO": "BAJO"}
            nivel_filtro = mapa[filtro]
            if nivel_filtro is None:
                df_vista = df_pred
            else:
                df_vista = df_activos[df_activos["NivelRiesgo"] == nivel_filtro]
            df_vista = df_vista.sort_values("HoraCita").reset_index(drop=True)

            badge_html = {
                "ALTO":  '<span class="badge badge-alto">ALTO</span>',
                "MEDIO": '<span class="badge badge-medio">MEDIO</span>',
                "BAJO":  '<span class="badge badge-bajo">BAJO</span>',
            }
            color_fila = {"ALTO": "#fee2e2", "MEDIO": "#fef9c3", "BAJO": "#f0fdf4"}

            filas_html = ""
            for _, f in df_vista.iterrows():
                nv = f["NivelRiesgo"]
                estado_manual = f["EstadoManual"]
                inactivo = estado_manual in ("Cancelada", "Reprogramada")
                bg = "#f1f5f9" if inactivo else color_fila[nv]
                estilo_texto = "text-decoration:line-through;color:#94a3b8;" if inactivo else ""
                etiqueta_estado = (
                    f'<span style="font-size:0.72rem;font-weight:700;color:#dc2626;">{estado_manual.upper()}</span>'
                    if inactivo else '<span style="color:#94a3b8;font-size:0.78rem;">Sin cambios</span>'
                )
                filas_html += f"""
                <tr style="background:{bg};{estilo_texto}">
                  <td style="padding:0.5rem 0.8rem;font-weight:600;">{f['HoraCita']}</td>
                  <td style="padding:0.5rem 0.8rem;color:#64748b;font-size:0.82rem;">{f['PacienteID']}</td>
                  <td style="padding:0.5rem 0.8rem;">{f['Nombre']}</td>
                  <td style="padding:0.5rem 0.8rem;text-align:center;">{int(f['Edad'])}</td>
                  <td style="padding:0.5rem 0.8rem;font-size:0.82rem;">{f['Barrio']}</td>
                  <td style="padding:0.5rem 0.8rem;text-align:center;font-weight:700;">{f['Probabilidad']*100:.1f}%</td>
                  <td style="padding:0.5rem 0.8rem;text-align:center;">{badge_html[nv]}</td>
                  <td style="padding:0.5rem 0.8rem;text-align:center;">{etiqueta_estado}</td>
                </tr>"""

            st.markdown(f"""
            <table style="width:100%;border-collapse:collapse;border-radius:10px;overflow:hidden;font-size:0.85rem;">
              <thead>
                <tr style="background:#1a3a5c;color:white;">
                  <th style="padding:0.6rem 0.8rem;text-align:left;">Hora</th>
                  <th style="padding:0.6rem 0.8rem;text-align:left;">ID</th>
                  <th style="padding:0.6rem 0.8rem;text-align:left;">Paciente</th>
                  <th style="padding:0.6rem 0.8rem;text-align:center;">Edad</th>
                  <th style="padding:0.6rem 0.8rem;text-align:left;">Barrio</th>
                  <th style="padding:0.6rem 0.8rem;text-align:center;">Riesgo %</th>
                  <th style="padding:0.6rem 0.8rem;text-align:center;">Nivel</th>
                  <th style="padding:0.6rem 0.8rem;text-align:center;">Estado</th>
                </tr>
              </thead>
              <tbody>{filas_html}</tbody>
            </table>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### 📞 Lista de priorización — Campaña de confirmación (RF04)")
            st.caption(
                "Pacientes activos ordenados de mayor a menor riesgo, listos para "
                "exportar o contactar por SMS/llamada. Excluye a los ya cancelados "
                "o reprogramados."
            )

            umbral_pct = st.slider(
                "Mostrar pacientes con riesgo mayor o igual a:",
                min_value=0, max_value=100, value=50, step=5,
                key="rf04_umbral", format="%d%%"
            )
            umbral = umbral_pct / 100.0

            df_prioridad = df_activos[df_activos["Probabilidad"] >= umbral].sort_values(
                "Probabilidad", ascending=False
            ).reset_index(drop=True)

            if len(df_prioridad) == 0:
                st.info("Ningún paciente activo supera ese umbral de riesgo.")
            else:
                tabla_prioridad = df_prioridad[
                    ["PacienteID", "Nombre", "Telefono", "HoraCita", "Probabilidad", "NivelRiesgo"]
                ].copy()
                tabla_prioridad["Probabilidad"] = tabla_prioridad["Probabilidad"].apply(
                    lambda p: f"{p*100:.1f}%"
                )
                st.dataframe(tabla_prioridad, use_container_width=True, hide_index=True)

                st.download_button(
                    label=f"📞 Descargar lista de priorización — {len(df_prioridad)} pacientes (CSV)",
                    data=tabla_prioridad.to_csv(index=False).encode("utf-8"),
                    file_name=f"prioridad_confirmacion_{date.today().isoformat()}.csv",
                    mime="text/csv",
                    key="dl_prioridad"
                )

            st.markdown("---")
            with st.container(border=True):
                st.markdown("**✏️ Registrar confirmación telefónica (RF05)**")
                st.caption(
                    "Si el paciente llamó para cancelar o reprogramar, regístralo aquí. "
                    "El cambio se guarda de inmediato y afecta el cálculo de riesgo y "
                    "de overbooking en tiempo real, incluso si recargas la página."
                )

                opciones_pac = [f"{r.PacienteID} — {r.Nombre}" for r in df_pred.itertuples()]
                col_e1, col_e2 = st.columns([2, 1])
                with col_e1:
                    seleccion = st.selectbox("Paciente", opciones_pac, key="rf05_paciente")

                pid_actual = seleccion.split(" — ")[0]
                estado_actual_pid = df_pred.loc[
                    df_pred["PacienteID"] == pid_actual, "EstadoManual"
                ].iloc[0]

                opciones_estado = ["Sin cambios (revertir)", "Cancelada", "Reprogramada"]
                mapa_estado_a_opcion = {
                    "Sin cambios": "Sin cambios (revertir)",
                    "Cancelada": "Cancelada",
                    "Reprogramada": "Reprogramada",
                }
                indice_default = opciones_estado.index(
                    mapa_estado_a_opcion.get(estado_actual_pid, "Sin cambios (revertir)")
                )

                with col_e2:
                    nuevo_estado = st.selectbox(
                        "Nuevo estado", opciones_estado,
                        index=indice_default, key="rf05_estado"
                    )
                st.caption(f"Estado actual de {seleccion.split(' — ',1)[1]}: **{estado_actual_pid}**")

                observacion = st.text_input(
                    "Observación (opcional)", placeholder="Ej. reprogramó para la próxima semana",
                    key="rf05_obs"
                )

                if st.button("Guardar cambio", key="rf05_guardar"):
                    pid_sel = pid_actual
                    nombre_sel = seleccion.split(" — ", 1)[1]
                    estado_final = "Sin cambios" if nuevo_estado.startswith("Sin cambios") else nuevo_estado

                    if estado_final == estado_actual_pid:
                        st.warning(
                            f"{nombre_sel} ya está en estado **{estado_actual_pid}** — "
                            "no hay ningún cambio que registrar."
                        )
                    else:
                        if guardar_edicion(pid_sel, nombre_sel, estado_actual_pid, estado_final, observacion):
                            st.success(f"Registrado: {nombre_sel} → {estado_final}")
                            st.rerun()

                ediciones_actuales = cargar_ediciones()
                if not ediciones_actuales.empty:
                    with st.expander(f"📜 Historial de cambios ({len(ediciones_actuales)} registros)"):
                        st.dataframe(
                            ediciones_actuales.sort_values("Timestamp", ascending=False),
                            use_container_width=True, hide_index=True
                        )
                        st.markdown("")
                        confirmar_reinicio = st.checkbox(
                            "Confirmo que quiero borrar TODO el historial de cambios de hoy",
                            key="rf05_confirmar_reinicio"
                        )
                        if st.button("🔄 Reiniciar ediciones del día", key="rf05_reiniciar",
                                     disabled=not confirmar_reinicio):
                            if reiniciar_ediciones():
                                st.success("Historial de ediciones reiniciado.")
                                st.rerun()

            st.markdown("")
            if len(df_vista) > 0:
                df_exportar = df_vista[
                    ["HoraCita","PacienteID","Nombre","Edad","Barrio","Probabilidad","NivelRiesgo","EstadoManual"]
                ].copy()
                df_exportar["Probabilidad"] = df_exportar["Probabilidad"].apply(lambda p: f"{p*100:.1f}%")
                etiqueta_filtro = "todos los niveles" if nivel_filtro is None else f"nivel {nivel_filtro}"
                st.download_button(
                    label=f"Descargar esta vista — {len(df_vista)} pacientes ({etiqueta_filtro}) (CSV)",
                    data=df_exportar.to_csv(index=False).encode("utf-8"),
                    file_name=f"agenda_{(nivel_filtro or 'todos').lower()}_{date.today().isoformat()}.csv",
                    mime="text/csv",
                    key="dl_vista"
                )

        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")

# ══════════════════════════════════════════════════════════════════
# PESTAÑA 3 — RF03: RECOMENDACIÓN DE OVERBOOKING
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        "##### Analiza la acumulación de riesgo por bloque horario y recomienda "
        "dónde conviene habilitar un cupo de sobreventa (overbooking)."
    )

    if "df_pred_agenda" not in st.session_state:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#94a3b8;">
          <div style="font-size:3rem;margin-bottom:1rem;">🛏️</div>
          <p style="font-size:1rem;font-weight:600;color:#64748b;">
            Aún no hay una agenda cargada
          </p>
          <p style="font-size:0.85rem;">
            Ve primero a la pestaña <strong>Agenda del día</strong> para cargar o
            calcular el riesgo de los pacientes. Este panel reutiliza ese cálculo.
          </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        df_pred = st.session_state["df_pred_agenda"]

        capacidad = st.number_input(
            "Capacidad normal por bloque horario (pacientes que el consultorio "
            "puede atender en el mismo horario sin sobreventa)",
            min_value=1, max_value=10, value=2, step=1, key="capacidad_bloque"
        )
        st.caption(
            "Este valor lo define el administrador según los médicos/consultorios "
            "disponibles — no depende del CSV cargado."
        )
        st.markdown("---")

        if "EstadoManual" in df_pred.columns:
            df_para_bloques = df_pred[~df_pred["EstadoManual"].isin(["Cancelada", "Reprogramada"])]
        else:
            df_para_bloques = df_pred

        resumen = (
            df_para_bloques.groupby("HoraCita")
            .agg(
                PacientesAgendados=("PacienteID", "count"),
                RiesgoAcumulado=("Probabilidad", "sum"),
            )
            .reset_index()
            .sort_values("HoraCita")
        )

        def evaluar_bloque(row):
            agendados = row["PacientesAgendados"]
            riesgo    = row["RiesgoAcumulado"]

            if agendados > capacidad:
                return "⚠️ Ya excede capacidad", "#fee2e2"
            elif agendados == capacidad:
                if riesgo >= 1.0:
                    return "🟢 Recomendado (+1 cupo)", "#dcfce7"
                else:
                    return "🔒 Lleno, sin riesgo suficiente", "#f1f5f9"
            else:
                return "— Cupo disponible (dentro de capacidad)", "#f8fafc"

        resumen[["Recomendacion", "_color"]] = resumen.apply(
            lambda r: pd.Series(evaluar_bloque(r)), axis=1
        )

        n_recomendados = (resumen["Recomendacion"].str.contains("Recomendado")).sum()
        st.markdown(f"**{n_recomendados} de {len(resumen)} bloques horarios** con overbooking recomendado.")
        st.markdown("")

        filas_html = ""
        for _, r in resumen.iterrows():
            filas_html += f"""
            <tr style="background:{r['_color']};">
              <td style="padding:0.5rem 0.8rem;font-weight:600;">{r['HoraCita']}</td>
              <td style="padding:0.5rem 0.8rem;text-align:center;">{r['PacientesAgendados']}</td>
              <td style="padding:0.5rem 0.8rem;text-align:center;">{capacidad}</td>
              <td style="padding:0.5rem 0.8rem;text-align:center;font-weight:700;">{r['RiesgoAcumulado']:.2f}</td>
              <td style="padding:0.5rem 0.8rem;">{r['Recomendacion']}</td>
            </tr>"""

        st.markdown(f"""
        <table style="width:100%;border-collapse:collapse;border-radius:10px;overflow:hidden;font-size:0.85rem;">
          <thead>
            <tr style="background:#1a3a5c;color:white;">
              <th style="padding:0.6rem 0.8rem;text-align:left;">Bloque horario</th>
              <th style="padding:0.6rem 0.8rem;text-align:center;">Agendados</th>
              <th style="padding:0.6rem 0.8rem;text-align:center;">Capacidad</th>
              <th style="padding:0.6rem 0.8rem;text-align:center;">Riesgo acumulado (Σ prob.)</th>
              <th style="padding:0.6rem 0.8rem;text-align:left;">Recomendación</th>
            </tr>
          </thead>
          <tbody>{filas_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        with st.expander("¿Cómo se calcula esto?"):
            st.markdown("""
- **Riesgo acumulado** = suma de las probabilidades individuales de inasistencia
  de todos los pacientes agendados en ese bloque horario. Matemáticamente
  representa el *número esperado de pacientes que van a faltar* en ese bloque.
- **— Cupo disponible**: el bloque tiene menos pacientes que la capacidad
  definida, es decir, ya hay espacio libre dentro de lo normal. No aplica
  "overbooking" porque no hace falta sobrevender nada.
- **🔒 Lleno, sin riesgo suficiente**: el bloque está exactamente en su
  capacidad máxima, pero el riesgo acumulado es menor a 1 — no hay suficiente
  probabilidad de que alguien falte como para justificar un cupo extra.
- **🟢 Recomendado (+1 cupo)**: el bloque está lleno **y** el riesgo acumulado
  es ≥ 1, es decir, se espera que al menos un paciente falte. Se recomienda
  agendar un paciente adicional como respaldo para ese horario.
- **⚠️ Ya excede capacidad**: el bloque ya tiene más pacientes agendados que
  la capacidad definida — no se recomienda seguir sumando citas, para no
  saturar al médico si todos los pacientes llegaran.
            """)

# ──────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  Modelo: XGBoost &nbsp;·&nbsp; Recall (Test): 0.7504 &nbsp;·&nbsp;
  ROC-AUC (Test): 0.7305 &nbsp;·&nbsp;
  Dataset: Kaggle — Medical Appointment No Shows (Arroba, 2016)
</div>
""", unsafe_allow_html=True)