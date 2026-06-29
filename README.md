# Predicción de Inasistencias en Citas Médicas

Modelo de Aprendizaje Estadístico para predecir la probabilidad de que un paciente
falte (*no-show*) a su cita médica ambulatoria, desarrollado como Proyecto del curso
**Aprendizaje Estadístico** — Universidad Privada Antenor Orrego (UPAO), 2026.

🔗 **Demo en vivo:** [App de Predicción de Citas](https://proyecto-noshow-citas-rim75ukjn5xjdlk7wdqeuk.streamlit.app/)

---

## 📌 Descripción del problema

El ausentismo en citas médicas (*no-show*) genera pérdidas operativas y financieras
en los centros de salud: tiempo médico ocioso, listas de espera más largas y uso
ineficiente de recursos. Este proyecto desarrolla un modelo predictivo que estima,
para cada cita agendada, la probabilidad de inasistencia, permitiendo al personal
administrativo priorizar recordatorios y estrategias de overbooking inteligente.

## 📊 Dataset

- **Fuente:** [Medical Appointment No Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments) (Arroba, 2016) — Kaggle.
- **Origen:** Citas ambulatorias del sistema de salud pública de Vitória, Espírito Santo, Brasil (mayo de 2016).
- **Volumen:** 110,527 registros · 14 variables originales.
- **Variable objetivo:** `No-show` (0 = Asistió, 1 = Faltó) — distribución 79.81% / 20.19%.

## 🧪 Metodología

1. **Preprocesamiento:** eliminación de identificadores (`PatientId`, `AppointmentID`), filtrado de inconsistencias lógicas (edades negativas, fechas ilógicas).
2. **Feature Engineering:** `LeadTime` (días entre agendamiento y cita), `DayOfWeek`.
3. **Encoding:** `Gender` y `No-show` binarios; `Neighbourhood` (Barrio) con Label Encoding.
4. **Scaling:** `MinMaxScaler` sobre `Age` y `LeadTime`.
5. **Balanceo de clases:** `class_weight` / `scale_pos_weight` para penalizar Falsos Negativos.
6. **Modelos comparados:** Regresión Logística, Random Forest, **XGBoost** (seleccionado).
7. **Validación:** Cross-Validation estratificada (5-fold) + evaluación final sobre Test Set aislado (80/20).

## 🏆 Resultados del modelo final (XGBoost)

| Métrica | Validación Cruzada | Test Set |
|---|---|---|
| Recall | 0.7407 | **0.7504** |
| ROC-AUC | 0.7345 | **0.7305** |
| Precisión | 0.3186 | 0.3139 |
| F1-Score | 0.4456 | 0.4426 |

El modelo detecta correctamente **3 de cada 4 pacientes** que efectivamente faltarán
a su cita, priorizando minimizar los Falsos Negativos (turnos perdidos) sobre los
Falsos Positivos (recordatorios innecesarios).

**Variables más influyentes:** `LeadTime` (tiempo de espera) y `Age` (edad del paciente).

## 📂 Estructura del repositorio

```
proyecto-noshow-citas/
├── app.py                   # Aplicación web (Streamlit)
├── requirements.txt         # Dependencias
├── data/                    # Dataset original
├── notebooks/                # Notebook de preprocesamiento y entrenamiento
├── models/                   # Modelo (.pkl) y objetos de preprocesamiento
├── src/                      # Script de exportación del modelo
├── tests/                    # Pruebas automatizadas (CI)
├── docs/                     # Informe completo y capturas de evidencia
└── .github/workflows/        # Integración continua (GitHub Actions)
```

## ⚙️ Ejecución local

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU-USUARIO/proyecto-noshow-citas.git
cd proyecto-noshow-citas

# 2. Crear entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
streamlit run app.py
```

La app se abrirá en `http://localhost:8501`.

## ✅ Pruebas automatizadas

El proyecto incluye pruebas unitarias (`tests/test_predict.py`) que verifican que el
modelo carga correctamente y produce predicciones válidas. Se ejecutan automáticamente
en cada `push` mediante GitHub Actions (ver pestaña **Actions** del repositorio).

Para ejecutarlas manualmente:

```bash
pip install pytest
pytest tests/ -v
```

## 🚀 Despliegue

La aplicación está desplegada gratuitamente en **Streamlit Community Cloud**,
conectada directamente a este repositorio: cada `push` a la rama `main` actualiza
automáticamente la app en producción.

## 👥 Equipo

- Muñoz Samame, Bryan (Coordinador)
- Narvaez Salas, Jhosep
- Lopez Marchan, Javier
- Pereda Flores, Josep

**Docente:** Ing. Teobaldo Sagastegui Chigne
**Curso:** Aprendizaje Estadístico — Ingeniería de Sistemas e Inteligencia Artificial, UPAO

## 📚 Referencias principales

- Arroba, J. (2016). *Medical Appointment No Shows*. Kaggle.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.
- Yang, Y., Madanian, S., & Parry, D. (2024). Enhancing Health Equity by Predicting Missed Appointments. *JMIR Medical Informatics*.

## 📄 Licencia

Proyecto académico desarrollado con fines educativos.
