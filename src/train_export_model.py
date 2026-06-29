# ============================================================
# EXPORTACIÓN DEL MODELO ENTRENADO (Capítulo VII - Despliegue)
# ============================================================
import joblib
import os
 
# 1. Crear la carpeta 'models' si no existe
os.makedirs("models", exist_ok=True)
 
# 2. Guardar el modelo XGBoost final
joblib.dump(modelo_seleccionado, 'models/xgboost_model.pkl')
 
# 3. Guardar el scaler (usado para Age y LeadTime)
joblib.dump(scaler, 'models/scaler.pkl')
 
# 4. Guardar el LabelEncoder de Barrio/Neighbourhood
joblib.dump(le_barrios, 'models/label_encoder_barrio.pkl')
 
# 5. Guardar el orden exacto de las columnas de entrada
joblib.dump(list(X_train.columns), 'models/feature_columns.pkl')
 
print("Archivos exportados correctamente en la carpeta 'models/':")
print(" - xgboost_model.pkl")
print(" - scaler.pkl")
print(" - label_encoder_barrio.pkl")
print(" - feature_columns.pkl")
print(f"\nOrden de columnas esperado por el modelo: {list(X_train.columns)}")
 