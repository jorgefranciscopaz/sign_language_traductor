#Entrena el modelo con los CSV en /datos
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# === Rutas ===
ruta_base = os.path.dirname(os.path.abspath(__file__))
ruta_datos = os.path.join(ruta_base, "..", "datos")
ruta_modelos = os.path.join(ruta_base, "..", "modelos")
os.makedirs(ruta_modelos, exist_ok=True)

# === Cargar datos ===
data = []
labels = []

for archivo in os.listdir(ruta_datos):
    if archivo.endswith(".csv"):
        df = pd.read_csv(os.path.join(ruta_datos, archivo), header=None)
        for i in range(len(df)):
            data.append(df.iloc[i, :-1])
            labels.append(df.iloc[i, -1])

X = np.array(data, dtype=np.float32)
y = np.array(labels)

# === División entrenamiento/prueba ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Entrenar modelo ===
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# === Evaluación ===
y_pred = modelo.predict(X_test)
precision = accuracy_score(y_test, y_pred)
print(f"[INFO] Precisión del modelo: {precision:.2f}")

# === Guardar modelo ===
modelo_path = os.path.join(ruta_modelos, "sign_language_model.pkl")
joblib.dump(modelo, modelo_path)
print(f"[INFO] Modelo guardado en: {modelo_path}")
