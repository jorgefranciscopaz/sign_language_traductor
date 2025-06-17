# scripts/entrenar_modelo.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
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

# Convertir a arrays
X = np.array(data, dtype=np.float32)
y = np.array(labels)

# === Validaciones ===
if len(X) == 0:
    print("[ERROR] No se encontraron datos. Asegúrate de capturar muestras en la carpeta /datos.")
    exit()

if len(np.unique(y)) < 2:
    print(f"[ERROR] Solo se detectó una clase ({np.unique(y)[0]}). Entrena al menos dos letras distintas.")
    exit()

if len(X) < 20:
    print(f"[ADVERTENCIA] Tienes muy pocos datos ({len(X)} muestras). El modelo puede ser poco confiable.")

# === División entrenamiento/prueba ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Inicializar modelo MLP ===
modelo = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)

# === Validación cruzada (opcional pero informativa) ===
print("[INFO] Ejecutando validación cruzada...")
scores = cross_val_score(modelo, X, y, cv=5)
print(f"[INFO] Precisión promedio en CV (5 folds): {scores.mean():.3f} (+/- {scores.std():.3f})")

# === Entrenamiento final con todos los datos ===
modelo.fit(X, y)

# === Evaluación rápida con datos separados
y_pred = modelo.predict(X_test)
precision = accuracy_score(y_test, y_pred)
print(f"[INFO] Precisión en conjunto de prueba: {precision:.2f}")

# === Guardar modelo ===
modelo_path = os.path.join(ruta_modelos, "sign_language_model.pkl")
joblib.dump(modelo, modelo_path)
print(f"[INFO] Modelo guardado en: {modelo_path}")
