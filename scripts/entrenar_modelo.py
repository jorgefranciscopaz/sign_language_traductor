# scripts/entrenar_modelo.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import cross_validate
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

# === Inicializar modelo base ===
modelo_base = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42, verbose=False)

# === Validación cruzada con retorno de modelos entrenados ===
print("[INFO] Ejecutando validación cruzada con retorno de modelos...")
resultados = cross_validate(modelo_base, X, y, cv=5, return_estimator=True, return_train_score=True)

# === Precisión por fold
precisiones = resultados["test_score"]
for i, score in enumerate(precisiones):
    print(f"[INFO] Fold {i+1}: Precisión = {score:.3f}")

# === Mejor modelo (mayor precisión)
indice_mejor = np.argmax(precisiones)
mejor_modelo = resultados["estimator"][indice_mejor]
print(f"[INFO] Mejor modelo seleccionado del fold {indice_mejor + 1} con precisión = {precisiones[indice_mejor]:.3f}")

# === Guardar modelo
modelo_path = os.path.join(ruta_modelos, "sign_language_model.pkl")
joblib.dump(mejor_modelo, modelo_path)
print(f"[INFO] Modelo guardado en: {modelo_path}")
