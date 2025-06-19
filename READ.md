# 🧠 Traductor de Lenguaje de Señas a Texto con IA

Este proyecto permite capturar letras del lenguaje de señas usando la cámara, entrenar un modelo de inteligencia artificial y traducir los gestos en frases que se envían automáticamente a Firebase Realtime Database.

---

## 📁 Estructura del Proyecto

```
/datos/                   # Datos capturados (archivos CSV por letra)
/modelos/                 # Modelos entrenados (.pkl)
/scripts/
  capturar_datos.py       # Captura datos de una letra
  entrenar_modelo.py      # Entrena el modelo con los datos
  detectar_letra.py       # Detecta letras en tiempo real y arma frases
  procesar_landmarks.py   # Normaliza landmarks
/enviar_firebase/
  enviar_a_firebase.py    # Envía las frases a Firebase
  config/firebase_key.json# Clave de acceso al Realtime Database
```

---

## 🔧 Requisitos

Instala las dependencias del proyecto con:

```bash
pip install opencv-python mediapipe joblib firebase-admin scikit-learn pandas numpy
```

---

## 🚀 ¿Cómo usar el sistema?

### 1. Capturar Datos

Ejecuta el siguiente script para guardar puntos de la mano asociados a una letra:

```bash
python scripts/capturar_datos.py
```

- Introduce la letra que estás capturando (ej: `A`).
- Usa la cámara para grabar los gestos.
- Presiona `q` para finalizar.
- Se generará un archivo `A.csv` en la carpeta `/datos`.

📌 Repite este proceso para varias letras.

---

### 2. Entrenar el Modelo

```bash
python scripts/entrenar_modelo.py
```

Este script:
- Carga los `.csv` desde `/datos/`.
- Entrena una red neuronal `MLPClassifier`.
- Realiza validación cruzada (`cv=5`).
- Guarda el mejor modelo en `/modelos/sign_language_model.pkl`.

---

### 3. Detectar Letras y Formar Frases

```bash
python scripts/detectar_letra.py
```

- Usa la cámara para detectar letras.
- Forma frases automáticamente.
- Controles disponibles:
  - `Espacio`: Agrega un espacio.
  - `Backspace`: Borra la última letra.
  - `Enter`: Envía la frase a Firebase.

---

## 🔗 Firebase

1. Coloca tu archivo `firebase_key.json` dentro de:  
   `/enviar_firebase/config/firebase_key.json`

2. Asegúrate de que la URL de la base de datos sea correcta en el archivo:  
   `enviar_firebase/enviar_a_firebase.py`

3. La frase se enviará a la rama `frases/` en el Realtime Database.

---

## 🧩 ¿Cómo funciona?

- MediaPipe detecta los puntos de la mano (landmarks).
- Se normalizan para hacerlos independientes del tamaño/posición.
- Se predice la letra usando un modelo entrenado.
- Se forma una frase con letras estables.
- La frase se muestra en pantalla y se puede enviar a Firebase.

---

## ⚠️ Recomendaciones

- Entrena con muchas muestras por letra para mejor precisión.
- Asegúrate de tener buena iluminación y una cámara clara.
- Nunca subas tu clave `firebase_key.json` a repositorios públicos.

---

## 📸 Créditos

- MediaPipe (detección de manos)
- Scikit-learn (modelo MLP)
- Firebase Realtime Database (envío de frases)
