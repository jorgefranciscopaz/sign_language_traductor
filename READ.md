# 🧠 Traductor Inteligente de Lenguaje de Señas a Texto

Este proyecto permite capturar letras del lenguaje de señas mediante una cámara, entrenar un modelo de inteligencia artificial y traducir gestos en frases que son enviadas automáticamente a Firebase Realtime Database.

---

## 📂 Estructura del Proyecto

```
/detectar_letras/
├── /datos/                    # Datos capturados (CSV por letra)
├── /modelos/                  # Modelos entrenados (.pkl)
├── /scripts/
│   ├── capturar_datos.py      # Captura landmarks de una letra
│   ├── detectar_letra.py      # Detecta letras y arma frases
│   ├── procesar_landmarks.py  # Normaliza landmarks
│   └── venv-mediapipe/        # Entorno virtual (detección)

/entrenar_modelo/
└── entrenar_modelo.py         # Entrena el modelo de clasificación

/enviar_firebase/
├── enviar_a_firebase.py       # Envía frases al Realtime Database
├── config/firebase_key.json   # Credencial de acceso a Firebase
└── venv-firebase/             # Entorno virtual para conexión Firebase

run_app.py                     # Ejecuta simultáneamente detección y envío
```

---

## 🔧 Requisitos

Instala las dependencias del proyecto con:

```bash
pip install opencv-python mediapipe joblib firebase-admin scikit-learn pandas numpy
```

---

## 🚀 ¿Cómo utilizar el sistema?

### 1. Captura de Datos

Ejecuta:

```bash
python detectar_letras/scripts/capturar_datos.py
```

- Ingresa la letra que estás capturando (ejemplo: `A`).
- El sistema iniciará la cámara y capturará los puntos de referencia de la mano.
- Presiona `q` para finalizar la captura.
- Se generará un archivo `A.csv` en la carpeta `/datos`.

📌 Repite este proceso para cada letra que desees reconocer.

---

### 2. Entrenamiento del Modelo

Ejecuta:

```bash
python entrenar_modelo/entrenar_modelo.py
```

- Carga automáticamente los archivos CSV de `/datos/`.
- Entrena un clasificador neuronal (`MLPClassifier`) con validación cruzada (`cv=5`).
- Guarda el modelo con mayor precisión en `/modelos/sign_language_model.pkl`.

---

### 3. Detección en Tiempo Real y Construcción de Frases

Ejecuta:

```bash
python detectar_letras/scripts/detectar_letra.py
```

- Detecta automáticamente letras a partir de los movimientos de la mano.
- Construye una frase a partir de letras reconocidas de forma estable.
- Controles disponibles:
  - `Espacio`: agrega un espacio.
  - `Backspace`: elimina el último carácter.
  - `Enter`: envía la frase a Firebase.

---

## 🔗 Configuración de Firebase

📌 **Importante:** esta configuración depende de cada usuario. Deberás crear tu propia instancia de Firebase Realtime Database y generar tus credenciales.

1. Coloca tu archivo `firebase_key.json` dentro de:

```
/enviar_firebase/config/firebase_key.json
```

2. Edita la URL del Realtime Database dentro de:

```
/enviar_firebase/enviar_a_firebase.py
```

3. Las frases se almacenarán en la rama `frases/` del Realtime Database.

---

## ⚙️ Funcionamiento Interno

- MediaPipe detecta los landmarks (puntos clave) de la mano en tiempo real.
- Los puntos se normalizan para hacer el modelo robusto ante escalas o posiciones distintas.
- El modelo clasifica la letra más probable.
- Se estabilizan predicciones para evitar errores.
- Se construyen frases completas que pueden enviarse directamente a Firebase.

---

## ⚠️ Recomendaciones

- Captura múltiples muestras por letra para mejorar la precisión.
- Usa buena iluminación y posición central de la mano ante la cámara.
- **Nunca subas tu archivo `firebase_key.json` a repositorios públicos.**

---

## 📸 Créditos

- **MediaPipe** – Detección de manos en tiempo real.
- **Scikit-learn** – Clasificador neuronal (MLP).
- **Firebase Realtime Database** – Almacenamiento remoto de frases.
