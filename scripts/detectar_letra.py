import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import pyttsx3  # <-- NUEVO
from collections import deque, Counter
import sys

# === Inicializar voz ===
voz = pyttsx3.init()
voz.setProperty('rate', 150)  # velocidad opcional

# Agrega la raíz del proyecto al sys.path para importar utils
ruta_base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ruta_base, ".."))

from utils.procesar_landmarks import normalizar_landmarks

# === Ruta del modelo ===
ruta_modelo = os.path.join(ruta_base, "..", "modelos", "sign_language_model.pkl")
modelo = joblib.load(ruta_modelo)

# === Inicializar MediaPipe ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# === Variables de predicción ===
historial = deque(maxlen=15)
letra_mostrada = ""
letra_anterior = ""
frase = ""

# === Captura de cámara ===
cap = cv2.VideoCapture(0)
print("[INFO] Presiona 'q' para salir. Espacio=agregar espacio, Backspace=borrar, Enter=leer y limpiar frase.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    letra_predicha = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            try:
                entrada = normalizar_landmarks(landmarks).reshape(1, -1)
                prediccion = modelo.predict(entrada)
                letra_predicha = prediccion[0]
            except Exception as e:
                print(f"[ERROR] Fallo al predecir: {e}")

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # === Filtrado por historial
    historial.append(letra_predicha)
    letra_estable = ""

    if len(historial) == historial.maxlen:
        conteo = Counter(historial)
        letra_estable, repeticiones = conteo.most_common(1)[0]

        if repeticiones >= int(historial.maxlen * 0.7):
            letra_mostrada = letra_estable

    # === Agregar letra si es diferente de la anterior
    if letra_mostrada != "" and letra_mostrada != letra_anterior:
        frase += letra_mostrada
        letra_anterior = letra_mostrada

    # === Mostrar en pantalla ===
    cv2.putText(frame, f"Letra: {letra_mostrada}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
    cv2.putText(frame, f"Frase: {frase}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

    cv2.imshow("Detección de Letra - Lenguaje de Señas", frame)

    # === Control de teclado ===
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):  # Espacio
        frase += ' '
        letra_anterior = ''
    elif key == 8:  # Backspace
        frase = frase[:-1]
        letra_anterior = ''
    elif key == 13:  # Enter → leer frase
        if frase.strip() != "":
            print(f"[VOZ] Leyendo frase: {frase}")
            voz.say(frase)
            voz.runAndWait()
        frase = ""
        letra_anterior = ""

cap.release()
cv2.destroyAllWindows()
