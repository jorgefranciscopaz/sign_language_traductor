#Usa la camara y el modelo para detectar letras
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

# === Rutas ===
ruta_base = os.path.dirname(os.path.abspath(__file__))
ruta_modelo = os.path.join(ruta_base, "..", "modelos", "sign_language_model.pkl")

# === Cargar modelo entrenado ===
modelo = joblib.load(ruta_modelo)

# === Inicializar MediaPipe ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# === Captura de cámara ===
cap = cv2.VideoCapture(0)

print("[INFO] Presiona 'q' para salir.")

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

            # Predecir letra
            entrada = np.array(landmarks).reshape(1, -1)
            prediccion = modelo.predict(entrada)
            letra_predicha = prediccion[0]

            # Dibujar la mano
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Mostrar letra predicha
    cv2.putText(frame, f"Letra: {letra_predicha}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
    cv2.imshow("Detección de Letra - Lenguaje de Señas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
