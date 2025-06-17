#Captura landmarks de una letra
import cv2
import mediapipe as mp
import csv
import os

# === CONFIGURACIÓN ===
letra = input("¿Qué letra estás grabando? (ej. A, B, C...): ").upper()
ruta_base = os.path.dirname(os.path.abspath(__file__))
ruta_datos = os.path.join(ruta_base, "..", "datos")
os.makedirs(ruta_datos, exist_ok=True)
archivo_csv = os.path.join(ruta_datos, f"{letra}.csv")

# === MediaPipe Hands ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# === Captura de cámara ===
cap = cv2.VideoCapture(0)

# === Escribir CSV ===
with open(archivo_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    print(f"[INFO] Grabando datos para la letra '{letra}'. Presiona 'q' para terminar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
                landmarks.append(letra)
                writer.writerow(landmarks)
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f'Recogiendo letra: {letra}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Captura de Datos - Lenguaje de Señas", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Captura finalizada.")
            break

cap.release()
cv2.destroyAllWindows()