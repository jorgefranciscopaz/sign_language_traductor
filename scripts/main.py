import cv2
import mediapipe as mp
import csv
import os

# Nombre de la letra que vas a grabar (CAMBIA ESTO antes de correr)
letra = 'A'

# Crear directorio si no existe
if not os.path.exists("../datos"):
    os.makedirs("../datos")

archivo_csv = f"datos/{letra}.csv"

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Abrir cámara
cap = cv2.VideoCapture(0)

# Crear archivo CSV
with open(archivo_csv, mode='w', newline='') as f:
    writer = csv.writer(f)

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
                landmarks.append(letra)  # Agregar etiqueta
                writer.writerow(landmarks)

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f'Recogiendo letra: {letra}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Captura de Datos", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

