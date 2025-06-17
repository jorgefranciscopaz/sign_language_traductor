import numpy as np

def normalizar_landmarks(landmarks):
    # Convertir a np.array y separar en pares (x, y)
    lm = np.array(landmarks).reshape(-1, 2)
    centro = lm[0]  # Punto de referencia: la muñeca
    lm_rel = lm - centro  # Coordenadas relativas
    max_dist = np.max(np.linalg.norm(lm_rel, axis=1))  # Escala por la distancia máxima
    if max_dist > 0:
        lm_norm = lm_rel / max_dist
    else:
        lm_norm = lm_rel
    return lm_norm.flatten()
