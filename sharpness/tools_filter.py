import numpy as np


def normalize(vector_):
    """
    Recebe um vetor com as energias de alta frequência 
    retorna um vetor normalizado na escala de 0 a 1.
    """
    # Converte para um array NumPy caso seja uma lista comum
    vector = np.array(vector_, dtype=np.float32)
    
    vaule_min = np.min(vector)
    vaule_max = np.max(vector)
    
    if vaule_max == vaule_min:
        return np.zeros_like(vector)
    
    normalized_vector = (vector - vaule_min) / (vaule_max - vaule_min)
    
    return normalized_vector

