import numpy as np


def normalize(vector_):
    vector = np.array(vector_, dtype=np.float32)
    val_min = np.min(vector)
    val_max = np.max(vector)
    if val_max == val_min:
        return np.zeros_like(vector)
    return (vector - val_min) / (val_max - val_min)
