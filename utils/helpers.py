import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Assumes vectors are already normalized to unit length."""
    return float(np.dot(a, b))