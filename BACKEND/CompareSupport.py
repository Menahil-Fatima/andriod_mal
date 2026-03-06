import numpy as np

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def abs_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))
