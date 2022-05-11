import numpy as np

def generate_halfmoon(n1, n2, max_angle=3.14):
    alpha = np.linspace(0, max_angle, n1)
    beta = np.linspace(0, max_angle, n2)
    X1 = np.vstack([np.cos(alpha), np.sin(alpha)])\
        + 0.1 * np.random.randn(2,n1)
    X2 = np.vstack([1 - np.cos(beta), 1 - np.sin(beta) - 0.5])\
        + 0.1 * np.random.randn(2,n2)
    y1, y2 = -np.ones(n1), np.ones(n2)
    return X1, y1, X2, y2
