import numpy as np

def lerp(y0, yf, x):
    x = np.clip(x, 0, 1)
    return y0 + (yf - y0) * x

def cubicBezier(y0:np.ndarray, yf:np.ndarray, x:float):
    yDiff = yf - y0
    bezier = x * x * x + 3 * (x * x * (1.0 - x))
    return y0 + yDiff*bezier

def cubicBezierFirstDerivative(y0:np.ndarray, yf:np.ndarray, x:float):
    yDiff = yf - y0
    bezier = 6.0 * x * (1.0-x)
    if x == 1:
        bezier = 0
        
    return yDiff * bezier
