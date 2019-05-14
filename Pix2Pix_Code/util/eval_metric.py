import numpy as np

def fnorm(x):
    return np.sqrt((x**2).sum())

def score(y_true, y_pred):
    return 100*(1 - 2*fnorm(y_true - y_pred) / (fnorm(y_true) + fnorm(y_pred))) 
