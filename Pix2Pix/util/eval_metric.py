import numpy as np

def fnorm(x):
    return np.sqrt((x**2).sum())

def seismic_cost(y_true, y_pred):
    return fnorm(y_true - y_pred) / (fnorm(y_true) + fnorm(y_pred))

def score(y_true, y_pred):
    return 100*(1 - seismic_cost(y_true, y_pred))

