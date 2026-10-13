import numpy as np


def grad_desc(x0, grad_fun, max_iter=50, learning_rate=0.01, tolerancia=0.00001):
    """
    Ejecuta descenso por gradiente y devuelve la trayectoria.

    Parameters
    ----------
    x0 : array-like
        Punto inicial.
    grad_fun : callable
        Gradiente de la función que se desea optimizar.
    max_iter : int
        Máximo número de iteraciones.
    learning_rate : float
        Tasa de aprendizaje.
    tolerancia : float
        Cambio mínimo en la norma de la solución para detener el algoritmo.
    """
    x = [np.asarray(x0, dtype=float)]

    for _ in range(max_iter):
        x_new = x[-1] - learning_rate * grad_fun(x[-1])
        x.append(x_new)

        if np.linalg.norm(x[-1] - x[-2]) < tolerancia:
            break

    return x
