import numpy as np


contador_f = 0
contador_rosenbrock = 0
contador_schwefel = 0
contador_grad = 0


def rosenbrock(x):
    """Calcula Rosenbrock e incrementa su contador de evaluaciones."""
    global contador_rosenbrock
    contador_rosenbrock += 1
    x = np.asarray(x, dtype=float)
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def schwefel(x):
    """Calcula Schwefel e incrementa su contador de evaluaciones."""
    global contador_schwefel
    contador_schwefel += 1
    x = np.asarray(x, dtype=float)
    n = len(x)
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def grad_rosenbrock(x):
    """Calcula el gradiente analítico de Rosenbrock."""
    global contador_grad
    contador_grad += 1

    x = np.asarray(x, dtype=float)
    n = len(x)
    grad = np.zeros(n)

    grad[0] = -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0])

    for i in range(1, n - 1):
        grad[i] = (
            200.0 * (x[i] - x[i - 1] ** 2)
            - 400.0 * x[i] * (x[i + 1] - x[i] ** 2)
            - 2.0 * (1.0 - x[i])
        )

    grad[-1] = 200.0 * (x[-1] - x[-2] ** 2)
    return grad


def grad_schwefel(x):
    """Calcula el gradiente analítico de Schwefel."""
    global contador_grad
    contador_grad += 1

    x = np.asarray(x, dtype=float)
    abs_x = np.abs(x)
    sqrt_x = np.sqrt(abs_x)
    return -(np.sin(sqrt_x) + 0.5 * sqrt_x * np.cos(sqrt_x))


def pyswarm_wrapper(X, funcion):
    """Adapta una función escalar para recibir una población de PySwarms."""
    return np.array([funcion(ind) for ind in X])
