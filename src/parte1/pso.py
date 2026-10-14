"""
Optimización por enjambre de partículas (PSO) implementada desde cero.
Responsable: Persona 3 (P3).

Variante: PSO de mejor global (gbest) con peso de inercia
(Kennedy & Eberhart, 1995; Shi & Eberhart, 1998).

Ecuaciones de actualización, para cada partícula i y cada iteración t:

    v_i(t+1) = w(t) * v_i(t)
               + c1 * r1 ⊙ (p_i - x_i(t))      (componente cognitiva)
               + c2 * r2 ⊙ (g   - x_i(t))      (componente social)
    x_i(t+1) = x_i(t) + v_i(t+1)

donde r1, r2 ~ U(0, 1)^n se muestrean por componente, p_i es la mejor posición
que ha visitado la partícula i y g la mejor posición de todo el enjambre.

El peso de inercia puede ser constante (w_inicial == w_final) o decrecer
linealmente de w_inicial a w_final (Shi & Eberhart, 1998).

Manejo de fronteras: si una componente sale de [lim_inf, lim_sup] se recorta
al límite y su velocidad se anula ("pared absorbente"). Así toda evaluación
de f ocurre dentro del dominio, y el resultado es comparable con los demás
métodos.

Conteo de evaluaciones: se evalúa f una vez por partícula al inicializar y
una vez por partícula en cada iteración, es decir

    evaluaciones de f = N * (T + 1),    evaluaciones de ∇f = 0.

El contador es interno (no depende de variables globales) y se verifica
contra esa fórmula.
"""

import numpy as np


class _ContadorEvaluaciones:
    """Envuelve una función objetivo y cuenta cuántas veces se llama."""

    def __init__(self, funcion):
        self.funcion = funcion
        self.n = 0

    def __call__(self, x):
        self.n += 1
        return float(self.funcion(x))


def pso(
    funcion,
    lim_inf,
    lim_sup,
    n_particulas=50,
    iteraciones=100,
    w_inicial=0.7298,
    w_final=0.7298,
    c1=1.49618,
    c2=1.49618,
    frac_vmax=0.2,
    semilla=56,
    guardar_historia=False,
):
    """
    Minimiza `funcion` sobre la caja [lim_inf, lim_sup] con PSO gbest.

    Parameters
    ----------
    funcion : callable
        Recibe un vector de dimensión n y devuelve un escalar.
    lim_inf, lim_sup : array-like de dimensión n
        Límites de la caja de búsqueda.
    n_particulas : int
        Tamaño del enjambre N.
    iteraciones : int
        Número de iteraciones T.
    w_inicial, w_final : float
        Peso de inercia al inicio y al final (decrece linealmente).
    c1, c2 : float
        Coeficientes cognitivo y social.
    frac_vmax : float
        Velocidad máxima por componente como fracción del ancho del dominio.
    semilla : int
        Semilla del generador aleatorio (np.random.default_rng).
    guardar_historia : bool
        Si es True, guarda las posiciones en cada iteración (para los GIF).

    Returns
    -------
    dict con x_opt, f_opt, evals_f, evals_grad, curva (mejor f tras cada
    iteración), evals_curva (evaluaciones acumuladas en cada punto de la curva)
    y, si se pide, historia_pos e historia_gbest.
    """
    rng = np.random.default_rng(semilla)
    f = _ContadorEvaluaciones(funcion)

    lim_inf = np.asarray(lim_inf, dtype=float)
    lim_sup = np.asarray(lim_sup, dtype=float)
    n = lim_inf.size
    ancho = lim_sup - lim_inf
    v_max = frac_vmax * ancho

    # Inicialización: posiciones uniformes en la caja, velocidades uniformes en [-v_max, v_max]
    x = rng.uniform(lim_inf, lim_sup, size=(n_particulas, n))
    v = rng.uniform(-v_max, v_max, size=(n_particulas, n))

    f_x = np.array([f(xi) for xi in x])
    p_best = x.copy()
    f_p_best = f_x.copy()
    k = int(np.argmin(f_p_best))
    g_best = p_best[k].copy()
    f_g_best = f_p_best[k]

    curva = [f_g_best]
    evals_curva = [f.n]
    historia_pos = [x.copy()] if guardar_historia else None
    historia_gbest = [g_best.copy()] if guardar_historia else None

    for t in range(iteraciones):
        # Peso de inercia (constante o lineal decreciente)
        if iteraciones > 1:
            w = w_inicial - (w_inicial - w_final) * t / (iteraciones - 1)
        else:
            w = w_inicial

        r1 = rng.random((n_particulas, n))
        r2 = rng.random((n_particulas, n))

        v = w * v + c1 * r1 * (p_best - x) + c2 * r2 * (g_best - x)
        v = np.clip(v, -v_max, v_max)
        x = x + v

        # Pared absorbente: recortar posición y anular la velocidad que empujaba hacia afuera
        fuera = (x < lim_inf) | (x > lim_sup)
        x = np.clip(x, lim_inf, lim_sup)
        v[fuera] = 0.0

        f_x = np.array([f(xi) for xi in x])

        mejora = f_x < f_p_best
        p_best[mejora] = x[mejora]
        f_p_best[mejora] = f_x[mejora]

        k = int(np.argmin(f_p_best))
        if f_p_best[k] < f_g_best:
            g_best = p_best[k].copy()
            f_g_best = f_p_best[k]

        curva.append(f_g_best)
        evals_curva.append(f.n)
        if guardar_historia:
            historia_pos.append(x.copy())
            historia_gbest.append(g_best.copy())

    assert f.n == n_particulas * (iteraciones + 1), "Conteo de evaluaciones inconsistente"

    resultado = {
        "x_opt": g_best,
        "f_opt": float(f_g_best),
        "evals_f": f.n,
        "evals_grad": 0,
        "curva": np.asarray(curva),
        "evals_curva": np.asarray(evals_curva),
    }
    if guardar_historia:
        resultado["historia_pos"] = np.asarray(historia_pos)
        resultado["historia_gbest"] = np.asarray(historia_gbest)
    return resultado
