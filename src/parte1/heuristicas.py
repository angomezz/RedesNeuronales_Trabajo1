import numpy as np
from scipy.optimize import differential_evolution


def Pob_inicial(N, dimension, lim_inf, lim_sup, semilla):
    np.random.seed(semilla)
    return np.random.uniform(lim_inf, lim_sup, size=(N, dimension))


def eval_pob(Poblacion, f_obj, lim_inf, lim_sup):
    n = Poblacion.shape[0]
    f_eval = np.zeros(n)

    for i in range(n):
        if np.all(Poblacion[i, :] >= lim_inf) and np.all(Poblacion[i, :] <= lim_sup):
            f_eval[i] = f_obj(Poblacion[i, :])
        else:
            f_eval[i] = np.inf

    return f_eval


def rank_pob(Poblacion, f_eval_pob):
    orden = np.argsort(f_eval_pob)
    return Poblacion[orden]


def elitismo(Poblacion, frac_e):
    n, _ = Poblacion.shape
    n_elite = int(np.round(n * frac_e))
    return Poblacion[0:n_elite, :]


def mutacion(Poblacion, frac_m, lim_inf, lim_sup):
    n, m = Poblacion.shape
    Poblacion_mutada = Poblacion.copy()

    n_mutantes = int(n * frac_m)
    indices_mutantes = np.random.choice(n, n_mutantes, replace=False)
    posicion_coordenada = np.random.choice(m, n_mutantes, replace=True)

    desv_est = Poblacion.std(axis=0)
    desviaciones = desv_est[posicion_coordenada]
    desviacion_a_sumar = np.random.standard_normal(n_mutantes) * desviaciones

    nuevos_valores = (
        Poblacion[indices_mutantes, posicion_coordenada] + desviacion_a_sumar
    )

    limites_inferiores = lim_inf[posicion_coordenada]
    limites_superiores = lim_sup[posicion_coordenada]

    Poblacion_mutada[indices_mutantes, posicion_coordenada] = np.clip(
        nuevos_valores,
        limites_inferiores,
        limites_superiores,
    )

    return Poblacion_mutada[indices_mutantes, :]


def cruzamiento(Poblacion, frac_c):
    n, m = Poblacion.shape

    n_cruzamiento = int(n * frac_c) + (int(n * frac_c) % 2)
    n_pares = n_cruzamiento // 2
    indices_padres = np.random.choice(n, n_cruzamiento, replace=False)

    Padres = Poblacion[indices_padres[0:n_pares], :]
    Madres = Poblacion[indices_padres[n_pares:n_cruzamiento], :]
    posicion_coordenada = np.random.choice(m, n_pares, replace=True)
    Hijos = np.zeros([n_cruzamiento, m])

    for i in range(n_pares):
        punto = posicion_coordenada[i]
        Hijos[i, :] = np.concatenate((Padres[i, 0:punto], Madres[i, punto:m]))
        Hijos[n_pares + i, :] = np.concatenate(
            (Madres[i, 0:punto], Padres[i, punto:m])
        )

    return Hijos


def evol_pob(Poblacion, lim_inf, lim_sup, funcion, frac_e, frac_m, frac_c):
    n, _ = Poblacion.shape
    f_aptitud = eval_pob(Poblacion, funcion, lim_inf, lim_sup)
    Poblacion_ordenada = rank_pob(Poblacion, f_aptitud)
    Poblacion_elite = elitismo(Poblacion_ordenada, frac_e)
    Poblacion_mutada = mutacion(Poblacion, frac_m, lim_inf, lim_sup)
    Poblacion_cruzamiento = cruzamiento(Poblacion, frac_c)
    next_Poblacion = np.concatenate(
        (Poblacion_elite, Poblacion_mutada, Poblacion_cruzamiento),
        axis=0,
    )
    return next_Poblacion[0:n, :]


def algort_evol(funcion, n_dim, lim_inf, lim_sup, semilla=56, n_iter=100):
    nueva_Pob = Pob_inicial(
        50,
        dimension=n_dim,
        lim_inf=lim_inf,
        lim_sup=lim_sup,
        semilla=semilla,
    )

    for _ in range(n_iter):
        nueva_Pob = evol_pob(
            nueva_Pob,
            lim_inf,
            lim_sup,
            funcion,
            frac_e=0.15,
            frac_m=0.1,
            frac_c=0.75,
        )

    f_final = eval_pob(nueva_Pob, funcion, lim_inf, lim_sup)
    pob_ordenada = rank_pob(nueva_Pob, f_final)

    return np.min(f_final), pob_ordenada[0]


def run_differential_evolution(
    funcion,
    bounds,
    maxiter=200,
    popsize=15,
    seed=None,
    tolerance=1e-6,
):
    """Ejecuta DE con la misma función y límites usados en main.py."""
    return differential_evolution(
        func=funcion,
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        tol=tolerance,
    )
