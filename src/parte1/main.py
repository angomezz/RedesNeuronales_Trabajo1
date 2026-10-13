"""
===============================================================================
MÓDULO DE OPTIMIZACIÓN Y COMPARACIÓN DE ALGORITMOS
===============================================================================
Este archivo contiene la implementación de varios algoritmos de optimización
(Descenso por Gradiente, PSO, Algoritmo Evolutivo y Evolución Diferencial)
junto con la evaluación de su desempeño sobre las funciones de prueba
Rosenbrock y Schwefel.
===============================================================================
"""

# =============================================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm
from scipy.optimize import differential_evolution
from pyswarms.utils.plotters.formatters import Designer, Mesher
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from IPython.display import Image

np.random.seed(56)


# =============================================================================
# 2. CONTADORES GLOBALES DE EVALUACIÓN
# =============================================================================
contador_f = 0
contador_rosenbrock = 0
contador_schwefel = 0
contador_grad = 0


# =============================================================================
# 3. FUNCIONES OBJETIVO Y GRADIENTES
# =============================================================================
def rosenbrock(x):
    """Calcula la función de Rosenbrock e incrementa su contador de evaluaciones."""
    global contador_rosenbrock
    contador_rosenbrock += 1
    x = np.asarray(x, dtype=float)
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2)


def schwefel(x):
    """Calcula la función de Schwefel e incrementa su contador de evaluaciones."""
    global contador_schwefel
    contador_schwefel += 1
    x = np.asarray(x, dtype=float)
    n = len(x)
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def grad_rosenbrock(x):
    """Calcula el gradiente analítico de la función de Rosenbrock."""
    global contador_grad
    contador_grad += 1

    x = np.asarray(x, dtype=float)
    n = len(x)
    grad = np.zeros(n)

    # Derivada para el primer elemento
    grad[0] = -400.0 * x[0] * (x[1] - x[0]**2) - 2.0 * (1.0 - x[0])

    # Derivadas para los elementos intermedios (si n > 2)
    for i in range(1, n - 1):
        grad[i] = 200.0 * (x[i] - x[i-1]**2) - 400.0 * x[i] * (x[i+1] - x[i]**2) - 2.0 * (1.0 - x[i])

    # Derivada para el último elemento
    grad[-1] = 200.0 * (x[-1] - x[-2]**2)

    return grad


def grad_schwefel(x):
    """Calcula el gradiente analítico de la función de Schwefel."""
    global contador_grad
    contador_grad += 1

    x = np.asarray(x, dtype=float)
    abs_x = np.abs(x)
    sqrt_x = np.sqrt(abs_x)

    # Derivada parcial vectorial para cada dimensión i
    grad = -(np.sin(sqrt_x) + 0.5 * sqrt_x * np.cos(sqrt_x))

    return grad


def pyswarm_wrapper(X, funcion):
    """Adaptador para evaluar funciones vectorizadas en PySwarms."""
    return np.array([funcion(ind) for ind in X])


# =============================================================================
# 4. ALGORITMOS DE OPTIMIZACIÓN
# =============================================================================

# --- Descenso por Gradiente ---
def grad_desc(x0, grad_fun, max_iter=50, learning_rate=0.01, tolerancia=0.00001):
    """
    x0: punto inicial
    grad_fun: gradiente de la función que se desea optimizar
    max_iter: máximo número de iteraciones
    learning_rate: tasa de aprendizaje
    tolerancia: mínimo cambio en la norma de la solución para detener el algoritmo
    """
    x = []
    x.append(x0)

    for t in range(max_iter):
        x_new = x[t] - learning_rate * grad_fun(x[t])
        x.append(x_new)
        cambio = np.linalg.norm(x[t+1] - x[t])
        if cambio < tolerancia:
            break
    return x


# --- Algoritmo Evolutivo (EA) ---
def Pob_inicial(N, dimension, lim_inf, lim_sup, semilla):
    np.random.seed(semilla)
    Pob = np.random.uniform(lim_inf, lim_sup, size=(N, dimension))
    return Pob


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
    (n, m) = Poblacion.shape
    n_elite = int(np.round(n * frac_e))
    return Poblacion[0:n_elite, :]


def mutacion(Poblacion, frac_m, lim_inf, lim_sup):
    (n, m) = Poblacion.shape
    Poblacion_mutada = Poblacion.copy()

    n_mutantes = int(n * frac_m)
    indices_mutantes = np.random.choice(n, n_mutantes, replace=False)
    posicion_coordenada = np.random.choice(m, n_mutantes, replace=True)

    desv_est = Poblacion.std(axis=0)
    desviaciones = desv_est[posicion_coordenada]
    desviacion_a_sumar = np.random.standard_normal(n_mutantes) * desviaciones

    nuevos_valores = Poblacion[indices_mutantes, posicion_coordenada] + desviacion_a_sumar

    limites_inferiores = lim_inf[posicion_coordenada]
    limites_superiores = lim_sup[posicion_coordenada]

    Poblacion_mutada[indices_mutantes, posicion_coordenada] = np.clip(
        nuevos_valores,
        limites_inferiores,
        limites_superiores
    )

    return Poblacion_mutada[indices_mutantes, :]


def cruzamiento(Poblacion, frac_c):
    (n, m) = Poblacion.shape

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
        Hijos[n_pares + i, :] = np.concatenate((Madres[i, 0:punto], Padres[i, punto:m]))

    return Hijos


def evol_pob(Poblacion, lim_inf, lim_sup, funcion, frac_e, frac_m, frac_c):
    (n, m) = Poblacion.shape
    f_aptitud = eval_pob(Poblacion, funcion, lim_inf, lim_sup)
    Poblacion_ordenada = rank_pob(Poblacion, f_aptitud)
    Poblacion_elite = elitismo(Poblacion_ordenada, frac_e)
    Poblacion_mutada = mutacion(Poblacion, frac_m, lim_inf, lim_sup)
    Poblacion_cruzamiento = cruzamiento(Poblacion, frac_c)
    next_Poblacion = np.concatenate((Poblacion_elite, Poblacion_mutada, Poblacion_cruzamiento), axis=0)
    return next_Poblacion[0:n, :]


def algort_evol(funcion, n_dim, lim_inf, lim_sup, semilla=56, n_iter=100):
    nueva_Pob = Pob_inicial(50, dimension=n_dim, lim_inf=lim_inf, lim_sup=lim_sup, semilla=semilla)

    for i in range(n_iter):
        nueva_Pob = evol_pob(nueva_Pob, lim_inf, lim_sup, funcion, frac_e=0.15, frac_m=0.1, frac_c=0.75)

    f_final = eval_pob(nueva_Pob, funcion, lim_inf, lim_sup)
    pob_ordenada = rank_pob(nueva_Pob, f_final)
    mejor_solucion = pob_ordenada[0]
    mejor_costo = np.min(f_final)

    return mejor_costo, mejor_solucion


# =============================================================================
# 5. PRUEBAS INICIALES
# =============================================================================
contador_f = 0
contador_grad = 0

x0 = np.array([-1.2, 1.0, 0.5])
trayectoria = grad_desc(x0, grad_fun=grad_rosenbrock, max_iter=20000, learning_rate=0.001)

print(f"Total de evaluaciones de f: {contador_f}")
print(f"Total de evaluaciones de gradiente (grad_f): {contador_grad}")

trayectoria = np.array(trayectoria)
dim = trayectoria.shape[1]
etiquetas = [f'$x_{{{i+1}}}$' for i in range(dim)]

t = np.arange(len(trayectoria))
plt.plot(t, trayectoria, label=etiquetas)
plt.xlabel('Iteraciones')
plt.ylabel('Valores de los parámetros')
plt.title('Comportamiento de la optimización')
plt.legend()
plt.grid()
plt.show()

f_obj = [rosenbrock(punto) for punto in trayectoria]

plt.plot(t, f_obj, label='Función objetivo')
plt.xlabel('Iteraciones')
plt.ylabel('Valor de la función objetivo')
plt.title('Comportamiento de la optimización')
plt.legend()
plt.grid()
plt.show()

# Test inicial PSO Rosenbrock
contador_rosenbrock = 0
xmin, xmax, ymin, ymax = -2, 2, -1, 3
options1 = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
bounds = (np.array([xmin, ymin]), np.array([xmax, ymax]))
optimizer1 = ps.single.GlobalBestPSO(n_particles=50, dimensions=2, options=options1, bounds=bounds)
cost1, pos1 = optimizer1.optimize(lambda X: pyswarm_wrapper(X, rosenbrock), iters=100, verbose=False)
print(f"Evaluaciones de Rosenbrock: {contador_rosenbrock}")

# Test inicial PSO Schwefel
contador_schwefel = 0
xmin, xmax, ymin, ymax = -500, 500, -500, 500
options1 = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
bounds = (np.array([xmin, ymin]), np.array([xmax, ymax]))
optimizer1 = ps.single.GlobalBestPSO(n_particles=50, dimensions=2, options=options1, bounds=bounds)
cost1, pos1 = optimizer1.optimize(lambda X: pyswarm_wrapper(X, schwefel), iters=100, verbose=False)
print(f"Evaluaciones de Schwefel: {contador_schwefel}")


# =============================================================================
# 6. EVALUACIONES EXPERIMENTALES (30 CORRIDAS POR ALGORITMO)
# =============================================================================

# --- ROSENBROCK 2D: GD ---
num_corridas = 30
dim = 2
max_iter = 2000
learning_rate = 0.00001

resultados_f_evals = []
resultados_grad_evals = []
valores_optimos = []

for i in range(num_corridas):
    contador_rosenbrock = 0
    contador_grad = 0
    np.random.seed(56 + i)
    x0 = np.random.uniform(-2, 2, dim)
    trayectoria = grad_desc(x0, grad_fun=grad_rosenbrock, max_iter=max_iter, learning_rate=learning_rate)
    optimo_encontrado = trayectoria[-1]
    valor_final = rosenbrock(optimo_encontrado)

    resultados_f_evals.append(contador_rosenbrock)
    resultados_grad_evals.append(contador_grad)
    valores_optimos.append(valor_final)

print("--- Estadísticas Descenso por Gradiente (30 corridas) ---")
print(f"Media valor óptimo: {np.mean(valores_optimos):.6f} (± {np.std(valores_optimos):.6f})")
print(f"Mejor valor: {np.min(valores_optimos):.6f} | Peor valor: {np.max(valores_optimos):.6f}")
print(f"Promedio evaluaciones de f: {np.mean(resultados_f_evals):.2f}")
print(f"Promedio evaluaciones de gradiente: {np.mean(resultados_grad_evals):.2f}")

# --- ROSENBROCK 2D: PSO ---
num_corridas = 30
dim = 2
n_particles = 50
iters = 100

valores_optimos_pso = []
evals_pso = []
bounds = (np.array([-2]*dim), np.array([2]*dim))
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

for i in range(num_corridas):
    contador_rosenbrock = 0
    np.random.seed(56 + i)
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dim, options=options, bounds=bounds)
    best_cost, best_pos = optimizer.optimize(lambda X: pyswarm_wrapper(X, rosenbrock), iters=iters, verbose=False)
    valores_optimos_pso.append(best_cost)
    evals_pso.append(contador_rosenbrock)

print("--- Estadísticas PSO (30 corridas) ---")
print(f"Media valor óptimo: {np.mean(valores_optimos_pso):.6f} (± {np.std(valores_optimos_pso):.6f})")
print(f"Mejor valor: {np.min(valores_optimos_pso):.6f} | Peor valor: {np.max(valores_optimos_pso):.6f}")
print(f"Evaluaciones de f por corrida: {evals_pso[0]}")

# --- ROSENBROCK 2D: EA ---
num_corridas = 30
n_dim = 2
lim_inf = -2.0 * np.ones(n_dim)
lim_sup = 2.0 * np.ones(n_dim)

valores_optimos_ae = []
evals_ae = []
mejor_valor_ae = float("inf")
mejor_pos_ae = None
umbral_exito = 1e-2

for i in range(num_corridas):
    contador_rosenbrock = 0
    best_cost, best_pos = algort_evol(
        rosenbrock,
        n_dim=n_dim,
        lim_inf=lim_inf,
        lim_sup=lim_sup,
        semilla=56 + i,
        n_iter=100,
    )
    if best_cost < mejor_valor_ae:
        mejor_valor_ae = best_cost
        mejor_pos_ae = best_pos

    valores_optimos_ae.append(best_cost)
    evals_ae.append(contador_rosenbrock)

exitos = sum(1 for v in valores_optimos_ae if v <= umbral_exito)
tasa_exito = (exitos / num_corridas) * 100

print(f"--- Estadísticas Algoritmo Evolutivo ({n_dim}D) ---")
print(f"Media valor óptimo: {np.mean(valores_optimos_ae):.6f} (± {np.std(valores_optimos_ae):.6f})")
print(f"Mejor valor: {np.min(valores_optimos_ae):.6f} | Peor valor: {np.max(valores_optimos_ae):.6f}")
print(f"Coordenadas óptimas (mejor posición): {mejor_pos_ae}")
print(f"Promedio evaluaciones de f por corrida: {np.mean(evals_ae):.2f}")
print(f"Tasa de éxito (f <= {umbral_exito}): {tasa_exito:.1f}%")

# --- ROSENBROCK 2D: DE ---
num_corridas = 30
dim = 2
bounds = [(-2.0, 2.0)] * dim

valores_optimos_de = []
evals_de = []
mejor_valor_de = float("inf")
mejor_pos_de = None

for i in range(num_corridas):
    contador_rosenbrock = 0
    res = differential_evolution(
        func=rosenbrock,
        bounds=bounds,
        maxiter=200,
        popsize=15,
        seed=56 + i,
        tol=1e-6,
    )
    best_cost = res.fun
    best_pos = res.x

    if best_cost < mejor_valor_de:
        mejor_valor_de = best_cost
        mejor_pos_de = best_pos

    valores_optimos_de.append(best_cost)
    evals_de.append(contador_rosenbrock)

print(f"--- Estadísticas DE Rosenbrock ({dim}D - 30 corridas) ---")
print(f"Media valor óptimo: {np.mean(valores_optimos_de):.6f} (± {np.std(valores_optimos_de):.6f})")
print(f"Mejor valor: {np.min(valores_optimos_de):.6f} | Peor valor: {np.max(valores_optimos_de):.6f}")
print(f"Coordenadas óptimas (posición): {mejor_pos_de}")
print(f"Promedio evaluaciones de f por corrida: {np.mean(evals_de):.2f}")

# --- ROSENBROCK 3D: GD ---
num_corridas = 30
dim = 3
max_iter = 2000
learning_rate = 0.00001

resultados_f_evals = []
resultados_grad_evals = []
valores_optimos = []
mejor_valor_gd = float('inf')
mejor_pos_gd = None

for i in range(num_corridas):
    contador_rosenbrock = 0
    contador_grad = 0

    np.random.seed(56 + i)
    x0 = np.random.uniform(-2, 2, dim)
    trayectoria = grad_desc(x0, grad_fun=grad_rosenbrock, max_iter=max_iter, learning_rate=learning_rate)

    optimo_encontrado = trayectoria[-1]
    valor_final = rosenbrock(optimo_encontrado)

    if valor_final < mejor_valor_gd:
        mejor_valor_gd = valor_final
        mejor_pos_gd = optimo_encontrado

    resultados_f_evals.append(contador_rosenbrock)
    resultados_grad_evals.append(contador_grad)
    valores_optimos.append(valor_final)

print("--- Estadísticas Descenso por Gradiente ---")
print(f"Mejor valor encontrado: {mejor_valor_gd:.6f}")
print(f"Coordenadas óptimas (posición): {mejor_pos_gd}")
print(f"Promedio evaluaciones de f: {np.mean(resultados_f_evals):.2f}")

# --- ROSENBROCK 3D: PSO ---
num_corridas = 30
dim = 3
n_particles = 50
iters = 100

valores_optimos_pso = []
evals_pso = []
mejor_valor_pso = float('inf')
mejor_pos_pso = None

bounds = (np.array([-2]*dim), np.array([2]*dim))
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

for i in range(num_corridas):
    contador_rosenbrock = 0
    np.random.seed(56 + i)
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dim, options=options, bounds=bounds)
    best_cost, best_pos = optimizer.optimize(lambda X: pyswarm_wrapper(X, rosenbrock), iters=iters, verbose=False)

    if best_cost < mejor_valor_pso:
        mejor_valor_pso = best_cost
        mejor_pos_pso = best_pos

    valores_optimos_pso.append(best_cost)
    evals_pso.append(contador_rosenbrock)

print("--- Estadísticas PSO ---")
print(f"Mejor valor encontrado: {mejor_valor_pso:.6f}")
print(f"Coordenadas óptimas (posición): {mejor_pos_pso}")
print(f"Evaluaciones de f por corrida: {evals_pso[0]}")

# --- ROSENBROCK 3D: EA ---
num_corridas = 30
n_dim = 3
lim_inf = -2.0 * np.ones(n_dim)
lim_sup = 2.0 * np.ones(n_dim)

valores_optimos_ae = []
evals_ae = []
mejor_valor_ae = float("inf")
mejor_pos_ae = None
umbral_exito = 1e-2

for i in range(num_corridas):
    contador_rosenbrock = 0
    best_cost, best_pos = algort_evol(
        rosenbrock,
        n_dim=n_dim,
        lim_inf=lim_inf,
        lim_sup=lim_sup,
        semilla=56 + i,
        n_iter=100,
    )

    if best_cost < mejor_valor_ae:
        mejor_valor_ae = best_cost
        mejor_pos_ae = best_pos

    valores_optimos_ae.append(best_cost)
    evals_ae.append(contador_rosenbrock)

exitos = sum(1 for v in valores_optimos_ae if v <= umbral_exito)
tasa_exito = (exitos / num_corridas) * 100

print(f"--- Estadísticas Algoritmo Evolutivo ({n_dim}D) ---")
print(f"Media valor óptimo: {np.mean(valores_optimos_ae):.6f} (± {np.std(valores_optimos_ae):.6f})")
print(f"Mejor valor: {np.min(valores_optimos_ae):.6f} | Peor valor: {np.max(valores_optimos_ae):.6f}")
print(f"Coordenadas óptimas (mejor posición): {mejor_pos_ae}")
print(f"Promedio evaluaciones de f por corrida: {np.mean(evals_ae):.2f}")
print(f"Tasa de éxito (f <= {umbral_exito}): {tasa_exito:.1f}%")

# --- ROSENBROCK 3D: DE ---
num_corridas = 30
dim = 3
bounds = [(-2.0, 2.0)] * dim

valores_optimos_de = []
evals_de = []
mejor_valor_de = float("inf")
mejor_pos_de = None

for i in range(num_corridas):
    contador_rosenbrock = 0
    res = differential_evolution(
        func=rosenbrock,
        bounds=bounds,
        maxiter=200,
        popsize=15,
        seed=56 + i,
        tol=1e-6,
    )

    best_cost = res.fun
    best_pos = res.x

    if best_cost < mejor_valor_de:
        mejor_valor_de = best_cost
        mejor_pos_de = best_pos

    valores_optimos_de.append(best_cost)
    evals_de.append(contador_rosenbrock)

print(f"--- Estadísticas DE Rosenbrock ({dim}D - 30 corridas) ---")
print(f"Media valor óptimo: {np.mean(valores_optimos_de):.6f} (± {np.std(valores_optimos_de):.6f})")
print(f"Mejor valor: {np.min(valores_optimos_de):.6f} | Peor valor: {np.max(valores_optimos_de):.6f}")
print(f"Coordenadas óptimas (posición): {mejor_pos_de}")
print(f"Promedio evaluaciones de f por corrida: {np.mean(evals_de):.2f}")

# --- SCHWEFEL 2D: GD ---
num_corridas = 30
dim = 2
max_iter = 2000
learning_rate = 0.1

resultados_f_evals = []
resultados_grad_evals = []
valores_optimos = []

for i in range(num_corridas):
    contador_schwefel = 0
    contador_grad = 0

    np.random.seed(56 + i)
    x0 = np.random.uniform(-500, 500, dim)
    trayectoria = grad_desc(
        x0,
        grad_fun=grad_schwefel,
        max_iter=max_iter,
        learning_rate=learning_rate,
    )

    optimo_encontrado = trayectoria[-1]
    valor_final = schwefel(optimo_encontrado)

    resultados_f_evals.append(contador_schwefel)
    resultados_grad_evals.append(contador_grad)
    valores_optimos.append(valor_final)

print("--- Estadísticas Descenso por Gradiente Schwefel (30 corridas) ---")
print(f"Media valor óptimo: {np.mean(valores_optimos):.6f} (± {np.std(valores_optimos):.6f})")
print(f"Mejor valor: {np.min(valores_optimos):.6f} | Peor valor: {np.max(valores_optimos):.6f}")
print(f"Promedio evaluaciones de f: {np.mean(resultados_f_evals):.2f}")
print(f"Promedio evaluaciones de gradiente: {np.mean(resultados_grad_evals):.2f}")

# --- SCHWEFEL 2D: PSO ---
num_corridas = 30
dim = 2
n_particles = 50
iters = 100

valores_optimos_pso = []
evals_pso = []

bounds = (np.array([-500] * dim), np.array([500] * dim))
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

for i in range(num_corridas):
    contador_schwefel = 0

    np.random.seed(56 + i)
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles, dimensions=dim, options=options, bounds=bounds
    )

    best_cost, best_pos = optimizer.optimize(
        lambda X: pyswarm_wrapper(X, schwefel), iters=iters, verbose=False
    )

    valores_optimos_pso.append(best_cost)
    evals_pso.append(contador_schwefel)

print("--- Estadísticas PSO Schwefel (30 corridas) ---")
print(f"Media valor óptimo: {np.mean(valores_optimos_pso):.6f} (± {np.std(valores_optimos_pso):.6f})")
print(f"Mejor valor: {np.min(valores_optimos_pso):.6f} | Peor valor: {np.max(valores_optimos_pso):.6f}")
print(f"Evaluaciones de f por corrida: {evals_pso[0]}")

# --- SCHWEFEL 2D: EA ---
num_corridas = 30
n_dim = 2
lim_inf = -500.0 * np.ones(n_dim)
lim_sup = 500.0 * np.ones(n_dim)

valores_optimos_ae = []
evals_ae = []
mejor_valor_ae = float("inf")
mejor_pos_ae = None
umbral_exito = 1e-2

for i in range(num_corridas):
    contador_schwefel = 0

    best_cost, best_pos = algort_evol(
        schwefel,
        n_dim=n_dim,
        lim_inf=lim_inf,
        lim_sup=lim_sup,
        semilla=56 + i,
        n_iter=100,
    )

    if best_cost < mejor_valor_ae:
        mejor_valor_ae = best_cost
        mejor_pos_ae = best_pos

    valores_optimos_ae.append(best_cost)
    evals_ae.append(contador_schwefel)

exitos = sum(1 for v in valores_optimos_ae if v <= umbral_exito)
tasa_exito = (exitos / num_corridas) * 100

print(f"--- Estadísticas Algoritmo Evolutivo ({n_dim}D) ---")
print(f"Media valor óptimo: {np.mean(valores_optimos_ae):.6f} (± {np.std(valores_optimos_ae):.6f})")
print(f"Mejor valor: {np.min(valores_optimos_ae):.6f} | Peor valor: {np.max(valores_optimos_ae):.6f}")
print(f"Coordenadas óptimas (mejor posición): {mejor_pos_ae}")
print(f"Promedio evaluaciones de f por corrida: {np.mean(evals_ae):.2f}")
print(f"Tasa de éxito (f <= {umbral_exito}): {tasa_exito:.1f}%")

# --- SCHWEFEL 2D: DE ---
num_corridas = 30
dim = 2
bounds = [(-500.0, 500.0)] * dim

valores_optimos_de = []
evals_de = []
mejor_valor_de = float("inf")
mejor_pos_de = None

for i in range(num_corridas):
    contador_schwefel = 0

    res = differential_evolution(
        func=schwefel,
        bounds=bounds,
        maxiter=200,
        popsize=15,
        seed=56 + i,
        tol=1e-6,
    )

    best_cost = res.fun
    best_pos = res.x

    if best_cost < mejor_valor_de:
        mejor_valor_de = best_cost
        mejor_pos_de = best_pos

    valores_optimos_de.append(best_cost)
    evals_de.append(contador_schwefel)

print(f"--- Estadísticas DE Schwefel ({dim}D - 30 corridas) ---")
print(f"Media valor óptimo: {np.mean(valores_optimos_de):.6f} (± {np.std(valores_optimos_de):.6f})")
print(f"Mejor valor: {np.min(valores_optimos_de):.6f} | Peor valor: {np.max(valores_optimos_de):.6f}")
print(f"Coordenadas óptimas (posición): {mejor_pos_de}")
print(f"Promedio evaluaciones de f por corrida: {np.mean(evals_de):.2f}")

# --- SCHWEFEL 3D: GD ---
num_corridas = 30
dim = 3
max_iter = 2000
learning_rate = 0.1

resultados_f_evals = []
resultados_grad_evals = []
valores_optimos = []

for i in range(num_corridas):
    contador_schwefel = 0
    contador_grad = 0

    np.random.seed(56 + i)
    x0 = np.random.uniform(-500, 500, dim)

    trayectoria = grad_desc(
        x0,
        grad_fun=grad_schwefel,
        max_iter=max_iter,
        learning_rate=learning_rate,
    )

    optimo_encontrado = trayectoria[-1]
    valor_final = schwefel(optimo_encontrado)

    resultados_f_evals.append(contador_schwefel)
    resultados_grad_evals.append(contador_grad)
    valores_optimos.append(valor_final)

print("--- Estadísticas Descenso por Gradiente Schwefel (30 corridas) ---")
print(f"Media valor óptimo: {np.mean(valores_optimos):.6f} (± {np.std(valores_optimos):.6f})")
print(f"Mejor valor: {np.min(valores_optimos):.6f} | Peor valor: {np.max(valores_optimos):.6f}")
print(f"Promedio evaluaciones de f: {np.mean(resultados_f_evals):.2f}")
print(f"Promedio evaluaciones de gradiente: {np.mean(resultados_grad_evals):.2f}")

# --- SCHWEFEL 3D: PSO ---
num_corridas = 30
dim = 3
n_particles = 50
iters = 100

valores_optimos_pso = []
evals_pso = []

bounds = (np.array([-500] * dim), np.array([500] * dim))
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

for i in range(num_corridas):
    contador_schwefel = 0

    np.random.seed(56 + i)
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles, dimensions=dim, options=options, bounds=bounds
    )

    best_cost, best_pos = optimizer.optimize(
        lambda X: pyswarm_wrapper(X, schwefel), iters=iters, verbose=False
    )

    valores_optimos_pso.append(best_cost)
    evals_pso.append(contador_schwefel)

print("--- Estadísticas PSO Schwefel (30 corridas) ---")
print(f"Media valor óptimo: {np.mean(valores_optimos_pso):.6f} (± {np.std(valores_optimos_pso):.6f})")
print(f"Mejor valor: {np.min(valores_optimos_pso):.6f} | Peor valor: {np.max(valores_optimos_pso):.6f}")
print(f"Evaluaciones de f por corrida: {evals_pso[0]}")

# --- SCHWEFEL 3D: EA ---
num_corridas = 30
n_dim = 3
lim_inf = -500.0 * np.ones(n_dim)
lim_sup = 500.0 * np.ones(n_dim)

valores_optimos_ae = []
evals_ae = []
mejor_valor_ae = float("inf")
mejor_pos_ae = None
umbral_exito = 1e-2

for i in range(num_corridas):
    contador_schwefel = 0

    best_cost, best_pos = algort_evol(
        schwefel,
        n_dim=n_dim,
        lim_inf=lim_inf,
        lim_sup=lim_sup,
        semilla=56 + i,
        n_iter=100,
    )

    if best_cost < mejor_valor_ae:
        mejor_valor_ae = best_cost
        mejor_pos_ae = best_pos

    valores_optimos_ae.append(best_cost)
    evals_ae.append(contador_schwefel)

exitos = sum(1 for v in valores_optimos_ae if v <= umbral_exito)
tasa_exito = (exitos / num_corridas) * 100

print(f"--- Estadísticas Algoritmo Evolutivo ({n_dim}D) ---")
print(f"Media valor óptimo: {np.mean(valores_optimos_ae):.6f} (± {np.std(valores_optimos_ae):.6f})")
print(f"Mejor valor: {np.min(valores_optimos_ae):.6f} | Peor valor: {np.max(valores_optimos_ae):.6f}")
print(f"Coordenadas óptimas (mejor posición): {mejor_pos_ae}")
print(f"Promedio evaluaciones de f por corrida: {np.mean(evals_ae):.2f}")
print(f"Tasa de éxito (f <= {umbral_exito}): {tasa_exito:.1f}%")

# --- SCHWEFEL 3D: DE ---
num_corridas = 30
dim = 3
bounds = [(-500.0, 500.0)] * dim

valores_optimos_de = []
evals_de = []
mejor_valor_de = float("inf")
mejor_pos_de = None

for i in range(num_corridas):
    contador_schwefel = 0

    res = differential_evolution(
        func=schwefel,
        bounds=bounds,
        maxiter=200,
        popsize=15,
        seed=56 + i,
        tol=1e-6,
    )

    best_cost = res.fun
    best_pos = res.x

    if best_cost < mejor_valor_de:
        mejor_valor_de = best_cost
        mejor_pos_de = best_pos

    valores_optimos_de.append(best_cost)
    evals_de.append(contador_schwefel)

print(f"--- Estadísticas DE Schwefel ({dim}D - 30 corridas) ---")
print(f"Media valor óptimo: {np.mean(valores_optimos_de):.6f} (± {np.std(valores_optimos_de):.6f})")
print(f"Mejor valor: {np.min(valores_optimos_de):.6f} | Peor valor: {np.max(valores_optimos_de):.6f}")
print(f"Coordenadas óptimas (posición): {mejor_pos_de}")
print(f"Promedio evaluaciones de f por corrida: {np.mean(evals_de):.2f}")


# =============================================================================
# 7. ANIMACIONES Y VISUALIZACIONES
# =============================================================================

# ---------- Funciones aux. para gráficos 2D ----------
def rosen_xy(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2


def schwefel_xy(x, y):
    return 418.9829 * 2 - (x * np.sin(np.sqrt(np.abs(x))) + y * np.sin(np.sqrt(np.abs(y))))


def animar_gd(trayectoria, f_xy, dominio, minimo, nombre_gif,
              titulo='GD', paso_frames=1, max_frames=150, log=True):
    """Genera y guarda una animación GIF de la trayectoria de Descenso por Gradiente."""
    tray = np.asarray(trayectoria)
    xmin, xmax, ymin, ymax = dominio

    idx = np.unique(np.linspace(0, len(tray) - 1, min(max_frames, len(tray))).astype(int))
    tray_s = tray[idx]

    X, Y = np.meshgrid(np.linspace(xmin, xmax, 300), np.linspace(ymin, ymax, 300))
    Z = f_xy(X, Y)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if log:
        niveles = np.logspace(-3, np.log10(Z.max()), 30)
        cs = ax.contourf(X, Y, Z + 1e-3, levels=niveles, norm=LogNorm(), cmap='coolwarm', alpha=0.8)
    else:
        cs = ax.contourf(X, Y, Z, levels=30, cmap='coolwarm', alpha=0.8)

    fig.colorbar(cs, ax=ax, label='f(x, y)')
    ax.plot(*minimo, 'r*', ms=14, label='Mínimo global')
    ax.plot(*tray_s[0], 'gs', ms=8, label='Inicio')

    linea, = ax.plot([], [], '-', color='black', lw=1.2)
    punto, = ax.plot([], [], 'o', color='black', ms=6, label='GD')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper right')

    def update(k):
        linea.set_data(tray_s[:k+1, 0], tray_s[:k+1, 1])
        punto.set_data([tray_s[k, 0]], [tray_s[k, 1]])
        ax.set_title(f'{titulo} – iteración {idx[k]} f = {f_xy(tray_s[k,0], tray_s[k,1]):.3g}')
        return linea, punto

    anim = FuncAnimation(fig, update, frames=len(tray_s))
    anim.save(nombre_gif, writer=PillowWriter(fps=15), dpi=80)
    plt.close(fig)
    return nombre_gif


# ---------- Animación Descenso por Gradiente (Rosenbrock) ----------
rng = np.random.default_rng(56)
x0 = rng.uniform(-2, 2, 2)

trayectoria = grad_desc(x0, grad_fun=grad_rosenbrock, max_iter=20000, learning_rate=0.001)

animar_gd(
    trayectoria,
    rosen_xy,
    dominio=(-2, 2, -1, 3),
    minimo=(1, 1),
    nombre_gif='gd_rosenbrock.gif',
    titulo='Rosenbrock – GD'
)

Image(filename='gd_rosenbrock.gif')

# ---------- Animación Superficie 3D PSO (Rosenbrock) ----------
np.random.seed(56)
options1 = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
bounds = (np.array([-2, -1]), np.array([2, 3]))
optimizer1 = ps.single.GlobalBestPSO(n_particles=50, dimensions=2,
                                     options=options1, bounds=bounds)   # <- faltaba esto
cost1, pos1 = optimizer1.optimize(fx.rosenbrock, iters=100)

limits = [(-2, 2), (-1, 3)]                       # [(xmin, xmax), (ymin, ymax)]
m1 = Mesher(func=fx.rosenbrock, limits=limits, levels=np.logspace(-1, 3.5, 25))
d1 = Designer(limits=limits, label=['x', 'y'])    # <- esto controla los ejes

animation1 = plot_contour(pos_history=optimizer1.pos_history,
                          mesher=m1, designer=d1, mark=(1, 1))
animation1.save('pso_rosenbrock.gif', writer='imagemagick', fps=10)