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
import pyswarms as ps
from . import funciones
from .funciones import grad_rosenbrock, grad_schwefel, pyswarm_wrapper, rosenbrock, schwefel
from .gradiente import grad_desc
from .experimentos import run_repeated_experiment
from .visualizacion import animar_gd, crear_animacion_pso_rosenbrock, rosen_xy, schwefel_xy

BASE_SEED = 56


# =============================================================================
# PRUEBAS INICIALES
# =============================================================================

x0 = np.array([-1.2, 1.0, 0.5])
trayectoria = grad_desc(x0, grad_fun=grad_rosenbrock, max_iter=20000, learning_rate=0.001)

print(f"Total de evaluaciones de f: {funciones.contador_f}")
print(f"Total de evaluaciones de gradiente (grad_f): {funciones.contador_grad}")

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
funciones.contador_rosenbrock = 0
np.random.seed(BASE_SEED)
xmin, xmax, ymin, ymax = -2, 2, -1, 3
options1 = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
bounds = (np.array([xmin, ymin]), np.array([xmax, ymax]))
optimizer1 = ps.single.GlobalBestPSO(n_particles=50, dimensions=2, options=options1, bounds=bounds)
cost1, pos1 = optimizer1.optimize(lambda X: pyswarm_wrapper(X, rosenbrock), iters=100, verbose=False)
print(f"Evaluaciones de Rosenbrock: {funciones.contador_rosenbrock}")

# Test inicial PSO Schwefel
funciones.contador_schwefel = 0
np.random.seed(BASE_SEED)
xmin, xmax, ymin, ymax = -500, 500, -500, 500
options1 = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
bounds = (np.array([xmin, ymin]), np.array([xmax, ymax]))
optimizer1 = ps.single.GlobalBestPSO(n_particles=50, dimensions=2, options=options1, bounds=bounds)
cost1, pos1 = optimizer1.optimize(lambda X: pyswarm_wrapper(X, schwefel), iters=100, verbose=False)
print(f"Evaluaciones de Schwefel: {funciones.contador_schwefel}")


# =============================================================================
# 6. EVALUACIONES EXPERIMENTALES (30 CORRIDAS POR ALGORITMO)
# =============================================================================

COMMON_PSO = {
    "n_particles": 50,
    "iterations": 100,
    "options": {"c1": 0.5, "c2": 0.3, "w": 0.9},
}
COMMON_EA = {"generations": 100}
COMMON_DE = {"maxiter": 200, "popsize": 15, "tolerance": 1e-6}


def imprimir_resumen(summary):
    print(
        f"{summary['algorithm'].upper()} {summary['dimension']}D | "
        f"media={summary['mean']:.6f} ± {summary['std']:.6f} | "
        f"mejor={summary['best']:.6f} | peor={summary['worst']:.6f} | "
        f"éxito={summary['success_rate']:.1f}% | "
        f"evals f={summary['mean_evals_f']:.2f} | "
        f"evals grad={summary['mean_evals_grad']:.2f}"
    )


EXPERIMENTS = [
    {
        "name": "Rosenbrock",
        "objective": rosenbrock,
        "gradient": grad_rosenbrock,
        "domain": (-2.0, 2.0),
    },
    {
        "name": "Schwefel",
        "objective": schwefel,
        "gradient": grad_schwefel,
        "domain": (-500.0, 500.0),
    },
]

ALGORITHMS = {
    "gd": {
        "parameters": {
            "max_iter": 2000,
            "learning_rate": 0.00001,
            "tolerance": 0.00001,
        }
    },
    "pso": {"parameters": COMMON_PSO},
    "ea": {"parameters": COMMON_EA},
    "de": {"parameters": COMMON_DE},
}

for experiment in EXPERIMENTS:
    print(f"--- {experiment['name']} ---")
    for dimension in (2, 3):
        for algorithm, configuration in ALGORITHMS.items():
            parameters = dict(configuration["parameters"])
            if algorithm == "gd" and experiment["name"] == "Schwefel":
                parameters["learning_rate"] = 0.1

            _, summary = run_repeated_experiment(
                algorithm=algorithm,
                objective=experiment["objective"],
                gradient=experiment["gradient"],
                dimension=dimension,
                domain=experiment["domain"],
                runs=30,
                base_seed=BASE_SEED,
                parameters=parameters,
                success_threshold=1e-2,
            )
            imprimir_resumen(summary)


# =============================================================================
# 7. ANIMACIONES Y VISUALIZACIONES
# =============================================================================
rng = np.random.default_rng(BASE_SEED)
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

crear_animacion_pso_rosenbrock(
    objective=rosenbrock,
    objective_wrapper=pyswarm_wrapper,
    seed=BASE_SEED,
)