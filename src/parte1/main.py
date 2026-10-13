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
from pyswarms.utils.plotters import plot_contour
from pyswarms.utils.plotters.formatters import Designer
from pyswarms.utils.plotters.formatters import Mesher
from IPython.core.display import Image
from matplotlib.colors import LogNorm
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation
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
#<<<<<<< HEAD

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
rng = np.random.default_rng(0)
#=======
rng = np.random.default_rng(BASE_SEED)
#>>>>>>> a7770cb037c5f1ef587ac01e61bf7f787888b202
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

#<<<<<<< HEAD
Image(filename='gd_rosenbrock.gif')

# ---------- Animación Superficie 3D PSO (Rosenbrock) ----------
options1 = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
bounds = (np.array([-2, -1]), np.array([2, 3]))
optimizer1 = ps.single.GlobalBestPSO(n_particles=50, dimensions=2,
                                        options=options1, bounds=bounds)   # <- faltaba esto
cost1, pos1 = optimizer1.optimize(rosenbrock, iters=100)

limits = [(-2, 2), (-1, 3)]                       # [(xmin, xmax), (ymin, ymax)]
m1 = Mesher(func=rosenbrock, limits=limits, levels=np.logspace(-1, 3.5, 25))
d1 = Designer(limits=limits, label=['x', 'y'])    # <- esto controla los ejes

animation1 = plot_contour(pos_history=optimizer1.pos_history,
                            mesher=m1, designer=d1, mark=(1, 1))
animation1.save('pso_rosenbrock.gif', writer='imagemagick', fps=10)
#=======
crear_animacion_pso_rosenbrock(
    objective=rosenbrock,
    objective_wrapper=pyswarm_wrapper,
    seed=BASE_SEED,
)
#>>>>>>> a7770cb037c5f1ef587ac01e61bf7f787888b202
