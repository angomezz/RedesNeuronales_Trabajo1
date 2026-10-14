"""
MÓDULO PRINCIPAL Y CLI - TRABAJO 1 (PARTE 1)

Permite ejecutar corridas individuales, el arnés de corridas repetidas o
generar las animaciones GIF, con todos los parámetros editables desde la
línea de comandos (sin tocar el código fuente).

Ejecutar desde la raíz del repositorio:

  python -m src.parte1.main --modo demo --algoritmo gd --funcion rosenbrock
  python -m src.parte1.main --modo demo --algoritmo pso --popsize 100 --iteraciones 200 --w 0.7
  python -m src.parte1.main --modo demo --algoritmo gd --funcion schwefel --learning_rate 0.05
  python -m src.parte1.main --modo arnes --runs 30 --guardar_csv resultados/arnes.csv
  python -m src.parte1.main --modo arnes --funcion rosenbrock --algoritmo de --runs 10
  python -m src.parte1.main --modo gif --algoritmo gd --learning_rate 0.002 --salida figuras

Los valores por defecto de cada algoritmo se definen una sola vez en DEFECTOS
y los usan por igual el demo y el arnés, de modo que lo que se ve en vivo
coincide con lo reportado en las tablas.
"""

import argparse
import csv
import os
import time

import numpy as np

from .experimentos import run_one_experiment, run_repeated_experiment
from .funciones import (
    grad_rosenbrock,
    grad_schwefel,
    pyswarm_wrapper,
    rosenbrock,
    schwefel,
)
from .gradiente import grad_desc
from .visualizacion import animar_gd, crear_animacion_pso_rosenbrock, rosen_xy

BASE_SEED = 56

PROBLEMAS = {
    "rosenbrock": {
        "objective": rosenbrock,
        "gradient": grad_rosenbrock,
        "domain": (-2.0, 2.0),
    },
    "schwefel": {
        "objective": schwefel,
        "gradient": grad_schwefel,
        "domain": (-500.0, 500.0),
    },
}
ALGORITMOS = ("gd", "pso", "ea", "de")

# Tasa de aprendizaje de GD según la función (valores usados en el arnés).
LR_GD = {"rosenbrock": 1e-5, "schwefel": 0.1}

# Valores por defecto de cada algoritmo (idénticos a los del arnés original).
DEFECTOS = {
    "gd": {"iteraciones": 2000, "tolerancia": 1e-5},
    "pso": {"iteraciones": 100, "popsize": 50, "c1": 0.5, "c2": 0.3, "w": 0.9},
    "ea": {"iteraciones": 100},
    "de": {"iteraciones": 200, "popsize": 15, "tolerancia": 1e-6},
}

# Valores por defecto de la animación de GD (necesita más pasos para verse).
GIF_GD = {"learning_rate": 0.001, "iteraciones": 20000}


def construir_parser():
    parser = argparse.ArgumentParser(
        description=(
            "CLI para Optimización Numérica y Metaheurística - "
            "Trabajo 1 (Parte 1)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Ejecutar desde la raíz del repositorio:")[1],
    )

    parser.add_argument(
        "--modo",
        type=str,
        default="demo",
        choices=["demo", "arnes", "gif"],
        help=(
            "'demo' (una corrida), 'arnes' (corridas repetidas con "
            "estadísticas) o 'gif' (animaciones)."
        ),
    )
    parser.add_argument(
        "--funcion",
        type=str,
        default=None,
        choices=[*PROBLEMAS, "todas"],
        help=(
            "Función objetivo. Demo: rosenbrock por defecto. "
            "Arnés: 'todas' por defecto."
        ),
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        choices=[2, 3],
        help="Dimensión. Demo: 2 por defecto. Arnés: 2 y 3 por defecto.",
    )
    parser.add_argument(
        "--algoritmo",
        type=str,
        default=None,
        choices=[*ALGORITMOS, "todos"],
        help=(
            "'gd' (gradiente), 'pso' (enjambre), 'ea' (evolutivo) o "
            "'de' (evolución diferencial). Demo: de por defecto. "
            "Arnés y gif: 'todos' por defecto."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=BASE_SEED,
        help="Semilla aleatoria (en el arnés es la semilla base).",
    )

    grupo = parser.add_argument_group(
        "Hiperparámetros (si se omiten, se usan los de DEFECTOS)"
    )
    grupo.add_argument(
        "--iteraciones",
        "--max_iter",
        dest="iteraciones",
        type=int,
        default=None,
        help=(
            "Iteraciones: max_iter en GD, iteraciones en PSO, "
            "generaciones en EA, maxiter en DE."
        ),
    )
    grupo.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Tasa de aprendizaje de GD (por defecto depende de la función).",
    )
    grupo.add_argument(
        "--tolerancia",
        type=float,
        default=None,
        help="Tolerancia de parada en GD y DE.",
    )
    grupo.add_argument(
        "--popsize",
        type=int,
        default=None,
        help=(
            "PSO: número de partículas. DE: multiplicador de scipy "
            "(población real = popsize x dimensión). EA: sin efecto."
        ),
    )
    grupo.add_argument("--w", type=float, default=None, help="PSO: inercia.")
    grupo.add_argument(
        "--c1", type=float, default=None, help="PSO: coeficiente cognitivo."
    )
    grupo.add_argument(
        "--c2", type=float, default=None, help="PSO: coeficiente social."
    )

    arnes = parser.add_argument_group("Arnés")
    arnes.add_argument(
        "--runs", type=int, default=30, help="Corridas independientes."
    )
    arnes.add_argument(
        "--umbral",
        type=float,
        default=1e-2,
        help="Umbral de f(x*) para contar una corrida como exitosa.",
    )
    arnes.add_argument(
        "--guardar_csv",
        type=str,
        default=None,
        help="Ruta de un CSV donde guardar el resumen del arnés.",
    )

    parser.add_argument(
        "--salida",
        type=str,
        default="figuras",
        help="Carpeta donde se guardan los GIF (modo gif).",
    )
    return parser


def elegir(valor, defecto):
    return defecto if valor is None else valor


def construir_parametros(algoritmo, funcion, args):
    """Parámetros del algoritmo con las claves que espera experimentos.py."""
    d = DEFECTOS[algoritmo]
    if algoritmo == "gd":
        return {
            "max_iter": elegir(args.iteraciones, d["iteraciones"]),
            "learning_rate": elegir(args.learning_rate, LR_GD[funcion]),
            "tolerance": elegir(args.tolerancia, d["tolerancia"]),
        }
    if algoritmo == "pso":
        return {
            "n_particles": elegir(args.popsize, d["popsize"]),
            "iterations": elegir(args.iteraciones, d["iteraciones"]),
            "options": {
                "c1": elegir(args.c1, d["c1"]),
                "c2": elegir(args.c2, d["c2"]),
                "w": elegir(args.w, d["w"]),
            },
        }
    if algoritmo == "ea":
        return {"generations": elegir(args.iteraciones, d["iteraciones"])}
    return {
        "maxiter": elegir(args.iteraciones, d["iteraciones"]),
        "popsize": elegir(args.popsize, d["popsize"]),
        "tolerance": elegir(args.tolerancia, d["tolerancia"]),
    }


def poblacion_efectiva(algoritmo, parametros, dim):
    """Población real que usa cada algoritmo (para comparar con justicia)."""
    if algoritmo == "pso":
        return parametros["n_particles"]
    if algoritmo == "de":
        # Se asume scipy.optimize.differential_evolution: popsize x dimensión.
        return parametros["popsize"] * dim
    return None


def ejecutar_demo(args):
    funcion = args.funcion or "rosenbrock"
    algoritmo = args.algoritmo or "de"
    dim = args.dim or 2
    problema = PROBLEMAS[funcion]
    parametros = construir_parametros(algoritmo, funcion, args)
    poblacion = poblacion_efectiva(algoritmo, parametros, dim)

    print("\n=======================================================")
    print(" EJECUTANDO DEMO EN VIVO")
    print(
        f" Algoritmo: {algoritmo.upper()} | "
        f"Función: {funcion.capitalize()} | Dim: {dim}D | "
        f"Semilla: {args.seed}"
    )
    print(f" Parámetros: {parametros}")
    if poblacion is not None:
        print(f" Población efectiva: {poblacion}")
    print("=======================================================\n")

    inicio = time.perf_counter()
    resultado = run_one_experiment(
        algorithm=algoritmo,
        objective=problema["objective"],
        gradient=problema["gradient"],
        dimension=dim,
        domain=problema["domain"],
        seed=args.seed,
        parameters=parametros,
    )
    duracion = time.perf_counter() - inicio

    print("--- RESULTADOS OBTENIDOS ---")
    print(f"Coordenadas del Óptimo x*: {resultado['x_opt']}")
    print(f"Valor Óptimo Hallado f(x*): {resultado['f_opt']:.8f}")
    print(f"Evaluaciones de f(x): {resultado['evals_f']}")
    print(f"Evaluaciones de Gradiente ∇f(x): {resultado['evals_grad']}")
    print(f"Tiempo de ejecución: {duracion:.2f} s")
    print("=======================================================\n")


def ejecutar_arnes(args):
    funciones = (
        list(PROBLEMAS)
        if args.funcion in (None, "todas")
        else [args.funcion]
    )
    algoritmos = (
        list(ALGORITMOS)
        if args.algoritmo in (None, "todos")
        else [args.algoritmo]
    )
    dimensiones = [2, 3] if args.dim is None else [args.dim]

    print(f"\n=== EJECUTANDO ARNÉS ({args.runs} CORRIDAS POR CONFIGURACIÓN) ===")
    print(f"Semilla base: {args.seed} | Umbral de éxito: {args.umbral}\n")

    filas = []
    inicio = time.perf_counter()
    for nombre in funciones:
        problema = PROBLEMAS[nombre]
        print(f"\n************ {nombre.upper()} ************")
        for dim in dimensiones:
            for algoritmo in algoritmos:
                parametros = construir_parametros(algoritmo, nombre, args)
                _, resumen = run_repeated_experiment(
                    algorithm=algoritmo,
                    objective=problema["objective"],
                    gradient=problema["gradient"],
                    dimension=dim,
                    domain=problema["domain"],
                    runs=args.runs,
                    base_seed=args.seed,
                    parameters=parametros,
                    success_threshold=args.umbral,
                )
                print(
                    f"{resumen['algorithm'].upper():<4} "
                    f"{resumen['dimension']}D | "
                    f"media={resumen['mean']:.6f} ± "
                    f"{resumen['std']:.6f} | "
                    f"mejor={resumen['best']:.6f} | "
                    f"peor={resumen['worst']:.6f} | "
                    f"éxito={resumen['success_rate']:.1f}% | "
                    f"evals f={resumen['mean_evals_f']:.2f} | "
                    f"evals grad={resumen['mean_evals_grad']:.2f}"
                )
                filas.append({"funcion": nombre, **resumen})

    print(f"\nTiempo total del arnés: {time.perf_counter() - inicio:.1f} s")

    if args.guardar_csv and filas:
        carpeta = os.path.dirname(args.guardar_csv)
        if carpeta:
            os.makedirs(carpeta, exist_ok=True)
        with open(args.guardar_csv, "w", newline="", encoding="utf-8") as f:
            escritor = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
            escritor.writeheader()
            escritor.writerows(filas)
        print(f"Resumen guardado en '{args.guardar_csv}'.")


def generar_gifs(args):
    algoritmos = (
        ["gd", "pso"]
        if args.algoritmo in (None, "todos")
        else [args.algoritmo]
    )
    os.makedirs(args.salida, exist_ok=True)
    print("\n=== GENERANDO ANIMACIONES GIF (Rosenbrock, 2D) ===")

    if "gd" in algoritmos:
        learning_rate = elegir(args.learning_rate, GIF_GD["learning_rate"])
        max_iter = elegir(args.iteraciones, GIF_GD["iteraciones"])
        extra = {}
        if args.tolerancia is not None:
            extra["tolerancia"] = args.tolerancia

        rng = np.random.default_rng(args.seed)
        x0 = rng.uniform(-2, 2, 2)
        trayectoria = grad_desc(
            x0,
            grad_fun=grad_rosenbrock,
            max_iter=max_iter,
            learning_rate=learning_rate,
            **extra,
        )
        ruta = os.path.join(args.salida, "gd_rosenbrock.gif")
        animar_gd(
            trayectoria,
            rosen_xy,
            dominio=(-2, 2, -1, 3),
            minimo=(1, 1),
            nombre_gif=ruta,
            titulo="Rosenbrock - GD",
        )
        print(
            f"GIF '{ruta}' generado "
            f"(lr={learning_rate}, max_iter={max_iter}, semilla={args.seed})."
        )

    if "pso" in algoritmos:
        if any(
            valor is not None
            for valor in (args.popsize, args.w, args.c1, args.c2)
        ):
            print(
                "Aviso: la animación de PSO usa la configuración definida en "
                "visualizacion.py; --popsize, --w, --c1 y --c2 no la modifican."
            )
        ruta = os.path.join(args.salida, "pso_rosenbrock.gif")
        crear_animacion_pso_rosenbrock(
            objective=rosenbrock,
            objective_wrapper=pyswarm_wrapper,
            nombre_gif=ruta,
            seed=args.seed,
        )
        print(f"GIF '{ruta}' generado (semilla={args.seed}).\n")


def validar_argumentos(parser, args):
    if args.modo == "demo":
        if args.funcion == "todas" or args.algoritmo == "todos":
            parser.error(
                "El modo demo ejecuta una sola corrida: indique una función "
                "y un algoritmo concretos."
            )
    if args.modo == "gif":
        if args.funcion not in (None, "rosenbrock"):
            parser.error("Las animaciones solo existen para Rosenbrock.")
        if args.algoritmo in ("ea", "de"):
            parser.error("Las animaciones solo existen para 'gd' y 'pso'.")
        if args.dim == 3:
            parser.error("Las animaciones solo existen en 2D.")
    if args.runs < 1:
        parser.error("--runs debe ser al menos 1.")
    if args.algoritmo == "ea" and args.popsize is not None:
        print("Aviso: --popsize no tiene efecto sobre EA (tamaño interno).")


def main():
    parser = construir_parser()
    args = parser.parse_args()
    validar_argumentos(parser, args)

    if args.modo == "demo":
        ejecutar_demo(args)
    elif args.modo == "arnes":
        ejecutar_arnes(args)
    else:
        generar_gifs(args)


if __name__ == "__main__":
    main()