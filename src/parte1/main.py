"""
MÓDULO PRINCIPAL Y CLI - TRABAJO 1 (PARTE 1)

Permite ejecutar corridas individuales, el arnés de 30 corridas completas
o generar las animaciones GIF mediante argumentos ingresados por terminal.
"""

import argparse

from .funciones import grad_rosenbrock, grad_schwefel, pyswarm_wrapper, rosenbrock, schwefel
from .experimentos import run_one_experiment, run_repeated_experiment
from .visualizacion import animar_gd, crear_animacion_pso_rosenbrock, rosen_xy

BASE_SEED = 56


def construir_parser():
    parser = argparse.ArgumentParser(
        description=(
            "CLI para Optimización Numérica y Metaheurística - "
            "Trabajo 1 (Parte 1)"
        )
    )

    parser.add_argument(
        "--modo",
        type=str,
        default="demo",
        choices=["demo", "arnes", "gif"],
        help=(
            "Modo de ejecución: 'demo' (corrida única rápida), "
            "'arnes' (30 corridas) o 'gif' (generar animaciones)."
        ),
    )
    parser.add_argument(
        "--funcion",
        type=str,
        default="rosenbrock",
        choices=["rosenbrock", "schwefel"],
        help="Función objetivo a optimizar.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=2,
        choices=[2, 3],
        help="Dimensión del espacio de búsqueda.",
    )
    parser.add_argument(
        "--algoritmo",
        type=str,
        default="de",
        choices=["gd", "pso", "ea", "de"],
        help=(
            "Algoritmo: 'gd' (gradiente), 'pso' (enjambre), "
            "'ea' (evolutivo) o 'de' (evolución diferencial)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=BASE_SEED,
        help="Semilla aleatoria.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=2000,
        help="Iteraciones máximas para GD.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.00001,
        help="Tasa de aprendizaje para GD.",
    )
    parser.add_argument(
        "--popsize",
        type=int,
        default=50,
        help="Tamaño de población para PSO/EA/DE.",
    )
    return parser


def obtener_configuracion_problema(nombre_func):
    if nombre_func.lower() == "rosenbrock":
        return {
            "objective": rosenbrock,
            "gradient": grad_rosenbrock,
            "domain": (-2.0, 2.0),
            "xy_func": rosen_xy,
            "minimo": (1, 1),
        }
    if nombre_func.lower() == "schwefel":
        return {
            "objective": schwefel,
            "gradient": grad_schwefel,
            "domain": (-500.0, 500.0),
            "xy_func": None,
            "minimo": (420.9687, 420.9687),
        }
    raise ValueError(f"Función desconocida: {nombre_func}")


def ejecutar_demo(args):
    config = obtener_configuracion_problema(args.funcion)

    if args.algoritmo == "gd":
        params = {
            "max_iter": args.max_iter,
            "learning_rate": (
                args.learning_rate
                if args.funcion == "rosenbrock"
                else 0.1
            ),
            "tolerance": 1e-5,
        }
    elif args.algoritmo == "pso":
        params = {
            "n_particles": args.popsize,
            "iterations": 100,
            "options": {"c1": 0.5, "c2": 0.3, "w": 0.9},
        }
    elif args.algoritmo == "ea":
        params = {"generations": 100}
    else:
        params = {
            "maxiter": 200,
            "popsize": max(1, args.popsize // 3),
            "tolerance": 1e-6,
        }

    print("\n=======================================================")
    print(" EJECUTANDO DEMO EN VIVO")
    print(
        f" Algoritmo: {args.algoritmo.upper()} | "
        f"Función: {args.funcion.capitalize()} | Dim: {args.dim}D"
    )
    print(
        f" Semilla: {args.seed} | "
        f"Población/Iteraciones: {args.popsize}/{args.max_iter}"
    )
    print("=======================================================\n")

    resultado = run_one_experiment(
        algorithm=args.algoritmo,
        objective=config["objective"],
        gradient=config["gradient"],
        dimension=args.dim,
        domain=config["domain"],
        seed=args.seed,
        parameters=params,
    )

    print("--- RESULTADOS OBTENIDOS ---")
    print(f"Coordenadas del Óptimo x*: {resultado['x_opt']}")
    print(f"Valor Óptimo Hallado f(x*): {resultado['f_opt']:.8f}")
    print(f"Evaluaciones de f(x): {resultado['evals_f']}")
    print(f"Evaluaciones de Gradiente ∇f(x): {resultado['evals_grad']}")
    print("=======================================================\n")


def ejecutar_arnes_completo(base_seed):
    print("\n=== EJECUTANDO ARNÉS COMPLETO (30 CORRIDAS) ===\n")

    experiments = [
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
    algorithms = {
        "gd": {
            "parameters": {
                "max_iter": 2000,
                "learning_rate": 0.00001,
                "tolerance": 1e-5,
            }
        },
        "pso": {
            "parameters": {
                "n_particles": 50,
                "iterations": 100,
                "options": {"c1": 0.5, "c2": 0.3, "w": 0.9},
            }
        },
        "ea": {"parameters": {"generations": 100}},
        "de": {
            "parameters": {
                "maxiter": 200,
                "popsize": 15,
                "tolerance": 1e-6,
            }
        },
    }

    for experiment in experiments:
        print(f"\n************ {experiment['name'].upper()} ************")
        for dimension in (2, 3):
            for algorithm, configuration in algorithms.items():
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
                    base_seed=base_seed,
                    parameters=parameters,
                    success_threshold=1e-2,
                )
                print(
                    f"{summary['algorithm'].upper():<4} "
                    f"{summary['dimension']}D | "
                    f"media={summary['mean']:.6f} ± "
                    f"{summary['std']:.6f} | "
                    f"mejor={summary['best']:.6f} | "
                    f"peor={summary['worst']:.6f} | "
                    f"éxito={summary['success_rate']:.1f}% | "
                    f"evals f={summary['mean_evals_f']:.2f} | "
                    f"evals grad={summary['mean_evals_grad']:.2f}"
                )


def generar_gifs_demo(seed):
    print("\n=== GENERANDO ANIMACIONES GIF ===")

    import numpy as np

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2, 2, 2)
    from .gradiente import grad_desc

    trayectoria = grad_desc(
        x0,
        grad_fun=grad_rosenbrock,
        max_iter=20000,
        learning_rate=0.001,
    )
    animar_gd(
        trayectoria,
        rosen_xy,
        dominio=(-2, 2, -1, 3),
        minimo=(1, 1),
        nombre_gif="gd_rosenbrock.gif",
        titulo="Rosenbrock - GD",
    )
    print("GIF 'gd_rosenbrock.gif' generado con éxito.")

    crear_animacion_pso_rosenbrock(
        objective=rosenbrock,
        objective_wrapper=pyswarm_wrapper,
        nombre_gif="pso_rosenbrock.gif",
        seed=seed,
    )
    print("GIF 'pso_rosenbrock.gif' generado con éxito.\n")


def main():
    parser = construir_parser()
    args = parser.parse_args()

    if args.modo == "demo":
        ejecutar_demo(args)
    elif args.modo == "arnes":
        ejecutar_arnes_completo(args.seed)
    else:
        generar_gifs_demo(args.seed)


if __name__ == "__main__":
    main()
