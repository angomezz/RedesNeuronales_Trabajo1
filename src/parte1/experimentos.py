import numpy as np
import pyswarms as ps

from . import funciones
from .gradiente import grad_desc
from .heuristicas import algort_evol, run_differential_evolution


def _reset_counters(objective):
    if objective is funciones.rosenbrock:
        funciones.contador_rosenbrock = 0
    elif objective is funciones.schwefel:
        funciones.contador_schwefel = 0
    else:
        raise ValueError("Función objetivo no reconocida.")
    funciones.contador_grad = 0


def _function_evaluations(objective):
    if objective is funciones.rosenbrock:
        return funciones.contador_rosenbrock
    if objective is funciones.schwefel:
        return funciones.contador_schwefel
    raise ValueError("Función objetivo no reconocida.")


def _run_gradient_descent(
    objective,
    gradient,
    dimension,
    domain,
    seed,
    max_iter,
    learning_rate,
    tolerance,
):
    _reset_counters(objective)
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(domain[0], domain[1], dimension)
    trajectory = grad_desc(
        x0,
        grad_fun=gradient,
        max_iter=max_iter,
        learning_rate=learning_rate,
        tolerancia=tolerance,
    )
    x_opt = np.asarray(trajectory[-1])
    f_opt = objective(x_opt)
    return x_opt, f_opt, trajectory


def _run_pso(objective, dimension, domain, seed, n_particles, iterations, options):
    _reset_counters(objective)
    np.random.seed(seed)
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=dimension,
        options=options,
        bounds=(
            np.full(dimension, domain[0]),
            np.full(dimension, domain[1]),
        ),
    )
    f_opt, x_opt = optimizer.optimize(
        lambda population: funciones.pyswarm_wrapper(population, objective),
        iters=iterations,
        verbose=False,
    )
    return np.asarray(x_opt), f_opt, optimizer.pos_history


def run_one_experiment(
    algorithm,
    objective,
    gradient,
    dimension,
    domain,
    seed,
    parameters,
):
    """Ejecuta una corrida y devuelve sus métricas observables."""
    if algorithm == "gd":
        x_opt, f_opt, history = _run_gradient_descent(
            objective,
            gradient,
            dimension,
            domain,
            seed,
            parameters["max_iter"],
            parameters["learning_rate"],
            parameters["tolerance"],
        )
    elif algorithm == "pso":
        x_opt, f_opt, history = _run_pso(
            objective,
            dimension,
            domain,
            seed,
            parameters["n_particles"],
            parameters["iterations"],
            parameters["options"],
        )
    elif algorithm == "ea":
        _reset_counters(objective)
        lower = np.full(dimension, domain[0])
        upper = np.full(dimension, domain[1])
        f_opt, x_opt = algort_evol(
            objective,
            n_dim=dimension,
            lim_inf=lower,
            lim_sup=upper,
            semilla=seed,
            n_iter=parameters["generations"],
        )
        history = None
    elif algorithm == "de":
        _reset_counters(objective)
        result = run_differential_evolution(
            funcion=objective,
            bounds=[domain] * dimension,
            maxiter=parameters["maxiter"],
            popsize=parameters["popsize"],
            seed=seed,
            tolerance=parameters["tolerance"],
        )
        x_opt, f_opt = result.x, result.fun
        history = None
    else:
        raise ValueError(f"Algoritmo no reconocido: {algorithm}")

    return {
        "algorithm": algorithm,
        "dimension": dimension,
        "seed": seed,
        "x_opt": np.asarray(x_opt),
        "f_opt": float(f_opt),
        "evals_f": _function_evaluations(objective),
        "evals_grad": funciones.contador_grad,
        "history": history,
    }


def summarize_results(results, success_threshold):
    values = np.asarray([result["f_opt"] for result in results])
    evals_f = np.asarray([result["evals_f"] for result in results])
    evals_grad = np.asarray([result["evals_grad"] for result in results])
    successes = np.count_nonzero(values <= success_threshold)

    return {
        "algorithm": results[0]["algorithm"],
        "dimension": results[0]["dimension"],
        "runs": len(results),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "best": float(np.min(values)),
        "worst": float(np.max(values)),
        "success_rate": float(100 * successes / len(results)),
        "mean_evals_f": float(np.mean(evals_f)),
        "mean_evals_grad": float(np.mean(evals_grad)),
    }


def run_repeated_experiment(
    algorithm,
    objective,
    gradient,
    dimension,
    domain,
    runs=30,
    base_seed=56,
    parameters=None,
    success_threshold=1e-2,
):
    """Ejecuta varias semillas y calcula las estadísticas del enunciado."""
    if parameters is None:
        parameters = {}

    results = [
        run_one_experiment(
            algorithm=algorithm,
            objective=objective,
            gradient=gradient,
            dimension=dimension,
            domain=domain,
            seed=base_seed + run,
            parameters=parameters,
        )
        for run in range(runs)
    ]
    return results, summarize_results(results, success_threshold)
