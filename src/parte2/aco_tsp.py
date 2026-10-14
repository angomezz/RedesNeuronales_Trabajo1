"""
Algoritmo de Colonias de Hormigas (ACO) para el problema del viajero (Trabajo 1, Parte 2).
Implementación para Persona 5 (P5).

Esquema canónico: Sistema de Colonias de Hormigas Elitista (EAS - Dorigo & Stützle, 2004).
Permite ejecutar sobre ciudades sintéticas (seed 56) o sobre la matriz de costos real de España.

Uso:
    python scripts/aco_tsp.py --seed 56 --n_cities 40 --ants 40 --iters 100
    python scripts/aco_tsp.py --matrix data/cost_matrix.csv --coords data/coordinates.csv --ants 47 --iters 100 --no-show
    python scripts/aco_tsp.py --matrix data/cost_matrix.csv --coords data/coordinates.csv --gif --no-show
"""

import argparse
import csv
import json
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix

CONFIG = {
    "seed": 56,                  # Semilla aleatoria para reproducibilidad
    "n_cities": 40,              # Número de ciudades sintéticas (si no hay matriz)
    "matrix": None,              # Archivo CSV con matriz de costos (n x n)
    "coords": None,              # Archivo CSV con coordenadas (id, ciudad, lat, lon)
    "ants": 47,                  # Número de hormigas por iteración
    "iters": 100,                # Número de iteraciones
    "alpha": 1.0,                # Peso de feromona
    "beta": 3.0,                 # Peso de visibilidad heurística (1/costo)
    "rho": 0.1,                  # Tasa de evaporación de feromonas (1 - rho de retención)
    "q": 100.0,                  # Constante de depósito de feromona
    "elite": 2.0,                # Peso elitista para el mejor recorrido global
    "patience": 0,               # Iteraciones sin mejora para parar (0 = no parar)
    "out": "results/aco_result.json",
    "gif": False,                # Generar GIF animado de evolución del recorrido
    "out_gif": "results/spain_tsp_aco.gif",
    "show": True,
}


def parse_config(argv=None):
    """Lee la configuración por defecto y la sobrescribe con argumentos de CLI."""
    p = argparse.ArgumentParser(description="Colonias de Hormigas (ACO) para el TSP")
    for nombre, valor in CONFIG.items():
        if isinstance(valor, bool):
            continue
        tipo = type(valor) if valor is not None else str
        p.add_argument(f"--{nombre}", type=tipo, default=valor)
    p.add_argument("--no-show", action="store_true", help="no abrir ventanas de gráficos")
    p.add_argument("--gif", action="store_true", help="generar GIF animado del recorrido")
    args, _ = p.parse_known_args(argv)
    cfg = vars(args)
    cfg["show"] = not cfg.pop("no_show")
    if args.gif:
        cfg["gif"] = True
    return cfg


def load_problem(cfg):
    """Devuelve (M, coords, names). Coords y names pueden ser None."""
    if cfg["matrix"]:
        M = np.loadtxt(cfg["matrix"], delimiter=",")
        assert M.shape[0] == M.shape[1], "La matriz de costos debe ser cuadrada"
        coords, names = None, None
        if cfg["coords"] and pathlib.Path(cfg["coords"]).exists():
            with open(cfg["coords"], mode="r", encoding="utf-8") as f:
                reader = list(csv.DictReader(f))
            coords = np.array([[float(r["lon"]), float(r["lat"])] for r in reader])
            names = [r["ciudad"] for r in reader]
        return M, coords, names

    # Ciudades sintéticas con seed fija
    n = cfg["n_cities"]
    coords = np.random.RandomState(cfg["seed"]).random_sample(2 * n).reshape(n, 2)
    names = [f"Ciudad {i}" for i in range(n)]
    return distance_matrix(coords, coords), coords, names


def tour_cost(M, tour):
    """Costo del ciclo cerrado: suma de los arcos, incluido el regreso a la salida."""
    return float(M[tour, np.roll(tour, -1)].sum())


def nearest_neighbor(M, start=0):
    """Línea base: ir siempre a la ciudad más cercana no visitada."""
    visited = [start]
    while len(visited) < len(M):
        d = M[visited[-1]].copy()
        d[visited] = np.inf
        visited.append(int(np.argmin(d)))
    return np.array(visited)


def select_next_city(rng, current, unvisited, pheromone, eta, alpha, beta):
    """Selecciona la siguiente ciudad usando la regla probabilística de la ruleta."""
    unvisited_arr = np.array(list(unvisited), dtype=int)
    tau = pheromone[current, unvisited_arr] ** alpha
    eta_val = eta[current, unvisited_arr] ** beta
    weights = tau * eta_val
    total = weights.sum()

    if total == 0 or np.isnan(total):
        probs = np.ones(len(unvisited_arr)) / len(unvisited_arr)
    else:
        probs = weights / total

    chosen_idx = rng.choice(len(unvisited_arr), p=probs)
    return int(unvisited_arr[chosen_idx])


def run_aco(M, cfg):
    """
    Ejecuta el algoritmo ACO canónico con refuerzo elitista.
    Retorna diccionario con resultados, costos y caminos de mejora.
    """
    rng = np.random.default_rng(cfg["seed"])
    n = len(M)
    n_ants = cfg["ants"]
    n_iters = cfg["iters"]
    alpha = cfg["alpha"]
    beta = cfg["beta"]
    rho = cfg["rho"]
    Q = cfg["q"]
    elite_w = cfg["elite"]

    t0 = time.time()
    evals = 0

    # Visibilidad heurística (1 / costo) protegida
    with np.errstate(divide="ignore"):
        eta = 1.0 / M
        np.fill_diagonal(eta, 0.0)

    # Inicialización de feromona basada en la heurística del vecino más cercano
    nn_tour = nearest_neighbor(M)
    nn_cost = tour_cost(M, nn_tour)
    tau_0 = n_ants / (nn_cost if nn_cost > 0 else 1.0)
    pheromone = np.full((n, n), tau_0, dtype=float)

    mejor_costo = float("inf")
    mejor_tour = None
    hist_mejor = []
    hist_media = []
    caminos = []
    sin_mejora = 0

    for it in range(1, n_iters + 1):
        tours = []
        costos = []

        for k in range(n_ants):
            start = k % n
            tour = [start]
            unvisited = set(range(n))
            unvisited.remove(start)
            current = start

            while unvisited:
                nxt = select_next_city(rng, current, unvisited, pheromone, eta, alpha, beta)
                tour.append(nxt)
                unvisited.remove(nxt)
                current = nxt

            c = tour_cost(M, np.array(tour))
            evals += 1
            tours.append(tour)
            costos.append(c)

            if c < mejor_costo - 1e-12:
                mejor_costo = c
                mejor_tour = list(tour)
                caminos.append((it, list(tour)))
                sin_mejora = 0

        # Evaporación canónica: tau <- (1 - rho) * tau
        pheromone = (1.0 - rho) * pheromone

        # Depósito de feromona por todas las hormigas
        for tour, cost in zip(tours, costos):
            deposit = Q / cost
            for i in range(n):
                u, v = tour[i], tour[(i + 1) % n]
                pheromone[u, v] += deposit
                pheromone[v, u] += deposit

        # Refuerzo elitista de la mejor solución global
        if elite_w > 0 and mejor_tour is not None:
            elite_deposit = elite_w * (Q / mejor_costo)
            for i in range(n):
                u, v = mejor_tour[i], mejor_tour[(i + 1) % n]
                pheromone[u, v] += elite_deposit
                pheromone[v, u] += elite_deposit

        # Piso de seguridad contra estancamiento absoluto
        pheromone = np.maximum(pheromone, tau_0 * 0.001)

        hist_mejor.append(mejor_costo)
        hist_media.append(float(np.mean(costos)))

        if cfg["patience"] and sin_mejora >= cfg["patience"]:
            break

    return {
        "best_tour": mejor_tour,
        "best_cost": mejor_costo,
        "evaluations": evals,
        "iterations_run": len(hist_mejor),
        "seconds": round(time.time() - t0, 2),
        "history_best": hist_mejor,
        "history_mean": hist_media,
        "improvement_paths": caminos,
    }


def generate_gif(coords, names, caminos, history_best, out_gif_path):
    """Genera un GIF animado de la evolución del recorrido sobre las coordenadas."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    if coords is None:
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    def update(frame_idx):
        ax.clear()
        it, tour = caminos[frame_idx]
        cost = history_best[min(it - 1, len(history_best) - 1)]

        ax.set_title(
            f"Optimización ACO - TSP España\nMejora en Iteración {it} | Costo: €{cost:,.2f}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
        ax.grid(True, linestyle=":", alpha=0.6)

        # Ciudades
        ax.scatter(coords[:, 0], coords[:, 1], c="darkred", s=45, zorder=4, edgecolors="black")

        # Nombres de capitales clave
        if names:
            key_cities = {"Madrid", "Barcelona", "Valencia", "Sevilla", "Bilbao", "A Coruña", "Zaragoza"}
            for idx, nm in enumerate(names):
                if nm in key_cities:
                    ax.annotate(nm, (coords[idx, 0] + 0.15, coords[idx, 1] + 0.1), fontsize=8, fontweight="bold")

        # Circuito cerrado
        closed_tour = tour + [tour[0]]
        ax.plot(coords[closed_tour, 0], coords[closed_tour, 1], color="royalblue", linewidth=1.8, alpha=0.85, zorder=3)

    anim = FuncAnimation(fig, update, frames=len(caminos), interval=300)
    out_path = pathlib.Path(out_gif_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=3))
    plt.close()
    print(f"GIF animado generado exitosamente: {out_path}")


if __name__ == "__main__":
    cfg = parse_config()
    M, coords, names = load_problem(cfg)
    res = run_aco(M, cfg)

    nn = nearest_neighbor(M)
    nn_cost = tour_cost(M, nn)

    print("=" * 60)
    print("   COLONIAS DE HORMIGAS (ACO) - TSP")
    print("=" * 60)
    print(f"Ciudades: {len(M)} | Semilla: {cfg['seed']}")
    print(f"Hormigas: {cfg['ants']} | Iteraciones: {res['iterations_run']}")
    print(f"Mejor costo ACO:           {res['best_cost']:,.2f}")
    print(f"Costo vecino más cercano:  {nn_cost:,.2f}")
    print(f"Evaluaciones de f(x):      {res['evaluations']:,}")
    print(f"Tiempo de ejecución:       {res['seconds']} s")
    print("-" * 60)

    salida = pathlib.Path(cfg["out"])
    salida.parent.mkdir(parents=True, exist_ok=True)
    guardar = {k: v for k, v in res.items() if k != "improvement_paths"}
    guardar.update({
        "config": cfg,
        "nearest_neighbor_cost": nn_cost,
        "improvement_paths": res["improvement_paths"],
    })
    salida.write_text(json.dumps(guardar, indent=2), encoding="utf-8")
    print(f"Resultados guardados en {salida}")

    # Gráfico lado a lado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    if coords is not None:
        t = np.array(res["best_tour"] + [res["best_tour"][0]])
        ax1.scatter(coords[:, 0], coords[:, 1], color="darkred", s=30)
        ax1.plot(coords[t, 0], coords[t, 1], "-b", linewidth=1.5)
        ax1.set_title("Mejor recorrido (ACO)")
        ax1.grid(True, linestyle=":", alpha=0.6)
    else:
        ax1.text(0.5, 0.5, "Sin coordenadas\n(matriz de costos)", ha="center", va="center")
        ax1.set_axis_off()

    ax2.plot(res["history_best"], label="Mejor hasta ahora", color="royalblue", linewidth=2)
    ax2.plot(res["history_mean"], label="Media de las hormigas", color="orange", alpha=0.6)
    ax2.axhline(nn_cost, color="gray", linestyle="--", label="Vecino más cercano")
    ax2.set_xlabel("Iteración")
    ax2.set_ylabel("Costo del recorrido")
    ax2.set_title("Convergencia ACO")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(salida.with_suffix(".png"), dpi=120)

    if cfg["gif"] and coords is not None:
        generate_gif(coords, names, res["improvement_paths"], res["history_best"], cfg["out_gif"])

    if cfg["show"]:
        plt.show()


"""
============================================================
   COLONIAS DE HORMIGAS (ACO) - TSP
============================================================
Ciudades: 40 | Semilla: 56
Hormigas: 47 | Iteraciones: 100
Mejor costo ACO:           5.33
Costo vecino más cercano:  6.77
Evaluaciones de f(x):      4,700
Tiempo de ejecución:       2.67 s
------------------------------------------------------------
Resultados guardados en results\aco_result.json
"""