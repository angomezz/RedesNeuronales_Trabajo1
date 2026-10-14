"""
Algoritmo genetico para el problema del viajero (Trabajo 1, Parte 2).

Mismo esquema del notebook de colonias de hormigas visto en clase: ciudades
sinteticas con seed(56), matriz de distancias con scipy y recorrido graficado.

Uso (un solo comando, parzmetros por linea de comandos):
    python scripts/ga_tsp.py --seed 56 --n_cities 40 --pop 200 --gens 500
    python scripts/ga_tsp.py --matrix data/cost_matrix.csv --pop 300 --gens 800 --no-show

Tambien se puede ejecutar celda por celda (VS Code / Jupyter): las celdas 1 a 4
definen todo y la celda 5 en adelante corre el experimento.
"""

# %% [1] Imports y configuracion
import argparse
import json
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix

CONFIG = {
    "seed": 56,          # semilla del GA (la de las ciudades sinteticas es la misma)
    "n_cities": 40,      # solo se usa con ciudades sinteticas
    "matrix": None,      # ruta a un CSV con la matriz de costos (n x n); None = sintetico
    "pop": 200,          # tamaño de la poblacion
    "gens": 500,         # numero de generaciones
    "k": 3,              # tamaño del torneo
    "pc": 0.9,           # probabilidad de cruce
    "pm": 0.2,           # probabilidad de mutacion (por individuo)
    "elite": 2,          # individuos que pasan intactos a la siguiente generacion
    "patience": 0,       # generaciones sin mejora para parar (0 = no parar antes)
    "out": "results/ga_result.json",
    "show": True,
}


def parse_config(argv=None):
    """Lee la configuracion por defecto y la sobrescribe con la iínea de comandos."""
    p = argparse.ArgumentParser(description="Algoritmo genetico para el TSP")
    for nombre, valor in CONFIG.items():
        if isinstance(valor, bool):
            continue
        tipo = type(valor) if valor is not None else str
        p.add_argument(f"--{nombre}", type=tipo, default=valor)
    p.add_argument("--no-show", action="store_true", help="no abrir ventanas de graficos")
    args, _ = p.parse_known_args(argv)  # parse_known_args: no falla dentro de un notebook
    cfg = vars(args)
    cfg["show"] = not cfg.pop("no_show")
    return cfg


# %% [2] Datos del problema
def load_problem(cfg):
    """Devuelve (M, coords). coords es None si la matriz viene de un archivo."""
    if cfg["matrix"]:
        M = np.loadtxt(cfg["matrix"], delimiter=",")
        assert M.shape[0] == M.shape[1], "La matriz de costos debe ser cuadrada"
        return M, None
    # Mismas ciudades que el notebook de clase (np.random.seed + np.random.random)
    n = cfg["n_cities"]
    coords = np.random.RandomState(cfg["seed"]).random_sample(2 * n).reshape(n, 2)
    return distance_matrix(coords, coords), coords


# %% [3] Utilidades: costo de un recorrido y linea base
def tour_cost(M, tour):
    """Costo del ciclo cerrado: suma de los arcos, incluido el regreso a la salida."""
    return float(M[tour, np.roll(tour, -1)].sum())


def nearest_neighbor(M, start=0):
    """Linea base: ir siempre a la ciudad mas cercana no visitada."""
    visited = [start]
    while len(visited) < len(M):
        d = M[visited[-1]].copy()
        d[visited] = np.inf
        visited.append(int(np.argmin(d)))
    return np.array(visited)


# %% [4] Operadores geneticos
def tournament(rng, costs, k):
    """Seleccion por torneo: de k individuos al azar gana el de menor costo."""
    candidatos = rng.choice(len(costs), size=k, replace=False)
    return candidatos[np.argmin(costs[candidatos])]


def order_crossover(rng, p1, p2):
    """OX: copia un segmento de p1 y completa con las ciudades restantes en el orden de p2."""
    n = len(p1)
    a, b = np.sort(rng.choice(n, size=2, replace=False))
    b += 1                                    # el segmento es p1[a:b]
    hijo = np.full(n, -1)
    hijo[a:b] = p1[a:b]
    en_segmento = set(p1[a:b].tolist())
    orden_p2 = np.concatenate([p2[b:], p2[:b]])           # p2 leido desde el corte b
    restantes = [g for g in orden_p2 if g not in en_segmento]
    posiciones = list(range(b, n)) + list(range(0, a))    # huecos, desde b en forma circular
    hijo[posiciones] = restantes
    return hijo


def inversion_mutation(rng, tour):
    """Mutacion por inversion: invierte un segmento al azar (equivale a un movimiento 2-opt)."""
    i, j = np.sort(rng.choice(len(tour), size=2, replace=False))
    nuevo = tour.copy()
    nuevo[i:j + 1] = nuevo[i:j + 1][::-1]
    return nuevo


# %% [5] Algoritmo genetico
def run_ga(M, cfg):
    """Corre el GA. Devuelve un diccionario con el mejor recorrido, su costo e historial."""
    rng = np.random.default_rng(cfg["seed"])
    n, pop_size, n_elite = len(M), cfg["pop"], cfg["elite"]
    t0 = time.time()
    evals = 0

    poblacion = np.array([rng.permutation(n) for _ in range(pop_size)])
    costos = np.array([tour_cost(M, ind) for ind in poblacion])
    evals += pop_size

    mejor_costo = float(costos.min())
    mejor_tour = poblacion[np.argmin(costos)].copy()
    hist_mejor, hist_media = [mejor_costo], [float(costos.mean())]
    caminos = [(0, mejor_tour.tolist())]      # mejor recorrido cada vez que mejora (para el GIF)
    sin_mejora = 0

    for gen in range(1, cfg["gens"] + 1):
        orden = np.argsort(costos)
        nueva = [poblacion[i].copy() for i in orden[:n_elite]]
        costos_nueva = [costos[i] for i in orden[:n_elite]]

        while len(nueva) < pop_size:
            p1 = poblacion[tournament(rng, costos, cfg["k"])]
            p2 = poblacion[tournament(rng, costos, cfg["k"])]
            hijo = order_crossover(rng, p1, p2) if rng.random() < cfg["pc"] else p1.copy()
            if rng.random() < cfg["pm"]:
                hijo = inversion_mutation(rng, hijo)
            nueva.append(hijo)
            costos_nueva.append(tour_cost(M, hijo))
            evals += 1

        poblacion, costos = np.array(nueva), np.array(costos_nueva)

        if costos.min() < mejor_costo - 1e-12:
            mejor_costo = float(costos.min())
            mejor_tour = poblacion[np.argmin(costos)].copy()
            caminos.append((gen, mejor_tour.tolist()))
            sin_mejora = 0
        else:
            sin_mejora += 1
        hist_mejor.append(mejor_costo)
        hist_media.append(float(costos.mean()))

        if cfg["patience"] and sin_mejora >= cfg["patience"]:
            break

    return {
        "best_tour": mejor_tour.tolist(),
        "best_cost": mejor_costo,
        "evaluations": evals,
        "generations_run": len(hist_mejor) - 1,
        "seconds": round(time.time() - t0, 2),
        "history_best": hist_mejor,
        "history_mean": hist_media,
        "improvement_paths": caminos,
    }


# %% [6] Ejecucion del experimento
if __name__ == "__main__":
    cfg = parse_config()
    M, coords = load_problem(cfg)
    res = run_ga(M, cfg)

    nn = nearest_neighbor(M)
    nn_cost = tour_cost(M, nn)
    print(f"Ciudades: {len(M)} | semilla: {cfg['seed']}")
    print(f"Mejor costo GA:            {res['best_cost']:.4f}")
    print(f"Costo vecino mas cercano:  {nn_cost:.4f}")
    print(f"Evaluaciones de costo:     {res['evaluations']}")
    print(f"Generaciones / tiempo:     {res['generations_run']} / {res['seconds']} s")

    salida = pathlib.Path(cfg["out"])
    salida.parent.mkdir(parents=True, exist_ok=True)
    guardar = {k: v for k, v in res.items() if k != "improvement_paths"}
    guardar.update({"config": cfg, "nearest_neighbor_cost": nn_cost,
                    "improvement_paths": res["improvement_paths"]})
    salida.write_text(json.dumps(guardar, indent=2), encoding="utf-8")
    print(f"Resultados guardados en {salida}")

    # %% [7] Graficos lado a lado: recorrido y convergencia
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    if coords is not None:
        t = np.array(res["best_tour"] + [res["best_tour"][0]])
        ax1.scatter(coords[:, 0], coords[:, 1])
        ax1.plot(coords[t, 0], coords[t, 1], "-r")
        ax1.set_title("Mejor recorrido (GA)")
        ax1.grid()
    else:
        ax1.text(0.5, 0.5, "Sin coordenadas\n(matriz de costos)", ha="center", va="center")
        ax1.set_axis_off()
    ax2.plot(res["history_best"], label="Mejor hasta ahora")
    ax2.plot(res["history_mean"], label="Media de la poblacion", alpha=0.6)
    ax2.axhline(nn_cost, color="gray", linestyle="--", label="Vecino mas cercano")
    ax2.set_xlabel("Generación")
    ax2.set_ylabel("Costo del recorrido")
    ax2.set_title("Convergencia")
    ax2.legend()
    ax2.grid()
    fig.tight_layout()
    fig.savefig(salida.with_suffix(".png"), dpi=120)
    if cfg["show"]:
        plt.show()


"""
Ciudades: 40 | semilla: 56
Mejor costo GA:            5.3969
Costo vecino mas cercano:  6.7662
Evaluaciones de costo:     99200
Generaciones / tiempo:     500 / 3.94 s
Resultados guardados en results\ga_result.json
"""