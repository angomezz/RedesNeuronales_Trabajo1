"""
Arnés de 30 corridas independientes del algoritmo genético (GA) para el TSP de España.
Persona 4 (P4).

Ejecuta el GA de ga_tsp.py con semillas fijas (1, 2, ..., 30) sobre la matriz de costos
y reporta: mejor, peor, media, desviación estándar (poblacional), mediana, evaluaciones,
tiempo promedio y tasa de éxito.

Tasa de éxito: porcentaje de corridas cuyo costo queda a menos del 3 % del mejor costo
encontrado en las 30 corridas (mismo criterio que el arnés del ACO, para poder comparar).

Uso (desde la raíz del repositorio):
    python src/parte2/benchmark_ga_30.py
    python src/parte2/benchmark_ga_30.py --runs 10 --pop 200 --gens 300
    python src/parte2/benchmark_ga_30.py --matrix data/cost_matrix.csv --umbral 0.05
"""

import argparse
import csv
import json
import pathlib
import sys
import time

import numpy as np

AQUI = pathlib.Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(AQUI))
from ga_tsp import CONFIG, nearest_neighbor, run_ga, tour_cost  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="30 corridas independientes del GA (P4)")
    p.add_argument("--matrix", default=str(RAIZ / "data" / "cost_matrix.csv"))
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--pop", type=int, default=300)
    p.add_argument("--gens", type=int, default=800)
    p.add_argument("--k", type=int, default=CONFIG["k"])
    p.add_argument("--pc", type=float, default=CONFIG["pc"])
    p.add_argument("--pm", type=float, default=CONFIG["pm"])
    p.add_argument("--elite", type=int, default=CONFIG["elite"])
    p.add_argument("--umbral", type=float, default=0.03,
                   help="éxito si el costo queda a menos de este porcentaje del mejor (0.03 = 3 %%)")
    p.add_argument("--out", default=str(RAIZ / "results" / "ga_benchmark_30"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    M = np.loadtxt(args.matrix, delimiter=",")
    nn_cost = tour_cost(M, nearest_neighbor(M))

    print("=" * 70)
    print(f"   ARNÉS DE {args.runs} CORRIDAS DEL ALGORITMO GENÉTICO (P4)")
    print(f"   pop={args.pop} gens={args.gens} k={args.k} pc={args.pc} pm={args.pm} elite={args.elite}")
    print("=" * 70)

    filas = []
    t0 = time.time()
    for s in range(1, args.runs + 1):
        cfg = dict(CONFIG)
        cfg.update({"seed": s, "pop": args.pop, "gens": args.gens, "k": args.k,
                    "pc": args.pc, "pm": args.pm, "elite": args.elite, "show": False})
        r = run_ga(M, cfg)
        filas.append({"semilla": s, "costo": round(r["best_cost"], 2),
                      "evaluaciones": r["evaluations"], "segundos": r["seconds"],
                      "recorrido": " ".join(map(str, r["best_tour"]))})
        print(f"  corrida {s:2d}: costo = {r['best_cost']:9.2f} € | {r['seconds']:5.1f} s")

    costos = np.array([f["costo"] for f in filas])
    mejor = costos.min()
    exito = 100.0 * np.mean(costos <= mejor * (1 + args.umbral))
    resumen = {
        "corridas": args.runs, "mejor": float(mejor), "peor": float(costos.max()),
        "media": float(costos.mean()), "desv_est": float(costos.std()),
        "mediana": float(np.median(costos)),
        "tasa_exito_%": float(exito), "umbral": args.umbral,
        "evaluaciones_por_corrida": int(filas[0]["evaluaciones"]),
        "segundos_promedio": float(np.mean([f["segundos"] for f in filas])),
        "vecino_mas_cercano": float(nn_cost),
        "config": vars(args),
    }

    salida = pathlib.Path(args.out)
    salida.mkdir(parents=True, exist_ok=True)
    with open(salida / "corridas.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    (salida / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False),
                                         encoding="utf-8")

    print("-" * 70)
    print(f"Mejor: {resumen['mejor']:.2f} € | Peor: {resumen['peor']:.2f} € | "
          f"Media: {resumen['media']:.2f} ± {resumen['desv_est']:.2f} €")
    print(f"Mediana: {resumen['mediana']:.2f} € | Éxito (≤ {args.umbral:.0%} del mejor): {exito:.1f} %")
    print(f"Evaluaciones por corrida: {resumen['evaluaciones_por_corrida']} | "
          f"Vecino más cercano: {nn_cost:.2f} €")
    print(f"Tiempo total: {time.time() - t0:.1f} s | Resultados en {salida}/")


if __name__ == "__main__":
    main()
