"""
Arnés de 30 corridas independientes de ACO para el TSP en España.
Módulo estadístico para Persona 5 (P5).

Ejecuta 30 corridas independientes con semillas del 1 al 30.
Reporta:
  - Mejor valor (mínimo)
  - Peor valor (máximo)
  - Media
  - Desviación estándar
  - Tiempo promedio de cómputo
  - Tasa de éxito (definida como porcentaje de soluciones a <= 3% del óptimo encontrado)

Uso:
  python scripts/benchmark_aco_30.py [--hourly-value 20.0] [--iters 80] [--ants 47]
"""

import argparse
import csv
import json
import pathlib
import time
import numpy as np

from aco_tsp import CONFIG, run_aco

def run_30_benchmark(hourly_value=20.0, n_runs=30, iters=80, ants=47):
    # Cargar matrices base
    dist_matrix = np.loadtxt("data/dist_matrix.csv", delimiter=",")
    time_matrix = np.loadtxt("data/time_matrix.csv", delimiter=",")
    toll_matrix = np.loadtxt("data/toll_matrix.csv", delimiter=",")

    # Seat León 1.5 TSI 130CV, WLTP 5.6 L/100km, Gasolina 95: 1.62 €/L
    fuel_cost_matrix = dist_matrix * (5.6 / 100.0) * 1.62
    cost_matrix = (time_matrix * hourly_value) + toll_matrix + fuel_cost_matrix

    print("=" * 70)
    print(f"   ARNÉS DE {n_runs} CORRIDAS INDEPENDIENTES DE ACO (P5)")
    print(f"   Configuración: V_hora = €{hourly_value:.2f}/h | Hormigas = {ants} | Iteraciones = {iters}")
    print("=" * 70)

    records = []
    t_start_total = time.time()

    for s in range(1, n_runs + 1):
        cfg = dict(CONFIG)
        cfg.update({
            "ants": ants,
            "iters": iters,
            "alpha": 1.0,
            "beta": 3.0,
            "rho": 0.1,
            "seed": s,
        })
        res = run_aco(cost_matrix, cfg)
        records.append({
            "run": s,
            "seed": s,
            "best_cost": round(res["best_cost"], 2),
            "evaluations": res["evaluations"],
            "seconds": res["seconds"],
            "best_tour": res["best_tour"],
        })
        print(f"Corrida {s:02d}/{n_runs:02d} | Semilla: {s:<4} | Mejor Costo: €{res['best_cost']:,.2f} | Tiempo: {res['seconds']:.2f} s")

    total_duration = time.time() - t_start_total

    costs = np.array([r["best_cost"] for r in records])
    times = np.array([r["seconds"] for r in records])
    evals = records[0]["evaluations"]

    min_val = np.min(costs)
    max_val = np.max(costs)
    mean_val = np.mean(costs)
    std_val = np.std(costs)

    # Definición de tasa de éxito: porcentaje de corridas con costo <= 3% sobre el mejor valor global
    threshold = min_val * 1.03
    success_count = int(np.sum(costs <= threshold))
    success_rate = (success_count / n_runs) * 100.0

    print("-" * 70)
    print("                      RESUMEN ESTADÍSTICO")
    print("-" * 70)
    print(f"  Número de corridas independientes : {n_runs}")
    print(f"  Mejor valor (Mínimo)              : €{min_val:,.2f}")
    print(f"  Peor valor (Máximo)               : €{max_val:,.2f}")
    print(f"  Media                             : €{mean_val:,.2f}")
    print(f"  Desviación Estándar               : €{std_val:,.2f}")
    print(f"  Tiempo promedio por corrida       : {np.mean(times):.2f} s (Total: {total_duration:.1f} s)")
    print(f"  Evaluaciones f(x) por corrida     : {evals:,}")
    print(f"  Umbral de éxito (<= +3% del mín)  : €{threshold:,.2f}")
    print(f"  Tasa de éxito                     : {success_rate:.1f}% ({success_count}/{n_runs})")
    print("=" * 70)

    out_dir = pathlib.Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_runs": n_runs,
        "hourly_value": hourly_value,
        "ants": ants,
        "iters": iters,
        "min_cost": round(float(min_val), 2),
        "max_cost": round(float(max_val), 2),
        "mean_cost": round(float(mean_val), 2),
        "std_cost": round(float(std_val), 2),
        "mean_seconds": round(float(np.mean(times)), 2),
        "total_seconds": round(total_duration, 2),
        "evaluations_per_run": evals,
        "threshold_success_3pct": round(float(threshold), 2),
        "success_rate_pct": round(float(success_rate), 1),
        "runs": records,
    }

    with open(out_dir / "benchmark_30_runs.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "benchmark_30_runs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "seed", "best_cost", "evaluations", "seconds"])
        writer.writeheader()
        for r in records:
            writer.writerow({k: r[k] for k in ["run", "seed", "best_cost", "evaluations", "seconds"]})

    print(f"Resultados guardados en results/benchmark_30_runs.json y results/benchmark_30_runs.csv")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arnés de 30 corridas independientes de ACO")
    parser.add_argument("--hourly-value", type=float, default=20.0, help="Valor de la hora del vendedor en €/h")
    parser.add_argument("--runs", type=int, default=30, help="Número de corridas")
    parser.add_argument("--iters", type=int, default=80, help="Número de iteraciones por corrida")
    parser.add_argument("--ants", type=int, default=47, help="Número de hormigas por iteración")
    args = parser.parse_args()
    run_30_benchmark(hourly_value=args.hourly_value, n_runs=args.runs, iters=args.iters, ants=args.ants)
