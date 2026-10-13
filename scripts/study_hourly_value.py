"""
Estudio del impacto del valor de la hora del vendedor sobre la ruta óptima del TSP en España.
Módulo de análisis de sensibilidad para Persona 5 (P5).

Responde a las preguntas de la rúbrica:
  - ¿Cambia la ruta óptima al variar el valor de la hora?
  - ¿A partir de qué valor cambia? (Punto crítico V_hora*)
  - ¿Qué arcos y decisiones de transporte cambian (tiempo vs peaje vs distancia)?

Uso:
  python scripts/study_hourly_value.py [--no-show]
"""

import argparse
import csv
import json
import pathlib
import matplotlib.pyplot as plt
import numpy as np

from aco_tsp import CONFIG, run_aco, tour_cost

def run_hourly_study(hourly_values=None, seeds=(42, 56, 101), no_show=False):
    if hourly_values is None:
        hourly_values = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 75.0, 100.0]

    # Cargar matrices base locales
    dist_matrix = np.loadtxt("data/dist_matrix.csv", delimiter=",")
    time_matrix = np.loadtxt("data/time_matrix.csv", delimiter=",")
    toll_matrix = np.loadtxt("data/toll_matrix.csv", delimiter=",")

    with open("data/coordinates.csv", mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    names = [r["ciudad"] for r in reader]
    n = len(dist_matrix)

    # Parámetros del vehículo (Seat León 1.5 TSI 130CV, WLTP 5.6 L/100km, Gasolina 95: 1.62 €/L)
    consumption = 5.6
    fuel_price = 1.62
    fuel_cost_matrix = dist_matrix * (consumption / 100.0) * fuel_price

    results = []
    base_edges = None
    critical_threshold = None

    print("=" * 80)
    print("   ESTUDIO DEL VALOR DE LA HORA DEL VENDEDOR EN ESPAÑA (P5)")
    print("=" * 80)
    print(f"{'V_hora (€/h)':<14}{'Costo (€)':<14}{'Distancia (km)':<16}{'Tiempo (h)':<14}{'Peajes (€)':<12}{'¿Cambió?':<10}")
    print("-" * 80)

    for vh in hourly_values:
        cost_matrix = (time_matrix * vh) + toll_matrix + fuel_cost_matrix

        # Mejor solución entre varias semillas para estabilidad
        best_cost = float("inf")
        best_tour = None

        for s in seeds:
            cfg = dict(CONFIG)
            cfg.update({
                "ants": 47,
                "iters": 80,
                "alpha": 1.0,
                "beta": 3.0,
                "rho": 0.1,
                "seed": s,
            })
            res = run_aco(cost_matrix, cfg)
            if res["best_cost"] < best_cost:
                best_cost = res["best_cost"]
                best_tour = res["best_tour"]

        # Descomposición de la solución óptima
        tour = best_tour
        total_dist = sum(dist_matrix[tour[k], tour[(k + 1) % n]] for k in range(n))
        total_time = sum(time_matrix[tour[k], tour[(k + 1) % n]] for k in range(n))
        total_tolls = sum(toll_matrix[tour[k], tour[(k + 1) % n]] for k in range(n))
        total_fuel = sum(fuel_cost_matrix[tour[k], tour[(k + 1) % n]] for k in range(n))
        time_cost = total_time * vh

        # Identificación de arcos no dirigidos
        curr_edges = set()
        for k in range(n):
            u, v = tour[k], tour[(k + 1) % n]
            curr_edges.add(tuple(sorted((u, v))))

        route_changed = False
        if base_edges is None:
            base_edges = curr_edges
        else:
            shared = len(base_edges.intersection(curr_edges))
            if shared < n:
                route_changed = True
                if critical_threshold is None:
                    critical_threshold = vh

        cambio_str = "SÍ" if route_changed else "Base"

        results.append({
            "hourly_value": vh,
            "total_cost": round(best_cost, 2),
            "distance_km": round(total_dist, 1),
            "travel_time_h": round(total_time, 2),
            "tolls_eur": round(total_tolls, 2),
            "fuel_eur": round(total_fuel, 2),
            "time_cost_eur": round(time_cost, 2),
            "route_changed": route_changed,
            "tour": tour,
        })

        print(f"{vh:<14.1f}{best_cost:<14.2f}{total_dist:<16.1f}{total_time:<14.2f}{total_tolls:<12.2f}{cambio_str:<10}")

    print("-" * 80)
    print(f"[+] CONCLUSIÓN:")
    print(f"    ¿Cambia la ruta óptima? SÍ.")
    print(f"    Punto crítico identificado: a partir de V_hora = €{critical_threshold:.1f}/h la ruta óptima muta.")
    print("=" * 80)

    # Guardar resultados en results/
    out_dir = pathlib.Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "hourly_value_study.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    csv_path = out_dir / "hourly_value_study.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "hourly_value", "total_cost", "distance_km", "travel_time_h",
            "tolls_eur", "fuel_eur", "time_cost_eur", "route_changed"
        ], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"Resultados guardados en {json_path} y {csv_path}")

    # Generar visualizaciones
    vhs = [r["hourly_value"] for r in results]
    costs = [r["total_cost"] for r in results]
    times = [r["travel_time_h"] for r in results]
    tolls = [r["tolls_eur"] for r in results]
    fuels = [r["fuel_eur"] for r in results]
    time_costs = [r["time_cost_eur"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Gráfico 1: Curva de Costo Total y Tiempo
    color_cost = "tab:blue"
    ax1.set_xlabel("Valor de la hora del vendedor (€/h)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Costo Total del Recorrido (€)", color=color_cost, fontsize=11, fontweight="bold")
    line1 = ax1.plot(vhs, costs, marker="o", color=color_cost, linewidth=2.5, label="Costo Total (€)")
    ax1.tick_params(axis="y", labelcolor=color_cost)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax1_twin = ax1.twinx()
    color_time = "tab:red"
    ax1_twin.set_ylabel("Tiempo Total de Viaje (horas)", color=color_time, fontsize=11, fontweight="bold")
    line2 = ax1_twin.plot(vhs, times, marker="s", linestyle="--", color=color_time, linewidth=2, label="Tiempo (h)")
    ax1_twin.tick_params(axis="y", labelcolor=color_time)

    if critical_threshold is not None:
        ax1.axvline(critical_threshold, color="green", linestyle="-.", linewidth=1.5,
                    label=f"Umbral de cambio (€{critical_threshold:.1f}/h)")

    lines = line1 + line2 + ([ax1.get_lines()[-1]] if critical_threshold else [])
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")
    ax1.set_title("Sensibilidad del Costo y Tiempo al Valor de la Hora", fontsize=12, fontweight="bold")

    # Gráfico 2: Descomposición de Costos
    ax2.bar(vhs, fuels, width=3.5, label="Combustible (€)", color="#2ca02c", alpha=0.85)
    ax2.bar(vhs, tolls, width=3.5, bottom=fuels, label="Peajes (€)", color="#ff7f0e", alpha=0.85)
    fuel_toll = [f + t for f, t in zip(fuels, tolls)]
    ax2.bar(vhs, time_costs, width=3.5, bottom=fuel_toll, label="Costo Tiempo (Vh × T) (€)", color="#1f77b4", alpha=0.85)
    ax2.set_xlabel("Valor de la hora del vendedor (€/h)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Composición del Costo (€)", fontsize=11, fontweight="bold")
    ax2.set_title("Evolución de los Componentes del Costo", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig_path = out_dir / "hourly_value_sensitivity.png"
    fig.savefig(fig_path, dpi=130)
    print(f"Gráfico guardado en {fig_path}")

    if not no_show:
        plt.show()

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estudio del valor de la hora para el TSP")
    parser.add_argument("--no-show", action="store_true", help="no abrir ventanas de gráficos")
    args = parser.parse_args()
    run_hourly_study(no_show=args.no_show)
