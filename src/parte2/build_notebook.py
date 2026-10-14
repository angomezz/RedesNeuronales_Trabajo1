"""
Script para generar el notebook interactivo Trabajo_1_ACO.ipynb
"""
import json

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

# Celda 0
add_md("""
# Trabajo 1: Optimización Combinatoria con Colonias de Hormigas (ACO)
## Recorrido Óptimo de las 47 Capitales de la España Peninsular (TSP)
**Rol:** Persona 5 (P5) — Juan José Baena Garcia (`jubaenag@unal.edu.co`)  
**Curso:** Redes Neuronales y Algoritmos Bioinspirados — Semestre 2026-02  
**Universidad Nacional de Colombia — Facultad de Minas**  
**Departamento de Ciencias de la Computación y de la Decisión**
""")

# Celda 1
add_code("""
import csv
import json
import pathlib
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display

# Importamos las utilidades modulares desarrolladas en scripts/
import sys
sys.path.insert(0, "scripts")
from aco_tsp import CONFIG, load_problem, nearest_neighbor, run_aco, tour_cost

print("Entorno y módulos de ACO cargados exitosamente.")
""")

# Celda 2
add_md("""
---
## 1. Datos del TSP: Las 47 Capitales Peninsulares de España

En cumplimiento con el enunciado:
- Se excluyen las islas (**Palma de Mallorca**, **Las Palmas de Gran Canaria**, **Santa Cruz de Tenerife**) por ser insulares y no accesibles por vía terrestre.
- Se excluyen **Ceuta** y **Melilla** por no ser provincias ni contar con acceso terrestre peninsular.
- Total exacto: **47 capitales peninsulares**.
- Todos los datos se cargan localmente desde `data/` para garantizar reproducibilidad total sin conexión a internet.
""")

# Celda 3
add_code("""
# Cargar lista de ciudades y coordenadas verificadas
df_cities = pd.read_csv("data/cities.csv")
df_coords = pd.read_csv("data/coordinates.csv")
df_capitals = pd.merge(df_cities, df_coords, on=["id", "ciudad"])

print(f"Total de capitales cargadas: {len(df_capitals)}")
display(df_capitals.head(10))
""")

# Celda 4
add_md("""
---
## 2. Modelo de Costo Real: Vehículo, Combustible, Peajes y Valor de la Hora

El costo de viajar entre la capital $i$ y la capital $j$ se define como:

$$C_{ij}(V_h) = (V_h \times T_{ij}) + P_{ij} + \\left( D_{ij} \\times \\frac{\\text{Consumo}}{100} \\times \\text{PrecioCombustible} \\right)$$

- **Vehículo elegido:** SEAT León 1.5 TSI 130 CV (gasolina). Consumo homologado WLTP: **5.6 L / 100 km** (Fuente: SEAT España / IDAE).
- **Precio del combustible:** Gasolina 95 E5: **1.62 € / L** (Fuente: MITECO Geoportal Gasolineras).
- **Peajes vigentes:** Tarifas oficiales 2026 (AP-68 Bilbao-Zaragoza: 33.75 €, AP-61 Madrid-Segovia: 7.30 €, AP-51 Madrid-Ávila: 9.85 €, AP-66 Huerna: 14.30 €, AP-71: 5.60 €, AP-46: 5.40 €). Autopistas liberadas (AP-1, AP-7 y AP-2) tienen peaje 0.00 €.
- **Velocidad promedio estimada:** 105.0 km/h en autovías/autopistas. Factor de sinuosidad vial: 1.28.
""")

# Celda 5
add_code("""
# Cargar las matrices de distancias, tiempos, peajes y costos
dist_matrix = np.loadtxt("data/dist_matrix.csv", delimiter=",")
time_matrix = np.loadtxt("data/time_matrix.csv", delimiter=",")
toll_matrix = np.loadtxt("data/toll_matrix.csv", delimiter=",")
cost_matrix = np.loadtxt("data/cost_matrix.csv", delimiter=",")

print(f"Dimensiones de la matriz de costos: {cost_matrix.shape}")
print(f"Costo promedio de arco entre capitales: €{np.mean(cost_matrix[cost_matrix > 0]):.2f}")
print(f"Peaje máximo registrado en un arco: €{np.max(toll_matrix):.2f}")
""")

# Celda 6
add_md("""
---
## 3. Algoritmo de Colonias de Hormigas (ACO) Implementado Desde Cero

El algoritmo utiliza la variante **Sistema de Colonias de Hormigas Elitista (EAS - Dorigo & Stützle, 2004)**:
1. **Regla de transición estocástica:**
   $$p_{ij}^k = \\frac{[\\tau_{ij}]^\\alpha [\\eta_{ij}]^\\beta}{\\sum_{l \\in J_k(i)} [\\tau_{il}]^\\alpha [\\eta_{il}]^\\beta}$$
   donde $\\eta_{ij} = 1 / C_{ij}$ es la visibilidad heurística y $J_k(i)$ son las ciudades no visitadas.
2. **Evaporación canónica:** $\\tau_{ij} \\leftarrow (1 - \\rho)\\tau_{ij}$.
3. **Depósito simétrico:** $\\tau_{ij} \\leftarrow \\tau_{ij} + \\sum_{k=1}^m \\frac{Q}{C_k} + e \\cdot \\frac{Q}{C_{\\text{best}}}$.
4. **Regla de comparación justa:** Conteo explícito de soluciones evaluadas ($m \\times T$ evaluaciones de $f$).
""")

# Celda 7
add_code("""
# Ejecución de una corrida representativa de ACO sobre España
cfg = dict(CONFIG)
cfg.update({
    "matrix": "data/cost_matrix.csv",
    "coords": "data/coordinates.csv",
    "ants": 47,
    "iters": 80,
    "alpha": 1.0,
    "beta": 3.0,
    "rho": 0.1,
    "q": 100.0,
    "elite": 2.0,
    "seed": 56,
    "show": False,
    "gif": True,
    "out_gif": "results/spain_tsp_aco.gif"
})

M, coords, names = load_problem(cfg)
res = run_aco(M, cfg)
nn_cost = tour_cost(M, nearest_neighbor(M))

print("=" * 60)
print("   RESULTADOS DE LA CORRIDA BASE DE ACO (Vh = 20 €/h)")
print("=" * 60)
print(f"Mejor costo encontrado por ACO : €{res['best_cost']:,.2f}")
print(f"Costo línea base (Vecino más cercano): €{nn_cost:,.2f}")
print(f"Evaluaciones de la función objetivo : {res['evaluations']:,} circuitos cerrados")
print(f"Tiempo de cómputo : {res['seconds']:.2f} segundos")
print("=" * 60)
""")

# Celda 8
add_code("""
# Visualización del recorrido óptimo y curva de convergencia
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

t = np.array(res["best_tour"] + [res["best_tour"][0]])
ax1.scatter(coords[:, 0], coords[:, 1], color="darkred", s=40, zorder=4)
ax1.plot(coords[t, 0], coords[t, 1], "-b", linewidth=1.6, alpha=0.85, zorder=3)
ax1.set_title(f"Recorrido Óptimo ACO (Costo: €{res['best_cost']:,.2f})", fontweight="bold")
ax1.set_xlabel("Longitud (°)")
ax1.set_ylabel("Latitud (°)")
ax1.grid(True, linestyle=":", alpha=0.6)

# Etiquetas de algunas ciudades clave
key_cities = {"Madrid", "Barcelona", "Valencia", "Sevilla", "Bilbao", "A Coruña", "Zaragoza"}
for idx, nm in enumerate(names):
    if nm in key_cities:
        ax1.annotate(nm, (coords[idx, 0] + 0.15, coords[idx, 1] + 0.1), fontsize=8, fontweight="bold")

ax2.plot(res["history_best"], label="Mejor hasta la iteración", color="royalblue", linewidth=2)
ax2.plot(res["history_mean"], label="Media de las hormigas", color="orange", alpha=0.6)
ax2.axhline(nn_cost, color="gray", linestyle="--", label="Vecino más cercano")
ax2.set_xlabel("Iteración")
ax2.set_ylabel("Costo del circuito (€)")
ax2.set_title("Curva de Convergencia ACO", fontweight="bold")
ax2.legend()
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()
""")

# Celda 9
add_md("""
---
## 4. Arnés Estadístico de 30 Corridas Independientes

En cumplimiento con la **regla de comparación justa**:
- Se ejecutan **30 corridas independientes** con semillas aleatorias documentadas (1 a 30).
- Se reportan: Mínimo, Máximo, Media, Desviación Estándar y Tasa de Éxito ($\le 3\%$ del óptimo global).
""")

# Celda 10
add_code("""
# Cargar o mostrar los resultados del arnés de 30 corridas
benchmark_path = pathlib.Path("results/benchmark_30_runs.json")

if benchmark_path.exists():
    with open(benchmark_path, "r", encoding="utf-8") as f:
        data_bench = json.load(f)
        
    df_runs = pd.DataFrame(data_bench["runs"])
    print(f"Estadísticas de {data_bench['n_runs']} corridas independientes:")
    print(f"  - Mejor valor (Mínimo)  : €{data_bench['min_cost']:,.2f}")
    print(f"  - Peor valor (Máximo)   : €{data_bench['max_cost']:,.2f}")
    print(f"  - Media                 : €{data_bench['mean_cost']:,.2f}")
    print(f"  - Desviación Estándar   : €{data_bench['std_cost']:,.2f}")
    print(f"  - Tiempo promedio       : {data_bench['mean_seconds']:.2f} s")
    print(f"  - Evaluaciones f(x)     : {data_bench['evaluations_per_run']:,}")
    print(f"  - Tasa de éxito (<+3%)  : {data_bench['success_rate_pct']:.1f}%")
    
    # Gráfica de distribución de costos
    plt.figure(figsize=(8, 4))
    plt.boxplot(df_runs["best_cost"], vert=False, patch_artist=True,
                boxprops=dict(facecolor="lightblue", color="blue"))
    plt.axvline(data_bench["min_cost"], color="green", linestyle="--", label=f"Mínimo: €{data_bench['min_cost']:,.2f}")
    plt.axvline(data_bench["mean_cost"], color="red", linestyle=":", label=f"Media: €{data_bench['mean_cost']:,.2f}")
    plt.xlabel("Costo del circuito (€)")
    plt.title("Distribución de Costos en 30 Corridas Independientes de ACO")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.show()
else:
    print("Ejecuta 'python scripts/benchmark_aco_30.py' para generar la tabla completa.")
""")

# Celda 11
add_md("""
---
## 5. Estudio del Impacto del Valor de la Hora del Vendedor ($V_h$)

### Pregunta de Investigación:
> *¿Cambia la ruta óptima al variar el valor de la hora? ¿A partir de qué valor cambia?*

Se evalúa la función de costo sobre el rango $V_h \\in [0, 100]\\text{ \\euro{}/h}$, midiendo el trade-off entre:
1. **Tiempo de viaje:** $V_h \\times T_{ij}$
2. **Costo de peajes:** $P_{ij}$
3. **Consumo de combustible:** $D_{ij} \\times \\frac{5.6}{100} \\times 1.62$
""")

# Celda 12
add_code("""
# Cargar y visualizar los resultados del estudio de sensibilidad
df_sens = pd.read_csv("results/hourly_value_study.csv")
display(df_sens)

# Mostrar la gráfica generada por el script
if pathlib.Path("results/hourly_value_sensitivity.png").exists():
    display(Image("results/hourly_value_sensitivity.png"))
""")

# Celda 13
add_md("""
### Hallazgos Principales del Estudio:
1. **¿Cambia la ruta óptima?** **SÍ**. La secuencia de arcos visitados cambia significativamente.
2. **¿A partir de qué valor?** A partir de **$V_h = 5.0\\text{ \\euro{}/h}$**.
   - Con $V_h = 0\\text{ \\euro{}/h}$ el conductor tiene costo de tiempo nulo, por lo que asume **0.00 € en peajes** y maximiza el ahorro de combustible.
   - Tan pronto como $V_h \\ge 5\\text{ \\euro{}/h}$, el costo temporal justifica pagar autopistas de peaje clave (como la AP-66 del Huerna: 14.30 €) para reducir el tiempo total de 62.3 h a 61.1 h.
   - Con $V_h = 100\\text{ \\euro{}/h}$, el tiempo representa más del 90% del costo total, elevando los peajes a 19.70 € para exprimir cada minuto de reducción en el recorrido.
""")

# Celda 14
add_md("""
---
## 6. GIF Animado de la Evolución de la Solución sobre España
""")

# Celda 15
add_code("""
gif_path = "results/spain_tsp_aco.gif"
if pathlib.Path(gif_path).exists():
    display(Image(gif_path))
else:
    print("Genera el GIF ejecutando: python scripts/aco_tsp.py --matrix data/cost_matrix.csv --coords data/coordinates.csv --gif --no-show")
""")

# Celda 16
add_md("""
---
## 7. Controlador Interactivo para Sustentación en Vivo (< 2 Minutos)

Permite al profesor y monitor solicitar cambios de parámetros en vivo (por ejemplo: *"aumente el valor de la hora a 80 €/h"*, *"cambie beta a 5.0"*, o *"modifique la evaporación rho a 0.2"*), prediciendo el resultado y ejecutándolo en segundos sin editar código fuente.
""")

# Celda 17
add_code("""
def demo_sustentacion(hourly_value=25.0, ants=47, iters=50, alpha=1.0, beta=3.0, rho=0.1, seed=42):
    t_start = time.time()
    
    # Recalcular matriz para el valor de hora solicitado
    fuel = dist_matrix * (5.6 / 100.0) * 1.62
    c_mat = (time_matrix * hourly_value) + toll_matrix + fuel
    
    cfg_demo = dict(CONFIG)
    cfg_demo.update({
        "ants": ants,
        "iters": iters,
        "alpha": alpha,
        "beta": beta,
        "rho": rho,
        "seed": seed
    })
    
    res = run_aco(c_mat, cfg_demo)
    dur = time.time() - t_start
    
    tour = res["best_tour"]
    n = len(tour)
    d = sum(dist_matrix[tour[k], tour[(k + 1) % n]] for k in range(n))
    t = sum(time_matrix[tour[k], tour[(k + 1) % n]] for k in range(n))
    p = sum(toll_matrix[tour[k], tour[(k + 1) % n]] for k in range(n))
    
    print("=" * 65)
    print("      DEMOSTRACIÓN EN VIVO - ACO TSP ESPAÑA (P5)")
    print("=" * 65)
    print(f"Parámetros: V_hora=€{hourly_value}/h | Hormigas={ants} | Iters={iters} | alpha={alpha} | beta={beta} | rho={rho}")
    print(f"Tiempo de ejecución       : {dur:.2f} s  (Criterio: < 120 s)")
    print(f"Evaluaciones f(x)         : {res['evaluations']:,} circuitos cerrados")
    print(f"Costo Total de la Ruta    : €{res['best_cost']:,.2f}")
    print(f"Distancia Total           : {d:,.1f} km")
    print(f"Tiempo Total de Viaje     : {t:.2f} horas")
    print(f"Peajes en la Ruta         : €{p:,.2f}")
    print("=" * 65)

# Prueba en vivo instantánea
demo_sustentacion(hourly_value=30.0, ants=47, iters=40)
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Guardar en RedesNeuronales_Trabajo1 y en Trabajo 1/
with open("Trabajo_1_ACO.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

import os
parent_nb = os.path.join("..", "Trabajo_1.ipynb")
with open(parent_nb, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebooks generados con éxito:")
print("  - Trabajo_1_ACO.ipynb (en RedesNeuronales_Trabajo1)")
print("  - ../Trabajo_1.ipynb (en la carpeta padre)")
