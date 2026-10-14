"""
Construye las matrices de distancia, tiempo, peajes y costo total del TSP en España.
Guarda los archivos en data/:
    - data/dist_matrix.csv (km)
    - data/time_matrix.csv (horas)
    - data/toll_matrix.csv (euros)
    - data/cost_matrix.csv (euros, con valor de la hora por defecto = 20.0 €/h)

Uso:
    python scripts/build_cost_matrices.py [--hourly-value 20.0]
"""

import argparse
import csv
import math
import pathlib
import numpy as np

# Raíz del proyecto (2 niveles arriba de 'src/parte2/build_cost_matrices.py')
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent

def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia en gran círculo en km sobre la esfera terrestre (R = 6371.0 km)."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * R * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

def build_matrices(hourly_value=20.0, consumption=5.6, fuel_price=1.62):
    # Definir carpeta data respecto a la raíz del proyecto
    data_dir = BASE_DIR / "data"
    coords_path = data_dir / "coordinates.csv"
    
    if not coords_path.exists():
        raise FileNotFoundError("No se encuentra {coords_path}")

    with open(coords_path, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    n = len(reader)
    names = [row["ciudad"] for row in reader]
    lats = [float(row["lat"]) for row in reader]
    lons = [float(row["lon"]) for row in reader]

    dist_matrix = np.zeros((n, n), dtype=float)
    time_matrix = np.zeros((n, n), dtype=float)
    toll_matrix = np.zeros((n, n), dtype=float)

    # Factor de sinuosidad promedio de la red vial en España: ~1.28
    # Velocidad promedio en autopistas/autovías: 105.0 km/h
    for i in range(n):
        for j in range(n):
            if i != j:
                d_geo = haversine_km(lats[i], lons[i], lats[j], lons[j])
                d_road = round(d_geo * 1.28, 2)
                t_road = round(d_road / 105.0, 3)
                dist_matrix[i, j] = d_road
                time_matrix[i, j] = t_road

    # Peajes vigentes en la red peninsular (2026):
    toll_pairs = [
        ({"Bilbao", "Zaragoza"}, 33.75),   # AP-68 (tramo completo)
        ({"Logroño", "Zaragoza"}, 17.50),  # AP-68 (tramo sur)
        ({"Bilbao", "Logroño"}, 16.25),   # AP-68 (tramo norte)
        ({"Madrid", "Segovia"}, 7.30),    # AP-61 / Túnel de Guadarrama
        ({"Madrid", "Ávila"}, 9.85),      # AP-51 / AP-6
        ({"Oviedo", "León"}, 14.30),      # AP-66 (Autopista del Huerna)
        ({"León", "Ourense"}, 5.60),      # AP-71 (León - Astorga)
        ({"Málaga", "Granada"}, 5.40),    # AP-46 (Alto de las Pedrizas)
    ]

    for pair, toll in toll_pairs:
        matches = [idx for idx, name in enumerate(names) if name in pair]
        if len(matches) == 2:
            i, j = matches
            toll_matrix[i, j] = toll
            toll_matrix[j, i] = toll

    # Cálculo de costo de combustible: Distancia * (Consumo / 100) * PrecioCombustible
    fuel_cost_matrix = dist_matrix * (consumption / 100.0) * fuel_price

    # Costo total: (ValorHora * Tiempo) + Peajes + Combustible
    cost_matrix = (time_matrix * hourly_value) + toll_matrix + fuel_cost_matrix

    # Asegurar que la carpeta 'data' existe antes de guardar
    data_dir.mkdir(parents=True, exist_ok=True)

    # Guardar matrices
    np.savetxt(data_dir / "dist_matrix.csv", dist_matrix, delimiter=",", fmt="%.2f")
    np.savetxt(data_dir / "time_matrix.csv", time_matrix, delimiter=",", fmt="%.3f")
    np.savetxt(data_dir / "toll_matrix.csv", toll_matrix, delimiter=",", fmt="%.2f")
    np.savetxt(data_dir / "cost_matrix.csv", cost_matrix, delimiter=",", fmt="%.2f")

    print(f"Matrices generadas para {n} capitales:")
    print(f"  - data/dist_matrix.csv (km)")
    print(f"  - data/time_matrix.csv (horas)")
    print(f"  - data/toll_matrix.csv (euros)")
    print(f"  - data/cost_matrix.csv (euros, con V_hora = €{hourly_value:.2f}/h)")

    return dist_matrix, time_matrix, toll_matrix, cost_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construye las matrices de costos para el TSP de España")
    parser.add_argument("--hourly-value", type=float, default=20.0, help="Valor de la hora del vendedor en €/h")
    args = parser.parse_args()
    build_matrices(hourly_value=args.hourly_value)

"""
Matrices generadas para 47 capitales:
  - data/dist_matrix.csv (km)
  - data/time_matrix.csv (horas)
  - data/toll_matrix.csv (euros)
  - data/cost_matrix.csv (euros, con V_hora = €20.00/h)
"""