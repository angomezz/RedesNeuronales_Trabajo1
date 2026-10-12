"""
Paso 1 de la Parte 2 (TSP): coordenadas de las 47 capitales peninsulares.

Uso:  python scripts/01_build_cities.py
Lee data/cities.csv (id, ciudad, provincia), escribe data/coordinates.csv (id, ciudad, lat, lon)
y guarda las respuestas crudas en data/raw/.
Estilo de celdas numeradas: se puede ejecutar como script o celda por celda en VS Code/Jupyter.
"""

# %% [1] Imports y configuración
import csv
import datetime as dt
import json
import math
import pathlib
import time

import requests

CONFIG = {
    "cities_csv": pathlib.Path("data/cities.csv"),          # entrada: id, ciudad, provincia
    "coords_csv": pathlib.Path("data/coordinates.csv"),     # salida: id, ciudad, lat, lon
    "raw_dir": pathlib.Path("data/raw"),
    "geocoder_url": "https://nominatim.openstreetmap.org/search",
    "user_agent": "trabajo1-tsp-espana/0.1 (contacto: PON-AQUI-TU-CORREO)",  # Nominatim lo exige
    "pausa_s": 1.1,  # política de uso: máximo 1 petición por segundo
    # Caja aproximada de la España peninsular (para detectar homónimos en otros países)
    "lat_min": 35.9, "lat_max": 43.9,
    "lon_min": -9.6, "lon_max": 3.5,
    "dist_min_km": 10.0,  # dos capitales distintas nunca deben estar tan cerca
}

# %% [2] Utilidades
def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia en línea recta (km) sobre la esfera terrestre (R = 6371 km)."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def geocodificar(ciudad, provincia):
    """Consulta Nominatim y devuelve (lat, lon, respuesta_cruda)."""
    params = {"q": f"{ciudad}, {provincia}, España", "format": "jsonv2", "limit": 1}
    resp = requests.get(
        CONFIG["geocoder_url"], params=params,
        headers={"User-Agent": CONFIG["user_agent"]}, timeout=30,
    )
    resp.raise_for_status()
    datos = resp.json()
    if not datos:
        raise ValueError(f"Sin resultados para {ciudad} ({provincia})")
    return float(datos[0]["lat"]), float(datos[0]["lon"]), datos[0]


# %% [3] Lectura de la lista de ciudades
with open(CONFIG["cities_csv"], newline="", encoding="utf-8") as f:
    ciudades = list(csv.DictReader(f))
assert len(ciudades) == 47, f"Se esperaban 47 ciudades, hay {len(ciudades)}"
CONFIG["raw_dir"].mkdir(parents=True, exist_ok=True)

# %% [4] Geocodificación (guarda la respuesta cruda de cada ciudad)
fecha_consulta = dt.date.today().isoformat()
for c in ciudades:
    lat, lon, crudo = geocodificar(c["ciudad"], c["provincia"])
    c["lat"], c["lon"] = round(lat, 5), round(lon, 5)
    (CONFIG["raw_dir"] / f"geocode_{int(c['id']):02d}.json").write_text(
        json.dumps(crudo, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    time.sleep(CONFIG["pausa_s"])
print(f"Geocodificadas {len(ciudades)} ciudades. Fecha de consulta: {fecha_consulta}")

# %% [5] Validaciones (si falla alguna, revisar a mano esa ciudad)
problemas = []
for c in ciudades:
    lat, lon = float(c["lat"]), float(c["lon"])
    if not (CONFIG["lat_min"] <= lat <= CONFIG["lat_max"] and CONFIG["lon_min"] <= lon <= CONFIG["lon_max"]):
        problemas.append(f"Fuera de la península: {c['ciudad']} ({lat}, {lon})")

for i in range(len(ciudades)):
    for j in range(i + 1, len(ciudades)):
        d = haversine_km(float(ciudades[i]["lat"]), float(ciudades[i]["lon"]),
                         float(ciudades[j]["lat"]), float(ciudades[j]["lon"]))
        if d < CONFIG["dist_min_km"]:
            problemas.append(f"Muy cerca ({d:.1f} km): {ciudades[i]['ciudad']} y {ciudades[j]['ciudad']}")

if problemas:
    print("REVISAR:\n- " + "\n- ".join(problemas))
else:
    print("Validaciones OK: 47 ciudades dentro de la península y sin duplicados.")

# %% [6] Guardado (cities.csv no se modifica; las coordenadas van en un archivo aparte)
with open(CONFIG["coords_csv"], "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "ciudad", "lat", "lon"], extrasaction="ignore")
    w.writeheader()
    w.writerows(ciudades)
print(f"Escrito {CONFIG['coords_csv']}")
