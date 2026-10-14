"""
Arnés de corridas independientes del PSO propio (Persona 3).

Ejecuta, para cada función y cada dimensión, R corridas independientes
(semillas semilla_base, semilla_base+1, ..., semilla_base+R-1) y reporta:
media, desviación estándar, mejor, peor, mediana, tasa de éxito
(f <= umbral), distancia media al minimizador global y número de
evaluaciones de f y de ∇f.

Equivalencia de costo usada para comparar con descenso por gradiente:
una evaluación de ∇f en dimensión n equivale a 2n evaluaciones de f
(diferencias centrales). Para PSO, evals_equivalentes = evals_f.

Uso (desde la raíz del repositorio):
    python -m src.parte1.p3_pso_experimentos
    python -m src.parte1.p3_pso_experimentos --funciones schwefel --dims 2 --corridas 30 --particulas 30
    python -m src.parte1.p3_pso_experimentos --config configs/pso.json
"""

import argparse
import csv
import json
import pathlib
import time

import numpy as np

from . import funciones
from .pso import pso

# Dominios iguales a los usados por el resto del equipo en main.py,
# para que la comparación entre métodos sea sobre la misma caja.
FUNCIONES = {
    "rosenbrock": {
        "f": funciones.rosenbrock,
        "dominio": (-2.0, 2.0),
        "x_opt": lambda n: np.ones(n),
        "f_opt": 0.0,
    },
    "schwefel": {
        "f": funciones.schwefel,
        "dominio": (-500.0, 500.0),
        "x_opt": lambda n: np.full(n, 420.9687),
        "f_opt": 0.0,
    },
}

DEFAULTS = {
    "funciones": ["rosenbrock", "schwefel"],
    "dims": [2, 3],
    "corridas": 30,
    "semilla_base": 56,
    "particulas": 50,
    "iteraciones": 100,
    "w_inicial": 0.7298,
    "w_final": 0.7298,
    "c1": 1.49618,
    "c2": 1.49618,
    "vmax": 0.2,
    "umbral": 1e-2,
    "salida": "results/parte1/pso",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="PSO propio: 30 corridas por configuración (P3)")
    p.add_argument("--config", type=str, default=None,
                   help="archivo JSON con parámetros; la línea de comandos tiene prioridad")
    p.add_argument("--funciones", nargs="+", choices=list(FUNCIONES))
    p.add_argument("--dims", nargs="+", type=int)
    p.add_argument("--corridas", type=int)
    p.add_argument("--semilla-base", dest="semilla_base", type=int)
    p.add_argument("--particulas", type=int)
    p.add_argument("--iteraciones", type=int)
    p.add_argument("--w-inicial", dest="w_inicial", type=float)
    p.add_argument("--w-final", dest="w_final", type=float)
    p.add_argument("--c1", type=float)
    p.add_argument("--c2", type=float)
    p.add_argument("--vmax", type=float, help="fracción del ancho del dominio")
    p.add_argument("--umbral", type=float, help="éxito si f <= umbral")
    p.add_argument("--salida", type=str)
    args = vars(p.parse_args(argv))

    cfg = dict(DEFAULTS)
    ruta_config = args.pop("config")
    if ruta_config:
        with open(ruta_config, encoding="utf-8") as fh:
            cfg.update(json.load(fh))
    cfg.update({k: v for k, v in args.items() if v is not None})
    return cfg


def correr(cfg):
    salida = pathlib.Path(cfg["salida"])
    salida.mkdir(parents=True, exist_ok=True)
    filas_corridas, filas_resumen = [], []
    t0 = time.time()

    for nombre in cfg["funciones"]:
        info = FUNCIONES[nombre]
        for n in cfg["dims"]:
            lo, hi = info["dominio"]
            valores, distancias, evals = [], [], []
            for r in range(cfg["corridas"]):
                semilla = cfg["semilla_base"] + r
                res = pso(
                    info["f"], np.full(n, lo), np.full(n, hi),
                    n_particulas=cfg["particulas"], iteraciones=cfg["iteraciones"],
                    w_inicial=cfg["w_inicial"], w_final=cfg["w_final"],
                    c1=cfg["c1"], c2=cfg["c2"], frac_vmax=cfg["vmax"], semilla=semilla,
                )
                dist = float(np.linalg.norm(res["x_opt"] - info["x_opt"](n)))
                valores.append(res["f_opt"])
                distancias.append(dist)
                evals.append(res["evals_f"])
                filas_corridas.append({
                    "funcion": nombre, "dim": n, "corrida": r + 1, "semilla": semilla,
                    "f_opt": res["f_opt"], "dist_x_opt": dist,
                    "x_opt": " ".join(f"{v:.6f}" for v in res["x_opt"]),
                    "evals_f": res["evals_f"], "evals_grad": 0,
                })

            v = np.asarray(valores)
            filas_resumen.append({
                "funcion": nombre, "dim": n, "corridas": len(v),
                "media": v.mean(), "desv_est": v.std(ddof=1),
                "mejor": v.min(), "peor": v.max(), "mediana": float(np.median(v)),
                "tasa_exito_%": 100.0 * np.mean(v <= cfg["umbral"]),
                "dist_media_x_opt": float(np.mean(distancias)),
                "evals_f": int(np.mean(evals)), "evals_grad": 0,
                "evals_equivalentes": int(np.mean(evals)),
            })

    with open(salida / "corridas.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas_corridas[0]))
        w.writeheader()
        w.writerows(filas_corridas)
    with open(salida / "resumen.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas_resumen[0]))
        w.writeheader()
        w.writerows(filas_resumen)
    with open(salida / "config.json", "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    print(f"PSO propio | N={cfg['particulas']} T={cfg['iteraciones']} "
          f"w={cfg['w_inicial']}->{cfg['w_final']} c1={cfg['c1']} c2={cfg['c2']} "
          f"vmax={cfg['vmax']} umbral={cfg['umbral']:g} | {cfg['corridas']} corridas")
    print(f"{'función':<11}{'n':>2} {'media':>11} {'desv':>11} {'mejor':>11} {'peor':>11} "
          f"{'éxito':>7} {'evals f':>8} {'∇f':>4}")
    for s in filas_resumen:
        print(f"{s['funcion']:<11}{s['dim']:>2} {s['media']:>11.4g} {s['desv_est']:>11.4g} "
              f"{s['mejor']:>11.4g} {s['peor']:>11.4g} {s['tasa_exito_%']:>6.1f}% "
              f"{s['evals_f']:>8} {s['evals_grad']:>4}")
    print(f"Tiempo total: {time.time() - t0:.1f} s | resultados en {salida}/")
    return filas_resumen


if __name__ == "__main__":
    correr(parse_args())
