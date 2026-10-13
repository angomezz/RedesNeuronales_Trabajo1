"""
Tests del algoritmo de Colonias de Hormigas (ACO).
Ejecutar con:  python -m pytest test/ -q
"""

import itertools
import pathlib
import sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from aco_tsp import CONFIG, load_problem, nearest_neighbor, run_aco, tour_cost


def cfg_con(**cambios):
    cfg = dict(CONFIG)
    cfg.update(cambios)
    return cfg


def test_recorrido_es_permutacion_valida():
    cfg = cfg_con(n_cities=20, ants=20, iters=10)
    M, _, _ = load_problem(cfg)
    res = run_aco(M, cfg)
    tour = res["best_tour"]
    assert sorted(tour) == list(range(20))


def test_costo_no_cambia_al_rotar_el_ciclo():
    M, _, _ = load_problem(cfg_con(n_cities=12))
    t = np.random.default_rng(1).permutation(12)
    base = tour_cost(M, t)
    for k in range(12):
        assert abs(tour_cost(M, np.roll(t, k)) - base) < 1e-9


def test_costo_no_cambia_al_invertir_si_la_matriz_es_simetrica():
    M, _, _ = load_problem(cfg_con(n_cities=12))
    t = np.random.default_rng(2).permutation(12)
    assert abs(tour_cost(M, t) - tour_cost(M, t[::-1])) < 1e-9


def test_cuenta_de_evaluaciones():
    cfg = cfg_con(n_cities=15, ants=25, iters=10)
    M, _, _ = load_problem(cfg)
    res = run_aco(M, cfg)
    assert res["evaluations"] == 25 * 10


def test_misma_semilla_mismo_resultado():
    cfg = cfg_con(n_cities=20, ants=20, iters=20, seed=123)
    M, _, _ = load_problem(cfg)
    res1 = run_aco(M, cfg)
    res2 = run_aco(M, cfg)
    assert abs(res1["best_cost"] - res2["best_cost"]) < 1e-12
    assert res1["best_tour"] == res2["best_tour"]


def test_coincide_con_fuerza_bruta_en_8_ciudades():
    cfg = cfg_con(n_cities=8, ants=15, iters=30, seed=42)
    M, _, _ = load_problem(cfg)
    optimo = min(tour_cost(M, np.array((0,) + p)) for p in itertools.permutations(range(1, 8)))
    res = run_aco(M, cfg)
    assert abs(res["best_cost"] - optimo) < 1e-9


def test_supera_al_vecino_mas_cercano_en_40_ciudades():
    cfg = cfg_con(seed=56, n_cities=40, ants=40, iters=60)
    M, _, _ = load_problem(cfg)
    res = run_aco(M, cfg)
    assert res["best_cost"] < tour_cost(M, nearest_neighbor(M))
