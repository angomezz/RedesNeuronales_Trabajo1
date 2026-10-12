"""Tests del algoritmo genético. Ejecutar con:  python -m pytest tests/ -q"""
import itertools
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from ga_tsp import (CONFIG, inversion_mutation, load_problem, nearest_neighbor,  # noqa: E402
                    order_crossover, run_ga, tour_cost)


def cfg_con(**cambios):
    cfg = dict(CONFIG)
    cfg.update(cambios)
    return cfg


def test_operadores_producen_permutaciones_validas():
    rng = np.random.default_rng(0)
    n = 47
    for _ in range(500):
        p1, p2 = rng.permutation(n), rng.permutation(n)
        hijo = order_crossover(rng, p1, p2)
        assert sorted(hijo.tolist()) == list(range(n))
        mut = inversion_mutation(rng, hijo)
        assert sorted(mut.tolist()) == list(range(n))


def test_costo_no_cambia_al_rotar_el_ciclo():
    M, _ = load_problem(cfg_con(n_cities=12))
    t = np.random.default_rng(1).permutation(12)
    base = tour_cost(M, t)
    for k in range(12):
        assert abs(tour_cost(M, np.roll(t, k)) - base) < 1e-9


def test_costo_no_cambia_al_invertir_si_la_matriz_es_simetrica():
    M, _ = load_problem(cfg_con(n_cities=12))
    t = np.random.default_rng(2).permutation(12)
    assert abs(tour_cost(M, t) - tour_cost(M, t[::-1])) < 1e-9


def test_coincide_con_fuerza_bruta_en_8_ciudades():
    cfg = cfg_con(n_cities=8, pop=100, gens=100)
    M, _ = load_problem(cfg)
    optimo = min(tour_cost(M, np.array((0,) + p)) for p in itertools.permutations(range(1, 8)))
    res = run_ga(M, cfg)
    assert abs(res["best_cost"] - optimo) < 1e-9


def test_misma_semilla_mismo_resultado():
    cfg = cfg_con(n_cities=20, pop=50, gens=50)
    M, _ = load_problem(cfg)
    assert run_ga(M, cfg)["best_cost"] == run_ga(M, cfg)["best_cost"]


def test_el_recorrido_final_es_valido_y_su_costo_es_el_reportado():
    cfg = cfg_con(n_cities=30, pop=80, gens=60)
    M, _ = load_problem(cfg)
    res = run_ga(M, cfg)
    t = np.array(res["best_tour"])
    assert sorted(t.tolist()) == list(range(30))
    assert abs(tour_cost(M, t) - res["best_cost"]) < 1e-9


def test_cuenta_de_evaluaciones():
    cfg = cfg_con(n_cities=15, pop=40, gens=10, elite=2)
    M, _ = load_problem(cfg)
    res = run_ga(M, cfg)
    assert res["evaluations"] == 40 + 10 * (40 - 2)


def test_supera_al_vecino_mas_cercano_en_40_ciudades():
    cfg = cfg_con()  # seed 56, 40 ciudades, igual que el notebook de clase
    M, _ = load_problem(cfg)
    res = run_ga(M, cfg)
    assert res["best_cost"] < tour_cost(M, nearest_neighbor(M))
