"""Pruebas del PSO propio (P3). Ejecutar con:  python -m pytest test/test_pso.py -q"""
import numpy as np

from src.parte1.pso import pso


def esfera(x):
    return float(np.sum(np.asarray(x) ** 2))


def test_conteo_evaluaciones_es_N_por_T_mas_1():
    llamadas = []
    res = pso(lambda x: llamadas.append(1) or esfera(x), [-5, -5], [5, 5],
              n_particulas=20, iteraciones=15, semilla=1)
    assert res["evals_f"] == len(llamadas) == 20 * (15 + 1)
    assert res["evals_grad"] == 0


def test_todas_las_posiciones_quedan_en_el_dominio():
    res = pso(esfera, [-1, -1, -1], [1, 1, 1], n_particulas=15, iteraciones=40,
              semilla=3, guardar_historia=True)
    H = res["historia_pos"]
    assert H.min() >= -1 and H.max() <= 1


def test_misma_semilla_mismo_resultado():
    a = pso(esfera, [-5, -5], [5, 5], semilla=56)
    b = pso(esfera, [-5, -5], [5, 5], semilla=56)
    assert a["f_opt"] == b["f_opt"] and np.array_equal(a["x_opt"], b["x_opt"])


def test_curva_del_mejor_global_es_no_creciente():
    res = pso(esfera, [-5, -5], [5, 5], semilla=7)
    assert np.all(np.diff(res["curva"]) <= 0)


def test_converge_en_la_esfera():
    res = pso(esfera, [-5] * 3, [5] * 3, n_particulas=30, iteraciones=150, semilla=0)
    assert res["f_opt"] < 1e-6
