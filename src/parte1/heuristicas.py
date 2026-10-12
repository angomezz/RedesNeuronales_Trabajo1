import numpy as np
from scipy.optimize import differential_evolution


def run_algortimo_evolutivo(obj_func, dim, pop_size=50, generations=100, seed=None):
	"""
	Algoritmo Evolutivo Continuo desarrollado paso a paso.

	obj_func: Objeto clase ObjectiveFunction que cuenta evaluaciones.
	dim: Dimensión del problema (2D o 3D).
	pop_size: Tamaño de la población (N).
	generations: Número de iteraciones o generaciones.
	"""
	if seed is not None:
		np.random.seed(seed)

	obj_func.reset_counters()
	low, high = obj_func.domain

	# 1. POBLACIÓN INICIAL
	pop = np.random.uniform(low, high, size=(pop_size, dim))
	best_x = None
	best_f = float("inf")

	for _ in range(generations):
		# 2. EVALUACIÓN Y RANKING
		fitness = np.array([obj_func.f(ind) for ind in pop])

		# Guardar la mejor solución global encontrada
		min_idx = np.argmin(fitness)
		if fitness[min_idx] < best_f:
			best_f = fitness[min_idx]
			best_x = pop[min_idx].copy()

		# Ordenar población de menor a mayor (minimización)
		order = np.argsort(fitness)

		# 3. ELITISMO (Top 10% pasa directo)
		elite_count = max(1, int(0.10 * pop_size))
		elite = pop[order[:elite_count]].copy()

		# 4. GENERACIÓN DE HIJOS (Cruzamiento + Mutación)
		offspring = []
		while len(offspring) < pop_size - elite_count:
			# Selección de dos padres al azar de la élite
			parent1 = elite[np.random.choice(elite_count)]
			parent2 = elite[np.random.choice(elite_count)]

			# Cruzamiento de un punto
			cut_point = np.random.randint(1, dim) if dim > 1 else 1
			child = np.concatenate([parent1[:cut_point], parent2[cut_point:]])

			# Mutación gaussiana con probabilidad del 20%
			if np.random.rand() < 0.20:
				mut_coord = np.random.randint(0, dim)
				noise = np.random.normal(0, 0.1 * (high - low))
				child[mut_coord] += noise

			# Control de bordes (clipping)
			child = np.clip(child, low, high)
			offspring.append(child)

		# Unir la élite con la nueva descendencia
		pop = np.vstack((elite, np.array(offspring)))

	return {
		"x_opt": best_x,
		"f_opt": best_f,
		"evals_f": obj_func.evals_f,
		"evals_grad": obj_func.evals_grad,
	}


def run_ecolucion_diferencial(obj_func, dim, popsize=15, maxiter=200, seed=None):
	"""
	Evolución Diferencial (DE) utilizando SciPy, conectada al contador explícito.
	DE destaca especialmente en valles estrechos como Rosenbrock.
	"""
	obj_func.reset_counters()
	bounds = [obj_func.domain] * dim
	result = differential_evolution(
		func=obj_func.f,
		bounds=bounds,
		maxiter=maxiter,
		popsize=popsize,
		seed=seed,
		tol=1e-6,
	)
	return {
		"x_opt": result.x,
		"f_opt": result.fun,
		"evals_f": obj_func.evals_f,
		"evals_grad": obj_func.evals_grad,
	}
