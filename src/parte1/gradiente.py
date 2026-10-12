import numpy as np


def gradient_descent(obj_func, x0, eta=0.001, max_iter=5000, tol=1e-6):
	"""Ejecuta descenso por gradiente registrando la trayectoria y los contadores."""
	obj_func.reset_counters()
	x = np.array(x0, dtype=float)
	history = [x.copy()]

	for _ in range(max_iter):
		grad = obj_func.grad(x)
		x_new = x - eta * grad

		if np.linalg.norm(x_new - x) < tol:
			x = x_new
			history.append(x.copy())
			break

		x = x_new
		history.append(x.copy())

	f_final = obj_func.f(x)

	return {
		"x_opt": x,
		"f_opt": f_final,
		"evals_f": obj_func.evals_f,
		"evals_grad": obj_func.evals_grad,
		"history": np.array(history),
	}
