import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm
import pyswarms as ps
from pyswarms.utils.plotters.formatters import Designer, Mesher
from pyswarms.utils.plotters import plot_contour


def rosen_xy(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2


def schwefel_xy(x, y):
    return 418.9829 * 2 - (
        x * np.sin(np.sqrt(np.abs(x))) + y * np.sin(np.sqrt(np.abs(y)))
    )


def animar_gd(
    trayectoria,
    f_xy,
    dominio,
    minimo,
    nombre_gif,
    titulo="GD",
    paso_frames=1,
    max_frames=150,
    log=True,
):
    """Genera y guarda un GIF de la trayectoria del descenso por gradiente."""
    tray = np.asarray(trayectoria)
    xmin, xmax, ymin, ymax = dominio

    idx = np.unique(
        np.linspace(0, len(tray) - 1, min(max_frames, len(tray))).astype(int)
    )
    tray_s = tray[idx]

    X, Y = np.meshgrid(
        np.linspace(xmin, xmax, 300),
        np.linspace(ymin, ymax, 300),
    )
    Z = f_xy(X, Y)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if log:
        niveles = np.logspace(-3, np.log10(Z.max()), 30)
        cs = ax.contourf(
            X,
            Y,
            Z + 1e-3,
            levels=niveles,
            norm=LogNorm(),
            cmap="coolwarm",
            alpha=0.8,
        )
    else:
        cs = ax.contourf(X, Y, Z, levels=30, cmap="coolwarm")

    fig.colorbar(cs, ax=ax, label="f(x, y)")
    ax.plot(*minimo, "r*", ms=14, label="Mínimo global")
    ax.plot(*tray_s[0], "gs", ms=8, label="Inicio")

    linea, = ax.plot([], [], "-", color="black", lw=1.2)
    punto, = ax.plot([], [], "o", color="black", ms=6, label="GD")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")

    def update(k):
        linea.set_data(tray_s[: k + 1, 0], tray_s[: k + 1, 1])
        punto.set_data([tray_s[k, 0]], [tray_s[k, 1]])
        ax.set_title(
            f"{titulo} - iteración {idx[k]} "
            f"f = {f_xy(tray_s[k, 0], tray_s[k, 1]):.3g}"
        )
        return linea, punto

    anim = FuncAnimation(fig, update, frames=len(tray_s))
    anim.save(nombre_gif, writer=PillowWriter(fps=15), dpi=80)
    plt.close(fig)
    return nombre_gif


def crear_animacion_pso_rosenbrock(
    objective,
    objective_wrapper,
    nombre_gif="pso_rosenbrock.gif",
    n_particles=50,
    iters=100,
    seed=None,
):
    """Genera el GIF de PSO sobre el contorno de Rosenbrock."""
    if seed is not None:
        np.random.seed(seed)

    options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
    bounds = (np.array([-2, -1]), np.array([2, 3]))
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=2,
        options=options,
        bounds=bounds,
    )
    optimizer.optimize(
        lambda X: objective_wrapper(X, objective),
        iters=iters,
        verbose=False,
    )

    limits = [(-2, 2), (-1, 3)]
    mesher = Mesher(
        func=lambda x: objective_wrapper(x, objective),
        limits=limits,
        levels=np.logspace(-1, 3.5, 25),
    )
    designer = Designer(limits=limits, label=["x", "y"])
    animation = plot_contour(
        pos_history=optimizer.pos_history,
        mesher=mesher,
        designer=designer,
        mark=(1, 1),
    )
    animation.save(nombre_gif, writer="imagemagick", fps=10)
    return nombre_gif
