"""
GIF animados del descenso por gradiente (GD) y del PSO propio, y figura de
convergencia GD vs. PSO en evaluaciones equivalentes (Persona 3).

Reutiliza sin modificar el código del equipo:
    - funciones.py   (Rosenbrock, Schwefel y sus gradientes analíticos)
    - gradiente.py   (grad_desc)
    - visualizacion.py (rosen_xy, schwefel_xy para las curvas de nivel)

Equivalencia de costo: una evaluación de ∇f en dimensión n cuenta como 2n
evaluaciones de f (diferencias centrales). En 2D, cada paso de GD cuesta 4.

Uso (desde la raíz del repositorio):
    python -m src.parte1.p3_gifs                         # los 4 GIF + figura
    python -m src.parte1.p3_gifs --funcion schwefel --metodo pso --semilla 7
    python -m src.parte1.p3_gifs --gd-lr-schwefel 0.05 --particulas 30
"""

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

from . import funciones
from .gradiente import grad_desc
from .pso import pso
from .visualizacion import rosen_xy, schwefel_xy

PROBLEMAS = {
    "rosenbrock": {
        "f": funciones.rosenbrock, "grad": funciones.grad_rosenbrock, "f_xy": rosen_xy,
        "dominio": (-2.0, 2.0), "x_opt": (1.0, 1.0), "log": True, "margen": 0.0,
        "titulo": "Rosenbrock",
    },
    "schwefel": {
        "f": funciones.schwefel, "grad": funciones.grad_schwefel, "f_xy": schwefel_xy,
        "dominio": (-500.0, 500.0), "x_opt": (420.9687, 420.9687), "log": False, "margen": 0.2,
        "titulo": "Schwefel",
    },
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="GIF de GD y PSO + figura de convergencia (P3)")
    p.add_argument("--funcion", choices=["rosenbrock", "schwefel", "ambas"], default="ambas")
    p.add_argument("--metodo", choices=["gd", "pso", "ambos"], default="ambos")
    p.add_argument("--semilla", type=int, default=56)
    p.add_argument("--gd-lr-rosenbrock", type=float, default=0.001)
    p.add_argument("--gd-lr-schwefel", type=float, default=0.1)
    p.add_argument("--gd-iter", type=int, default=20000)
    p.add_argument("--gd-tol", type=float, default=1e-5)
    p.add_argument("--particulas", type=int, default=30)
    p.add_argument("--iteraciones", type=int, default=60)
    p.add_argument("--w-inicial", type=float, default=0.7298)
    p.add_argument("--w-final", type=float, default=0.7298)
    p.add_argument("--c1", type=float, default=1.49618)
    p.add_argument("--c2", type=float, default=1.49618)
    p.add_argument("--max-frames", type=int, default=120)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--salida", type=str, default="results/parte1/gifs")
    return p.parse_args(argv)


def _barra(fig, ax, cs, prob):
    """Barra de color; en escala log usa potencias de 10 legibles."""
    if prob["log"]:
        return fig.colorbar(cs, ax=ax, label="$f(x_1, x_2)$", ticks=[1e-3, 1e-1, 1e1, 1e3])
    return fig.colorbar(cs, ax=ax, label="$f(x_1, x_2)$")


def _fondo(ax, prob):
    """Dibuja las curvas de nivel, el dominio y el mínimo global."""
    lo, hi = prob["dominio"]
    m = prob["margen"] * (hi - lo)
    vis_lo, vis_hi = lo - m, hi + m
    X, Y = np.meshgrid(np.linspace(vis_lo, vis_hi, 300), np.linspace(vis_lo, vis_hi, 300))
    Z = prob["f_xy"](X, Y)
    if prob["log"]:
        cs = ax.contourf(X, Y, Z + 1e-3, levels=np.logspace(-3, np.log10(Z.max()), 30),
                         norm=LogNorm(), cmap="viridis", alpha=0.85)
    else:
        cs = ax.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.85)
    if m > 0:
        ax.add_patch(Rectangle((lo, lo), hi - lo, hi - lo, fill=False, ls="--",
                               lw=1.2, ec="white", label="Dominio"))
    ax.plot(*prob["x_opt"], marker="*", color="red", ms=15, ls="none", label="Mínimo global")
    ax.set_xlim(vis_lo, vis_hi)
    ax.set_ylim(vis_lo, vis_hi)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    return cs


def correr_gd(prob, args):
    lo, hi = prob["dominio"]
    x0 = np.random.default_rng(args.semilla).uniform(lo, hi, 2)
    lr = args.gd_lr_rosenbrock if prob["titulo"] == "Rosenbrock" else args.gd_lr_schwefel
    tray = np.asarray(grad_desc(x0, grad_fun=prob["grad"], max_iter=args.gd_iter,
                                learning_rate=lr, tolerancia=args.gd_tol))
    return tray, lr


def gif_gd(prob, args, salida):
    lo, hi = prob["dominio"]
    tray, lr = correr_gd(prob, args)
    idx = np.unique(np.linspace(0, len(tray) - 1, min(args.max_frames, len(tray))).astype(int))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _barra(fig, ax, _fondo(ax, prob), prob)
    ax.plot(*tray[0], "s", color="white", mec="black", ms=8, label="Inicio")
    linea, = ax.plot([], [], "-", color="white", lw=1.5)
    punto, = ax.plot([], [], "o", color="orange", mec="black", ms=7, label="GD")
    ax.legend(loc="lower left", fontsize=8)

    def update(k):
        i = idx[k]
        linea.set_data(tray[: i + 1, 0], tray[: i + 1, 1])
        punto.set_data([tray[i, 0]], [tray[i, 1]])
        fuera = " (fuera del dominio)" if np.any((tray[i] < lo) | (tray[i] > hi)) else ""
        ax.set_title(f"{prob['titulo']} – GD (η={lr:g})\niteración {i}, "
                     f"evals ∇f={i}, f={prob['f_xy'](*tray[i]):.4g}{fuera}", fontsize=10)
        return linea, punto

    ruta = salida / f"gd_{prob['titulo'].lower()}.gif"
    FuncAnimation(fig, update, frames=len(idx)).save(ruta, writer=PillowWriter(fps=args.fps), dpi=80)
    plt.close(fig)
    return ruta, tray


def gif_pso(prob, args, salida):
    lo, hi = prob["dominio"]
    res = pso(prob["f"], [lo, lo], [hi, hi], n_particulas=args.particulas,
              iteraciones=args.iteraciones, w_inicial=args.w_inicial, w_final=args.w_final,
              c1=args.c1, c2=args.c2, semilla=args.semilla, guardar_historia=True)
    H, G = res["historia_pos"], res["historia_gbest"]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _barra(fig, ax, _fondo(ax, prob), prob)
    nube = ax.scatter(H[0, :, 0], H[0, :, 1], s=22, c="white", edgecolors="black", label="Partículas")
    mejor, = ax.plot([], [], "D", color="orange", mec="black", ms=8, label="Mejor global $g$")
    ax.legend(loc="lower left", fontsize=8)

    def update(t):
        nube.set_offsets(H[t])
        mejor.set_data([G[t, 0]], [G[t, 1]])
        ax.set_title(f"{prob['titulo']} – PSO (N={args.particulas})\niteración {t}, "
                     f"evals f={res['evals_curva'][t]}, mejor f={res['curva'][t]:.4g}", fontsize=10)
        return nube, mejor

    ruta = salida / f"pso_{prob['titulo'].lower()}.gif"
    FuncAnimation(fig, update, frames=len(H)).save(ruta, writer=PillowWriter(fps=args.fps), dpi=80)
    plt.close(fig)
    return ruta, res


def figura_convergencia(datos, salida):
    """Mejor f alcanzado vs. evaluaciones equivalentes de f (∇f = 2n evals)."""
    fig, axes = plt.subplots(1, len(datos), figsize=(6 * len(datos), 4.5), squeeze=False)
    for ax, (nombre, (prob, tray, res)) in zip(axes[0], datos.items()):
        n = 2
        # GD no evalúa f; se calcula aquí solo para graficar (no cuenta como evaluación).
        f_gd = np.array([prob["f_xy"](*p) for p in tray])
        evals_gd = 2 * n * np.arange(len(tray))
        lo, hi = prob["dominio"]
        fuera = np.any((tray < lo) | (tray > hi), axis=1)
        mejor_gd = np.minimum.accumulate(f_gd)
        ax.plot(evals_gd, mejor_gd, label="GD (1 ∇f = 2n evals de f)")
        if fuera.any():
            k = int(np.argmax(fuera))
            ax.plot(evals_gd[k:], mejor_gd[k:], color="red", lw=2.5,
                    label="GD fuera del dominio (no válido)")
            ax.axvline(evals_gd[k], color="red", ls=":", lw=1)
        ax.plot(res["evals_curva"], res["curva"], label="PSO")
        if prob["log"]:
            ax.set_yscale("log")
        else:
            ax.set_yscale("symlog", linthresh=1e-2)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlim(left=0)
        ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())
        ax.set_xlabel("Evaluaciones equivalentes de $f$")
        ax.set_ylabel("Mejor $f$ alcanzado")
        ax.set_title(prob["titulo"] + " 2D")
        ax.grid(True, ls=":", alpha=0.6)
        ax.legend()
    fig.tight_layout()
    ruta = salida / "convergencia_gd_vs_pso.png"
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    return ruta


def main(argv=None):
    args = parse_args(argv)
    salida = pathlib.Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    nombres = ["rosenbrock", "schwefel"] if args.funcion == "ambas" else [args.funcion]
    datos = {}
    for nombre in nombres:
        prob = PROBLEMAS[nombre]
        tray = res = None
        if args.metodo in ("gd", "ambos"):
            ruta, tray = gif_gd(prob, args, salida)
            print(f"GIF generado: {ruta}  (GD: {len(tray) - 1} iteraciones, x_final={np.round(tray[-1], 4)})")
        if args.metodo in ("pso", "ambos"):
            ruta, res = gif_pso(prob, args, salida)
            print(f"GIF generado: {ruta}  (PSO: {res['evals_f']} evals, f*={res['f_opt']:.4g})")
        if tray is not None and res is not None:
            datos[nombre] = (prob, tray, res)
    if datos:
        print(f"Figura generada: {figura_convergencia(datos, salida)}")


if __name__ == "__main__":
    main()
