# Cacería de la Alucinación 🎯 — Aportes de Persona 2

**Integrante:** Sebastián Ocampo Galvis (Persona 2)
**Curso:** Redes Neuronales y Algoritmos Bioinspirados — Semestre 2026-02  
**Categoría principal:** Código (solución subóptima, con cambios no justificados)

---

## Hallazgo 1: Corrección sobredimensionada de un error en la malla de contorno de PySwarms (animación de PSO sobre Rosenbrock)

**Tipo de hallazgo:** el diagnóstico de la IA fue correcto, pero el arreglo propuesto modificó tres cosas cuando el error se resolvía con una sola, y una de ellas no tenía relación con la falla.

### 1. Prompt utilizado (literal)
Se pegó el mensaje de error completo al asistente (GitHub Copilot):

```
2026-10-13 14:38:00,143 - matplotlib.animation - INFO - Animation.save using <class 'matplotlib.animation.PillowWriter'>
Traceback (most recent call last):
  File "<frozen runpy>", line 203, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Users\Usuario\Documents\GitHub\RedesNeuronales_Trabajo1\src\parte1\main.py", line 172, in <module>
    crear_animacion_pso_rosenbrock(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        objective=rosenbrock,
        ^^^^^^^^^^^^^^^^^^^^^
        objective_wrapper=pyswarm_wrapper,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        seed=BASE_SEED,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\Usuario\Documents\GitHub\RedesNeuronales_Trabajo1\src\parte1\visualizacion.py", line 121, in crear_animacion_pso_rosenbrock
    animation = plot_contour(
        pos_history=optimizer.pos_history,
    ...<2 lines>...
        mark=(1, 1),
    )
  File "C:\Users\Usuario\Documents\GitHub\RedesNeuronales_Trabajo1\.venv\Lib\site-packages\pyswarms\utils\plotters\plotters.py", line 221, in plot_contour
    (xx, yy, zz) = _mesh(mesher, n_processes=n_processes)
                   ~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Usuario\Documents\GitHub\RedesNeuronales_Trabajo1\.venv\Lib\site-packages\pyswarms\utils\plotters\plotters.py", line 431, in _mesh
    zz = z.reshape(xx.shape)
ValueError: cannot reshape array of size 1 into shape (4000,4000)
```

### 2. Parte relevante de la respuesta de la IA
> *"El error se debía a la función usada para construir la malla de la animación de PSO. `PySwarms` no llama a la función objetivo con un solo vector, sino con una matriz de puntos de forma (n_puntos, 2). La función `rosenbrock` del proyecto está diseñada para recibir un solo punto, por lo que devolvía un único escalar."*

La IA propuso tres cambios en `visualizacion.py`:

```python
# (a) Nueva función vectorizada que reescribe la fórmula de Rosenbrock
def _rosenbrock_mesh(points):
    points = np.asarray(points, dtype=float)
    x = points[:, 0]
    y = points[:, 1]
    return 100 * (y - x**2)**2 + (1 - x)**2

# (b) Mesher con la nueva función y delta = 0.01 (antes 0.001)
mesher = Mesher(
    func=_rosenbrock_mesh,
    delta=0.01,
    limits=limits,
    levels=np.logspace(-1, 3.5, 25),
)

# (c) Cambio del escritor del GIF "para no depender de ImageMagick en Windows"
animation.save(nombre_gif, writer=PillowWriter(fps=10))
```

### 3. Cómo sospechamos que había un error
- El proyecto ya contaba con `pyswarm_wrapper`, que adapta la función objetivo a la interfaz de PySwarms. La IA lo ignoró y duplicó la fórmula de Rosenbrock en otro archivo (`_rosenbrock_mesh`).
- El traceback **no pasa por `animation.save`**: la falla ocurre dentro de `plot_contour` → `_mesh` → `z.reshape(xx.shape)`, es decir, al construir la malla, antes de guardar nada. Por lo tanto, cambiar el escritor del GIF no podía corregir ese error.
- La primera línea del log muestra que, en esa misma ejecución, `Animation.save` ya había usado `PillowWriter` antes del fallo, por lo que la dependencia de ImageMagick no era un problema presente.

### 4. Evidencia que lo demuestra
- **Traceback:** la pila termina en `pyswarms/utils/plotters/plotters.py`, línea 431 (`zz = z.reshape(xx.shape)`), sin llegar a `animation.save`.
- **Aritmética de la malla:** el dominio es `[-2, 2] × [-1, 3]` y el `delta` por defecto del `Mesher` es 0.001, de modo que cada eje tiene 4/0.001 = 4000 puntos y la malla tiene 4000 × 4000 = 16 000 000 puntos, que coincide con la forma del error. En este punto el diagnóstico de la IA fue correcto.
- **Corrección mínima:** al pasar al `Mesher` la función envuelta, `func=lambda X: objective_wrapper(X, objective)`, el error desapareció sin tocar `delta` ni el escritor, lo que demuestra que los cambios (a) y (c) no eran necesarios para resolverlo.
- **Verificaciones adicionales** (script abajo):

| Verificación | Resultado |
|---|---|
| Máx. diferencia absoluta entre `_rosenbrock_mesh` y `rosenbrock` en 1000 puntos aleatorios | [COMPLETAR] |
| Tiempo de evaluación punto a punto (1000 puntos) | [COMPLETAR] |
| Tiempo de evaluación vectorizada (1000 puntos) | [COMPLETAR] |

```python
import time
import numpy as np
from src.parte1.funciones import rosenbrock          # ajustar al proyecto
from src.parte1.visualizacion import _rosenbrock_mesh

rng = np.random.default_rng(0)
pts = rng.uniform([-2, -1], [2, 3], size=(1000, 2))

ref = np.array([rosenbrock(p) for p in pts])
print("max |diferencia|:", np.max(np.abs(ref - _rosenbrock_mesh(pts))))

t0 = time.perf_counter(); [rosenbrock(p) for p in pts]; t1 = time.perf_counter()
_rosenbrock_mesh(pts); t2 = time.perf_counter()
print(f"punto a punto: {t1 - t0:.4f}s | vectorizado: {t2 - t1:.6f}s")
```

### 5. La corrección
Una sola línea, reutilizando el adaptador que ya existía en el proyecto en lugar de duplicar la fórmula:

```python
limits = [(-2, 2), (-1, 3)]
mesher = Mesher(
    func=lambda X: objective_wrapper(X, objective),  # evalúa con el wrapper del proyecto
    limits=limits,
    levels=np.logspace(-1, 3.5, 25),
)
```

*Nota:* reducir `delta` a 0.01 (160 000 puntos en lugar de 16 millones) es una mejora razonable de costo computacional, pero es una decisión independiente del error y debe justificarse como tal.

### 6. Lección aprendida
- Antes de aceptar un parche, contrastar **cada cambio** con la causa que muestra el traceback: si una modificación no está en la ruta de la falla, no es parte de la solución.
- Reutilizar las funciones del proyecto. Reescribir la fórmula de Rosenbrock en otro archivo crea el riesgo de que la versión de visualización y la del experimento diverjan sin que nadie lo note.
- Verificar que los puntos evaluados para dibujar el contorno **no se cuenten** como evaluaciones de f en la comparación justa del enunciado; de lo contrario, el conteo se inflaría con millones de evaluaciones que no pertenecen al algoritmo.