# Cacería de la Alucinación 🎯 — Aportes de Persona 1

**Integrante:** Tomas Gomez Cardona (Persona 1)
**Curso:** Redes Neuronales y Algoritmos Bioinspirados — Semestre 2026-02  
**Categoría principal:** Código (solución subóptima, con cambios no justificados)

---

# Hallazgo 1: Uso ambiguo del contador de evaluaciones de la función objetivo en código generado por IA

### 1. Prompt utilizado (literal)

> "Listo, entonces según lo que tengo que hacer en la parte 1, ¿qué sigue? Muéstrame el paso a paso del siguiente movimiento junto con una recomendación de código tuya".

Como contexto, se proporcionó el código Python desarrollado hasta ese momento, que implementaba las funciones de prueba de Rosenbrock y Schwefel, el descenso por gradiente y dos contadores globales: `contador_f` para las evaluaciones de la función objetivo y `contador_grad` para las evaluaciones del gradiente.

### 2. Parte relevante de la respuesta de la IA

La IA presentó una recomendación de código para implementar PSO con PySwarms y afirmó que permitiría contabilizar correctamente las llamadas a la función objetivo. En el código propuesto, modificó la función Schwefel de la siguiente manera:

```python
def schwefel(x):
    global contador_f
    x = np.asarray(x, dtype=float)
    contador_f += 1
    n = len(x)
    return 418.9829 * n - np.sum(
        x * np.sin(np.sqrt(np.abs(x)))
    )
```

Sin embargo, en el código original ya se utilizaba el mismo contador global dentro de la función Rosenbrock:

```python
def rosenbrock(x):
    global contador_f
    contador_f += 1

    x = np.asarray(x, dtype=float)
    return np.sum(
        100 * (x[1:] - x[:-1]**2)**2
        + (1 - x[:-1])**2
    )
```

De esta manera, ambas funciones quedaron asociadas a la misma variable global, `contador_f`, sin que la respuesta aclarara expresamente el alcance de ese contador ni la necesidad de reiniciarlo al cambiar de función o experimento.

### 3. Cómo sospechamos que había un error

Al revisar el código generado, identificamos que el contador de evaluaciones de Rosenbrock se reutilizó en Schwefel. Esto generó dudas sobre la validez de los resultados que se obtendrían al comparar ambas funciones de prueba.

El problema aparece cuando se ejecutan experimentos consecutivos sin reiniciar el contador: las evaluaciones de ambas funciones se acumulan en una única variable. Por tanto, el valor resultante no permite identificar cuántas evaluaciones correspondieron individualmente a Rosenbrock y cuántas a Schwefel.

Aunque el ejemplo propuesto por la IA reiniciaba `contador_f` antes de ejecutar PSO sobre Schwefel —lo que permite contabilizar esa ejecución de forma aislada—, la respuesta afirmó que el procedimiento contabilizaba correctamente las evaluaciones sin advertir esta limitación.

### 4. Evidencia que lo demuestra

La evidencia se encuentra en la comparación directa entre el código original y el código generado por la IA:

- **Código original:** la función `rosenbrock` incrementaba `contador_f` cada vez que se evaluaba. La función `schwefel`, en cambio, no incrementaba ningún contador.
- **Código generado por la IA:** se añadió `global contador_f` y `contador_f += 1` a la función `schwefel`, reutilizando la misma variable que ya contabilizaba las evaluaciones de Rosenbrock.
- **Consecuencia:** si ambas funciones se ejecutan dentro de una misma secuencia de experimentos sin reiniciar el contador, este registra la suma de sus evaluaciones, en lugar de proporcionar un conteo independiente para cada función.

Por lo tanto, el hallazgo corresponde a una deficiencia en la generación y explicación del código: la IA presentó la solución como una contabilización correcta sin precisar que el contador compartido debía gestionarse según el objetivo del experimento.

### 5. La corrección

Para registrar las evaluaciones de cada función de manera independiente, se recomienda utilizar contadores separados:

```python
contador_f_rosenbrock = 0
contador_f_schwefel = 0
```

Dentro de cada función, se debe incrementar únicamente el contador que le corresponde:

```python
def rosenbrock(x):
    global contador_f_rosenbrock
    contador_f_rosenbrock += 1

    x = np.asarray(x, dtype=float)
    return np.sum(
        100 * (x[1:] - x[:-1]**2)**2
        + (1 - x[:-1])**2
    )


def schwefel(x):
    global contador_f_schwefel
    contador_f_schwefel += 1

    x = np.asarray(x, dtype=float)
    n = len(x)
    return 418.9829 * n - np.sum(
        x * np.sin(np.sqrt(np.abs(x)))
    )
```

De este modo, los resultados de cada experimento pueden reportarse por separado, evitando confundir el número de evaluaciones de una función con el de la otra. Para garantizar comparaciones consistentes entre algoritmos, los contadores correspondientes también deben reiniciarse al comenzar cada corrida independiente.

### 6. Lección aprendida

El código generado por una inteligencia artificial puede parecer correcto y, aun así, presentar deficiencias de implementación que afecten la interpretación de los resultados experimentales. En este caso, la reutilización de un contador global sin explicar sus implicaciones podía dificultar la comparación entre funciones objetivo.

Por ello, es necesario revisar no solo la sintaxis y la ejecución del código generado por IA, sino también la lógica de las variables de control y la correspondencia entre las métricas calculadas y los resultados que se pretende reportar.

# Hallazgo 2: Error en la configuración de los límites de visualización en PySwarms

### 1. Prompt utilizado (literal)

> "Tengo un problema y es que únicamente grafica el GIF desde las coordenadas -1 a 1 en x y -1 a 1 en y, y las quiero diferentes, ¿cómo puedo modificar eso?"

Se proporcionó el siguiente fragmento de código:

```python
options1 = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

bounds = (np.array([-1, -1]), np.array([2, 3]))

optimizer1 = ps.single.GlobalBestPSO(
    n_particles=50,
    dimensions=2,
    options=options1
)

cost1, pos1 = optimizer1.optimize(fx.rosenbrock, iters=100)

m1 = Mesher(func=fx.rosenbrock)

animation1 = plot_contour(
    pos_history=optimizer1.pos_history,
    mesher=m1,
    mark=(0, 0)
)
```

El problema consistía en modificar los límites del gráfico para visualizar la trayectoria de las partículas en un dominio diferente de \([-1,1]\) en ambos ejes.

### 2. Parte relevante de la respuesta de la IA

Claude identificó correctamente que la variable `bounds` estaba definida, pero no se había pasado al constructor de `GlobalBestPSO`, y que `Mesher` utilizaba por defecto límites de \([-1,1]\).

Sin embargo, al recomendar una solución, proporcionó el siguiente código:

```python
bounds = (
    np.array([-2.048, -2.048]),
    np.array([2.048, 2.048])
)

optimizer1 = ps.single.GlobalBestPSO(
    n_particles=50,
    dimensions=2,
    options=options1,
    bounds=bounds
)

cost1, pos1 = optimizer1.optimize(fx.rosenbrock, iters=100)

m1 = Mesher(
    func=fx.rosenbrock,
    limits=[
        (-2.048, -2.048),
        (2.048, 2.048)
    ],
    levels=np.logspace(-1, 3.5, 25)
)

animation1 = plot_contour(
    pos_history=optimizer1.pos_history,
    mesher=m1,
    mark=(1, 1)
)
```

El error se encontraba en la estructura asignada al parámetro `limits` de `Mesher`. La IA organizó los valores como si cada tupla representara un punto de la forma \((x,y)\), cuando cada tupla debía representar el intervalo de un eje.

### 3. Cómo sospechamos que había un error

Al implementar la solución propuesta, el código produjo un error durante la generación de la gráfica. Al consultar nuevamente a Claude, reconoció que había invertido el formato de `limits` en su respuesta anterior.

La IA explicó que la estructura correcta era:

```python
limits = [
    (xmin, xmax),
    (ymin, ymax)
]
```

En cambio, su recomendación original utilizaba:

```python
limits = [
    (xmin, ymin),
    (xmax, ymax)
]
```

Esta diferencia provocaba un problema concreto. En el ejemplo de Claude, las coordenadas se interpretaban como dos intervalos degenerados:

- Eje X: desde `-2.048` hasta `-2.048`.
- Eje Y: desde `2.048` hasta `2.048`.

Como consecuencia, no se generaban los puntos necesarios para construir la malla de la función objetivo, lo que ocasionaba el error durante la representación de los contornos.

### 4. Evidencia que lo demuestra

La documentación oficial de PySwarms confirma que el parámetro `limits` de `Mesher` define el rango de cada eje y que, por defecto, utiliza `[(-1, 1), (-1, 1)]`.

Además, el código fuente de la librería muestra que la malla se construye interpretando la primera tupla como el intervalo del eje X y la segunda como el intervalo del eje Y. Para cada intervalo, se generan puntos mediante `numpy.arange`. Si los extremos son iguales, el intervalo queda vacío.

Fuentes oficiales:

- [Documentación de PySwarms: clase Mesher](https://pyswarms.readthedocs.io/en/latest/api/pyswarms.utils.plotters.html).
- [Código fuente de PySwarms: formatters y construcción de la malla](https://pyswarms.readthedocs.io/en/latest/_modules/pyswarms.utils.plotters.formatters.html).

También se identificó una limitación adicional en la corrección posterior: `plot_contour` utiliza un objeto `Designer` para establecer los límites visibles de los ejes, cuyo valor predeterminado también es \([-1,1]\). Por tanto, corregir únicamente `Mesher.limits` no garantiza que la animación muestre los nuevos límites solicitados. Es necesario configurar también `Designer.limits`.

Esta distinción puede verificarse en el [código fuente oficial de `plot_contour`](https://pyswarms.readthedocs.io/en/development/_modules/pyswarms/utils/plotters/plotters.html).

Finalmente, la documentación de PySwarms confirma que, para restringir las posiciones exploradas por el optimizador, el parámetro `bounds` debe pasarse explícitamente al constructor de `GlobalBestPSO`, como se explica en el [ejemplo oficial de optimización con límites](https://pyswarms.readthedocs.io/en/development/examples/basic_optimization.html).

### 5. La corrección

Para conservar los límites definidos originalmente, es decir, de -1 a 2 en el eje X y de -1 a 3 en el eje Y, se debe configurar por separado el dominio de búsqueda del optimizador y los límites de la visualización.

```python
# Límites de búsqueda del optimizador
bounds = (
    np.array([-1.0, -1.0]),
    np.array([2.0, 3.0])
)

optimizer1 = ps.single.GlobalBestPSO(
    n_particles=50,
    dimensions=2,
    options=options1,
    bounds=bounds
)

cost1, pos1 = optimizer1.optimize(
    fx.rosenbrock,
    iters=100
)

# Límites de la malla utilizada para dibujar los contornos
m1 = Mesher(
    func=fx.rosenbrock,
    limits=[(-1.0, 2.0), (-1.0, 3.0)],
    levels=np.logspace(-1, 3.5, 25).tolist()
)

# Límites visibles de los ejes de la animación
d1 = Designer(
    limits=[(-1.0, 2.0), (-1.0, 3.0)],
    label=['x-axis', 'y-axis']
)

animation1 = plot_contour(
    pos_history=optimizer1.pos_history,
    mesher=m1,
    designer=d1,
    mark=(1, 1)
)

animation1.save(
    'plot02.gif',
    writer='imagemagick',
    fps=10
)

Image(url='plot02.gif')
```

Con esta corrección, el optimizador queda restringido al dominio indicado, la malla se construye con intervalos válidos y los ejes de la animación utilizan los límites solicitados. El punto `(1, 1)` se conserva como referencia del mínimo global de Rosenbrock en dos dimensiones.

### 6. Lección aprendida

Este hallazgo evidencia que una inteligencia artificial puede identificar correctamente las causas generales de un problema y, aun así, generar código incorrecto al aplicar una solución concreta.

En este caso, Claude reconoció que los límites predeterminados de PySwarms explicaban el comportamiento de la gráfica, pero invirtió la estructura del parámetro `limits` en el código recomendado. Aunque posteriormente corrigió ese error, su explicación inicial ya había llevado a una implementación fallida.

Además, la solución requería distinguir entre tres configuraciones diferentes: los límites de búsqueda del algoritmo, los límites de la malla de contornos y los límites visibles de los ejes. La experiencia demuestra la importancia de verificar las recomendaciones de la IA contra la documentación oficial y ejecutar el código generado antes de incorporarlo a un trabajo académico.