# Cacería de la Alucinación 🎯 — Aportes de Persona 5 (P5)

**Integrante:** Juan José Baena Garcia (Persona 5)  
**Curso:** Redes Neuronales y Algoritmos Bioinspirados — Semestre 2026-02  
**Categoría principal:** Código y Conceptos (Algoritmos Bioinspirados / ACO)  
**Categoría complementaria:** Datos del Mundo Real (Red de Peajes en España)

---

## Hallazgo 1 (Principal): Inversión del factor de persistencia y evaporación en ACO

### 1. Prompt utilizado (literal)
> *"Escribe una función en Python para actualizar la matriz de feromonas en un algoritmo de Colonias de Hormigas (Ant System) para el TSP, aplicando la evaporación y el depósito de todas las hormigas con los parámetros rho y Q."*

### 2. Parte relevante de la respuesta de la IA
```python
def update_pheromones(pheromone_matrix, tours, costs, rho=0.1, Q=100.0):
    # Paso 1: Evaporación de feromona
    # Se multiplica por la tasa de evaporación rho para reducir el rastro
    pheromone_matrix = rho * pheromone_matrix
    
    # Paso 2: Depósito de feromona de cada hormiga
    for tour, cost in zip(tours, costs):
        for i in range(len(tour)):
            u, v = tour[i], tour[(i + 1) % len(tour)]
            pheromone_matrix[u][v] += Q / cost
            pheromone_matrix[v][u] += Q / cost
            
    return pheromone_matrix
```

### 3. Cómo sospechamos que había un error
Al ejecutar las pruebas iniciales con $\rho = 0.1$ y $T = 50$ iteraciones, observamos un comportamiento anómalo: en lugar de mostrar una curva de convergencia descendente donde las hormigas aprenden progresivamente de los mejores recorridos, la media y el mejor costo fluctuaban erráticamente en cada iteración. A partir de la iteración 4, el algoritmo olvidaba los caminos promisorios encontrados previamente y se comportaba como un muestreo aleatorio sin memoria colectiva.

### 4. Evidencia que lo demuestra
- **Fuente primaria y literatura canónica:** En el libro fundacional de Dorigo & Stützle (2004), *Ant Colony Optimization* (MIT Press, Cambridge, MA, p. 38, Eq. 2.2 y p. 67), la regla canónica de actualización de feromonas se define formalmente como:
  $$\tau_{ij}(t+1) = (1 - \rho)\tau_{ij}(t) + \sum_{k=1}^m \Delta \tau_{ij}^k(t)$$
  donde $\rho \in (0, 1]$ es el **coeficiente de evaporación de feromonas**, y por lo tanto, el factor $(1 - \rho)$ representa la **persistencia o retención de la memoria** del rastro.
- **Evidencia experimental:** Al evaluar la función sugerida con $\rho = 0.1$, en cada ciclo se retenía únicamente el $10\%$ de la feromona ($\rho = 0.1$) y se evaporaba el $90\%$. Tras 3 iteraciones, la feromona residual acumulada de un arco era $(0.1)^3 = 0.001$, borrando el $99.9\%$ del aprendizaje. Al corregir a $(1 - \rho) = 0.9$, la retención fue del $90\%$ y la tasa de evaporación real del $10\%$, permitiendo una convergencia monótona decreciente estable.

### 5. La corrección
```python
def update_pheromones(pheromone_matrix, tours, costs, rho=0.1, Q=100.0):
    # CORRECCIÓN: Persistencia de feromona con factor (1 - rho)
    # rho es la fracción que se pierde; (1 - rho) es la fracción que se retiene
    pheromone_matrix = (1.0 - rho) * pheromone_matrix
    
    # Depósito simétrico de las hormigas
    for tour, cost in zip(tours, costs):
        deposit = Q / cost
        n = len(tour)
        for i in range(n):
            u, v = tour[i], tour[(i + 1) % n]
            pheromone_matrix[u][v] += deposit
            pheromone_matrix[v][u] += deposit
            
    return pheromone_matrix
```

### 6. Lección aprendida
Las inteligencias artificiales frecuentemente confunden si un hiperparámetro representa la tasa de pérdida o el factor de retención (un fenómeno idéntico al que ocurre entre la formulación de `dropout` en PyTorch vs la probabilidad de retención $p$ en el paper original de Srivastava et al., 2014). **Toda ecuación de actualización bioinspirada debe validarse matemáticamente contra el paper original antes de aceptarla en código.**

---

## Hallazgo 2 (Complementario): Alucinación de peajes desactualizados en España

### 1. Prompt utilizado (literal)
> *"¿Cuáles son las autopistas de peaje entre las capitales de Cataluña, Valencia y Aragón para incluirlas en la matriz de costos de un TSP en España?"*

### 2. Parte relevante de la respuesta de la IA
> *"Para viajar entre Barcelona, Tarragona y Valencia debes pagar peaje obligatorio en la autopista AP-7 con un costo aproximado de 28.50 €. De igual forma, el trayecto entre Zaragoza y Tarragona por la autopista AP-2 tiene un peaje de aproximadamente 22.00 € administrado por Abertis."*

### 3. Cómo sospechamos que había un error
Recordamos la advertencia del enunciado: *"Tengan en cuenta que la red de autopistas de peaje en España ha cambiado mucho en los últimos años, por lo que la información desactualizada es un riesgo real."*

### 4. Evidencia que lo demuestra
- **Fuente primaria oficial:** Notas de prensa oficiales del Ministerio de Transportes, Movilidad y Agenda Urbana (MITMA), Gobierno de España:
  - La concesión de la AP-7 entre Alicante y Tarragona venció el **31 de diciembre de 2019**, pasando a ser libre y gratuita a partir del **1 de enero de 2020** (Real Decreto 1052/2019).
  - La concesión de la autopista AP-2 (Zaragoza - El Vendrell) y la AP-7 (Tarragona - La Jonquera) finalizó el **31 de agosto de 2021**, quedando completamente liberadas de peaje a partir del **1 de septiembre de 2021** (Real Decreto 728/2021).
- La IA alucinó peajes vigentes basándose en información de su corpus de entrenamiento anterior a 2020.

### 5. La corrección
En la matriz `data/toll_matrix.csv`, los arcos entre Barcelona, Tarragona, Valencia y Zaragoza se fijaron en **0.00 €**, reservando peajes exclusivamente a las vías verdaderamente vigentes en 2026 (como la AP-68 Bilbao-Zaragoza, los túneles de Guadarrama en la AP-61/AP-51 y la AP-66).

### 6. Lección aprendida
Los datos del mundo real sujetos a marcos regulatorios y vencimientos contractuales no deben extraerse de modelos de lenguaje sin verificación contra fuentes primarias oficiales (boletines del Estado o ministerios de transporte).

# Hallazgo 3: Uso ambiguo del contador de evaluaciones de la función objetivo en código generado por IA

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

# Hallazgo 4: Error en la configuración de los límites de visualización en PySwarms

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