# Optimización por enjambre de partículas (PSO)

**Autor de la sección:** Juan Felipe Miranda Arciniegas (Persona 3)

<!-- Numeración de figuras y tablas provisional: se ajusta al armar el blog. -->

## 1. ¿Cómo funciona?

PSO (*Particle Swarm Optimization*) fue propuesto por Kennedy y Eberhart (1995) y se inspira en cómo se mueven las bandadas de pájaros o los bancos de peces. La idea es tener un grupo de $N$ "partículas" que se mueven por el espacio de búsqueda. Cada partícula $i$ tiene:

- una posición $\mathbf{x}_i$, que es una posible solución,
- una velocidad $\mathbf{v}_i$,
- la mejor posición que ha encontrado ella misma, $\mathbf{p}_i$.

Además, todo el grupo conoce la mejor posición que ha encontrado cualquier partícula, $\mathbf{g}$.

En cada paso, la partícula se mueve combinando tres cosas: seguir en la dirección en la que venía, volver hacia su mejor punto y acercarse al mejor punto del grupo. A diferencia del descenso por gradiente, PSO solo necesita evaluar $f$; no usa el gradiente.

## 2. Ecuaciones

Usamos la clase `GlobalBestPSO` de la librería PySwarms (Miranda, 2018), versión 1.3.0, llamada desde `src/parte1/experimentos.py`. Esta clase implementa la versión de PSO con peso de inercia de Shi y Eberhart (1998), en la que todas las partículas siguen al mismo mejor global $\mathbf{g}$. En cada iteración $t$, para cada partícula $i$:

$$
\mathbf{v}_i(t+1) = w\,\mathbf{v}_i(t) + c_1\,\mathbf{r}_1 \odot \big(\mathbf{p}_i - \mathbf{x}_i(t)\big) + c_2\,\mathbf{r}_2 \odot \big(\mathbf{g} - \mathbf{x}_i(t)\big)
$$

$$
\mathbf{x}_i(t+1) = \mathbf{x}_i(t) + \mathbf{v}_i(t+1)
$$

donde:

- $w$ es el peso de inercia (cuánto conserva de la velocidad anterior),
- $c_1$ es el peso de ir hacia su propio mejor punto,
- $c_2$ es el peso de ir hacia el mejor punto del grupo,
- $\mathbf{r}_1$ y $\mathbf{r}_2$ son números aleatorios entre 0 y 1, uno por cada componente,
- $\odot$ significa multiplicar componente por componente.

Después de evaluar las nuevas posiciones, si una partícula encontró un punto mejor que su $\mathbf{p}_i$, lo guarda, y $\mathbf{g}$ se actualiza con el mejor de todos.

Revisamos el código de la librería para saber qué hace en los detalles que las ecuaciones no dicen:

- **Posiciones iniciales:** aleatorias y uniformes dentro del dominio.
- **Velocidades iniciales:** aleatorias entre 0 y 1 en cada componente.
- **Velocidad máxima:** no tiene; la velocidad no se limita.
- **Bordes del dominio:** usa la estrategia "periódica". Si una partícula se sale por un lado del dominio, vuelve a entrar por el lado contrario, como si los bordes estuvieran conectados. Así, todas las evaluaciones de $f$ ocurren dentro del dominio.
- **Números aleatorios:** usa el generador global de NumPy, que fijamos con `np.random.seed(semilla)` antes de cada corrida.

## 3. Parámetros

La Tabla 1 muestra los parámetros que usamos.

**Tabla 1.** *Parámetros del PSO usados en los experimentos.*

| Parámetro | Símbolo | Valor |
|---|---|---|
| Número de partículas | $N$ | 50 |
| Iteraciones | $T$ | 100 |
| Peso de inercia | $w$ | 0.9 |
| Peso hacia su mejor punto | $c_1$ | 0.5 |
| Peso hacia el mejor del grupo | $c_2$ | 0.3 |
| Umbral de éxito | — | $f \le 10^{-2}$ |
| Semillas | — | 56, 57, …, 85 |

*Fuente:* elaboración propia; valores de `src/parte1/main.py`.

<!-- Pendiente: confirmar con el equipo de dónde salen w = 0.9, c1 = 0.5 y c2 = 0.3 para citarlo. -->

Con estos valores, la inercia pesa mucho más que la atracción hacia los mejores puntos ($w = 0.9$ frente a $c_1 = 0.5$ y $c_2 = 0.3$). Eso hace que las partículas sigan moviéndose bastante y exploren, pero les cuesta quedarse quietas afinando cerca del mínimo. Este detalle se nota en los resultados de la sección 6.

## 4. Conteo de evaluaciones

PySwarms evalúa $f$ en todas las partículas al comienzo de cada iteración y después las mueve. La posición a la que llegan con el último movimiento ya no se evalúa. Por eso:

$$
\text{evaluaciones de } f = N \times T = 50 \times 100 = 5000, \qquad \text{evaluaciones de } \nabla f = 0.
$$

Contamos las evaluaciones con los contadores de `src/parte1/funciones.py`, que suman uno cada vez que se llama la función, y el resultado coincide con esta fórmula en todas las corridas. Para comparar con el descenso por gradiente usamos la equivalencia del enunciado: calcular $\nabla f$ por diferencias centrales en dimensión $n$ cuesta $2n$ evaluaciones de $f$.

## 5. Cómo hicimos los experimentos

Usamos los mismos dominios que en los demás métodos: Rosenbrock en $[-2, 2]^n$ y Schwefel en $[-500, 500]^n$, con $n = 2$ y $n = 3$. Para cada caso hicimos 30 corridas con semillas fijas de 56 a 85, y contamos una corrida como exitosa si terminó con $f \le 10^{-2}$.

Para repetir todos los experimentos de la Parte 1, incluido PSO:

```bash
python -m src.parte1.main --modo arnes
```

## 6. Resultados

La Tabla 2 resume las 30 corridas de cada caso.

**Tabla 2.** *Resultados del PSO en 30 corridas por caso.*

| Función | $n$ | Media | Desv. est. | Mejor | Peor | Mediana | Éxito | Evals. $f$ | Evals. $\nabla f$ |
|---|---|---|---|---|---|---|---|---|---|
| Rosenbrock | 2 | $1.38 \times 10^{-4}$ | $4.51 \times 10^{-4}$ | $4.29 \times 10^{-8}$ | $2.08 \times 10^{-3}$ | $6.58 \times 10^{-6}$ | 100.0 % | 5000 | 0 |
| Rosenbrock | 3 | $6.42 \times 10^{-2}$ | $8.60 \times 10^{-2}$ | $4.88 \times 10^{-4}$ | 0.389 | $2.62 \times 10^{-2}$ | 30.0 % | 5000 | 0 |
| Schwefel | 2 | 0.433 | 2.00 | $2.66 \times 10^{-5}$ | 11.19 | $5.48 \times 10^{-3}$ | 56.7 % | 5000 | 0 |
| Schwefel | 3 | 61.75 | 69.62 | $6.64 \times 10^{-4}$ | 217.53 | 8.12 | 6.7 % | 5000 | 0 |

*Fuente:* elaboración propia, `python -m src.parte1.main --modo arnes`. La desviación estándar es la poblacional (divide entre el número de corridas).

La Figura 1 muestra cómo se mueve el enjambre sobre la función de Rosenbrock.

**Figura 1.** *Movimiento del enjambre de PSO sobre la función de Rosenbrock (2D).*

![PSO en Rosenbrock](../pso_rosenbrock.gif)

*Fuente:* elaboración propia, `python -m src.parte1.main --modo gif` ($N = 50$, $T = 100$, semilla 56, región $[-2, 2] \times [-1, 3]$).

## 7. Análisis

**Rosenbrock.** En 2D, PSO encontró el mínimo en las 30 corridas (Tabla 2). En 3D, con el mismo número de evaluaciones, solo el 30 % pasó el umbral. Sin embargo, el enjambre no se perdió: la mediana de $f$ fue $2.6 \times 10^{-2}$ y la distancia mediana al mínimo $(1, 1, 1)$ fue 0.31. Lo que pasa es que el enjambre encuentra rápido el valle curvo de Rosenbrock, pero luego avanza muy lento dentro de él, porque el valle es muy angosto y casi plano a lo largo. Con una dimensión más y las mismas partículas no le alcanzan las evaluaciones para llegar al fondo.

**Schwefel 2D.** Aquí el éxito fue de 56.7 %, pero ese número engaña un poco. Revisando las 30 corridas, en 29 el enjambre llegó a la zona del mínimo global $(420.97, 420.97)$; solo en una corrida una coordenada quedó lejos. El problema es que muchas corridas se quedaron con valores entre 0.01 y 0.6, es decir, cerca del mínimo pero sin afinar lo suficiente para pasar el umbral. Esto encaja con los parámetros de la Tabla 1: con $w = 0.9$ las partículas conservan mucha velocidad, y con $c_1$ y $c_2$ pequeños la atracción hacia el mejor punto es débil, así que el enjambre explora bien pero le cuesta quedarse quieto en el fondo del mínimo.

**Schwefel 3D.** Este fue el caso más difícil: solo 6.7 % de éxito. A diferencia de 2D, aquí sí hay corridas atrapadas en mínimos locales: 22 de las 90 coordenadas (30 corridas por 3 coordenadas) terminaron lejos del óptimo, y 9 de ellas estaban en $x_j \approx -302.52$. Ese punto es el segundo mejor mínimo de cada coordenada. Como Schwefel es una suma de un término por coordenada,

$$
f(\mathbf{x}) = \sum_{j=1}^{n} \Big(418.9829 - x_j \sin\sqrt{|x_j|}\Big),
$$

cada coordenada que se queda en $-302.52$ le suma 118.44 al valor final. Por eso varias corridas terminaron cerca de 118.4. Con más coordenadas es más probable que alguna se quede en el mínimo equivocado, y como todas las partículas siguen al mismo $\mathbf{g}$, si el mejor punto del grupo está en el lugar equivocado, todo el enjambre se va hacia allá.

La comparación de PSO con el descenso por gradiente, los algoritmos evolutivos y la evolución diferencial se presenta en la discusión de la Parte 1.

## 8. Limitaciones

No ajustamos los parámetros del PSO para cada función. Por lo visto en Schwefel 2D, probablemente un $w$ más bajo, o uno que vaya bajando durante la corrida, ayudaría a afinar mejor al final. También es posible que un enjambre más grande se quede menos veces atrapado en Schwefel 3D, pero gastando más evaluaciones. Además, la Figura 1 es una sola corrida y sirve para ilustrar; las conclusiones con números salen de las 30 corridas de la Tabla 2.

## Referencias

<!-- Las tres referencias fueron verificadas el 13/10/2026. -->

Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. En *Proceedings of ICNN'95 – International Conference on Neural Networks* (Vol. 4, pp. 1942–1948). IEEE. https://doi.org/10.1109/ICNN.1995.488968

Miranda, L. J. V. (2018). PySwarms: A research toolkit for particle swarm optimization in Python. *Journal of Open Source Software, 3*(21), 433. https://doi.org/10.21105/joss.00433

Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer. En *1998 IEEE International Conference on Evolutionary Computation Proceedings* (pp. 69–73). IEEE. https://doi.org/10.1109/ICEC.1998.699146
