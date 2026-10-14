# Optimización por enjambre de partículas (PSO)

**Autor:** Juan Felipe Miranda Arciniegas (Persona 3)

<!-- Numeración de figuras y tablas provisional: se ajusta al armar el blog. -->

## 1. ¿Cómo funciona?

PSO (*Particle Swarm Optimization*) fue propuesto por Kennedy y Eberhart (1995) y se inspira en cómo se mueven las bandadas de pájaros o los bancos de peces. La idea es tener un grupo de $N$ "partículas" que se mueven por el espacio de búsqueda. Cada partícula $i$ tiene:

- una posición $\mathbf{x}_i$, que es una posible solución,
- una velocidad $\mathbf{v}_i$,
- la mejor posición que ha encontrado ella misma, $\mathbf{p}_i$.

Además, todo el grupo conoce la mejor posición que ha encontrado cualquier partícula, $\mathbf{g}$.

En cada paso, la partícula se mueve combinando tres cosas: seguir en la dirección en la que venía, volver hacia su mejor punto y acercarse al mejor punto del grupo. A diferencia del descenso por gradiente, PSO solo necesita evaluar $f$; no usa el gradiente.

## 2. Ecuaciones

Implementamos PSO desde cero (archivo `src/parte1/pso.py`), en la versión con peso de inercia de Shi y Eberhart (1998). En cada iteración $t$, para cada partícula $i$:

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

Después de moverse, si una partícula encontró un punto mejor que su $\mathbf{p}_i$, lo guarda, y $\mathbf{g}$ se actualiza con el mejor de todos.

Agregamos dos detalles:

- **Velocidad máxima.** La velocidad no puede pasar del 20 % del ancho del dominio. Si no, al principio las partículas pueden salir disparadas.
- **Bordes del dominio.** Si una partícula se sale del dominio, la devolvemos al borde y le ponemos velocidad cero en esa dirección. Así nunca evaluamos $f$ por fuera. Esto importa en Schwefel, porque fuera de $[-500, 500]$ la función da valores más bajos que su mínimo global, y un método que se salga puede reportar un resultado "mejor" que el óptimo sin haberlo encontrado.

## 3. Parámetros

Usamos $w = 0.7298$ y $c_1 = c_2 = 1.49618$ (Tabla 1). No los escogimos a mano: son los valores que proponen Clerc y Kennedy (2002), que analizaron con qué valores el enjambre converge sin quedarse oscilando. Ellos usan un "factor de constricción"

$$
\chi = \frac{2}{\left|2 - \varphi - \sqrt{\varphi^2 - 4\varphi}\right|}, \qquad \varphi = 4.1,
$$

que da $\chi \approx 0.7298$. Su versión es equivalente a la nuestra si se toma $w = \chi$ y $c_1 = c_2 = 2.05\,\chi \approx 1.49618$. Preferimos estos valores estándar para no ajustar el PSO a cada función y que la comparación con los otros métodos sea justa.

**Tabla 1.** *Parámetros del PSO usados en los experimentos.*

| Parámetro | Símbolo | Valor | Opción en la línea de comandos |
|---|---|---|---|
| Número de partículas | $N$ | 50 | `--particulas` |
| Iteraciones | $T$ | 100 | `--iteraciones` |
| Peso de inercia | $w$ | 0.7298 (constante) | `--w-inicial`, `--w-final` |
| Peso hacia su mejor punto | $c_1$ | 1.49618 | `--c1` |
| Peso hacia el mejor del grupo | $c_2$ | 1.49618 | `--c2` |
| Velocidad máxima | $v_{\max}$ | 20 % del ancho del dominio | `--vmax` |
| Umbral de éxito | — | $f \le 10^{-2}$ | `--umbral` |
| Semillas | — | 56, 57, …, 85 | `--semilla-base` |

*Fuente:* elaboración propia; valores por defecto de `src/parte1/p3_pso_experimentos.py`.

## 4. Conteo de evaluaciones

PSO evalúa $f$ una vez por partícula al empezar y una vez por partícula en cada iteración:

$$
\text{evaluaciones de } f = N\,(T + 1) = 50 \times 101 = 5050, \qquad \text{evaluaciones de } \nabla f = 0.
$$

El código cuenta las evaluaciones con un contador propio y revisa en cada corrida que el número coincida con esta fórmula. Para comparar con el descenso por gradiente usamos la equivalencia del enunciado: calcular $\nabla f$ por diferencias centrales en dimensión $n$ cuesta $2n$ evaluaciones de $f$.

## 5. Cómo hicimos los experimentos

Usamos los mismos dominios que el resto del equipo: Rosenbrock en $[-2, 2]^n$ y Schwefel en $[-500, 500]^n$, con $n = 2$ y $n = 3$. Para cada caso hicimos 30 corridas con semillas fijas de 56 a 85. Contamos una corrida como exitosa si terminó con $f \le 10^{-2}$. En Schwefel este umbral funciona bien, porque el siguiente mínimo local queda 118.44 por encima del global, así que no hay forma de confundirlos.

Para repetir todo:

```bash
python -m src.parte1.p3_pso_experimentos
```

## 6. Resultados

La Tabla 2 resume las 30 corridas de cada caso.

**Tabla 2.** *Resultados del PSO en 30 corridas por caso.*

| Función | $n$ | Media | Desv. est. | Mejor | Peor | Mediana | Éxito | Evals. $f$ | Evals. $\nabla f$ |
|---|---|---|---|---|---|---|---|---|---|
| Rosenbrock | 2 | $1.17 \times 10^{-7}$ | $1.48 \times 10^{-7}$ | $2.80 \times 10^{-9}$ | $5.33 \times 10^{-7}$ | $6.19 \times 10^{-8}$ | 100.0 % | 5050 | 0 |
| Rosenbrock | 3 | $2.86 \times 10^{-2}$ | $2.56 \times 10^{-2}$ | $9.18 \times 10^{-5}$ | $8.48 \times 10^{-2}$ | $2.28 \times 10^{-2}$ | 36.7 % | 5050 | 0 |
| Schwefel | 2 | 47.38 | 58.02 | $2.55 \times 10^{-5}$ | 118.44 | $2.55 \times 10^{-5}$ | 60.0 % | 5050 | 0 |
| Schwefel | 3 | 126.34 | 80.52 | $3.82 \times 10^{-5}$ | 355.32 | 118.44 | 13.3 % | 5050 | 0 |

*Fuente:* elaboración propia, `results/parte1/pso/resumen.csv`. La desviación estándar es la poblacional (divide entre el número de corridas).

La Figura 1 y la Figura 2 muestran cómo se mueve el enjambre en 2D. La Figura 3 compara, en una sola corrida, cuánto baja $f$ con cada método según el número de evaluaciones.

**Figura 1.** *Movimiento del enjambre de PSO en la función de Rosenbrock (2D).*

![PSO en Rosenbrock](../results/parte1/gifs/pso_rosenbrock.gif)

*Fuente:* elaboración propia, `python -m src.parte1.p3_gifs` ($N = 30$, $T = 60$, semilla 56).

**Figura 2.** *Movimiento del enjambre de PSO en la función de Schwefel (2D).*

![PSO en Schwefel](../results/parte1/gifs/pso_schwefel.gif)

*Fuente:* elaboración propia, `python -m src.parte1.p3_gifs` ($N = 30$, $T = 60$, semilla 56).

**Figura 3.** *Mejor valor de $f$ según el número de evaluaciones equivalentes: descenso por gradiente frente a PSO (2D).*

![Convergencia GD vs PSO](../results/parte1/gifs/convergencia_gd_vs_pso.png)

*Fuente:* elaboración propia, `python -m src.parte1.p3_gifs`. Cada evaluación de $\nabla f$ de GD se cuenta como $2n = 4$ evaluaciones de $f$. La parte roja es cuando GD ya está fuera del dominio.

## 7. Análisis

**Rosenbrock.** En 2D, PSO encontró el mínimo en las 30 corridas, con valores cercanos a $10^{-7}$ (Tabla 2). En 3D, con las mismas 5050 evaluaciones, solo el 36.7 % de las corridas pasó el umbral. Pero eso no significa que se perdiera: en promedio terminó a 0.27 del mínimo $(1, 1, 1)$, y la mediana de $f$ fue $2.3 \times 10^{-2}$. Lo que pasa es que el enjambre encuentra rápido el valle curvo de Rosenbrock, pero luego avanza muy lento dentro de él. El valle es muy angosto y casi plano a lo largo, y con una dimensión más y las mismas partículas no le alcanzan las evaluaciones para llegar al fondo.

**Schwefel.** Aquí pasa algo diferente. Casi todas las corridas que fallaron terminaron en un valor que es un múltiplo de 118.44: en 2D, 12 de las 30 corridas quedaron en $f = 118.44$, y en 3D hubo corridas con 118.44, 236.88 y 355.32. Hubo dos excepciones: una corrida en 3D terminó en $f = 118.61$ porque todavía estaba bajando hacia ese mismo mínimo, y otra estaba cerca del mínimo global pero con $f = 0.023$, un poco por encima del umbral.

La razón está en la forma de la función. Schwefel se puede escribir como una suma de términos, uno por cada coordenada:

$$
f(\mathbf{x}) = \sum_{j=1}^{n} \Big(418.9829 - x_j \sin\sqrt{|x_j|}\Big).
$$

Cada coordenada tiene su mínimo en $x_j \approx 420.97$, pero hay otro mínimo local en $x_j \approx -302.52$, al otro lado del dominio, que suma 118.44. Revisamos las 37 corridas que terminaron con $f > 1$, y en todas, las coordenadas que no llegaron al óptimo estaban cerca de $-302.52$ (por ejemplo, $\mathbf{x} = (420.97, -302.52)$ en 2D). Cada coordenada que se queda ahí suma 118.44. Por eso el éxito baja de 60 % en 2D a 13.3 % en 3D: con más coordenadas es más probable que alguna se quede atrapada. Esto pasa porque en esta versión de PSO todas las partículas siguen al mismo $\mathbf{g}$: si el mejor punto del grupo está en el mínimo equivocado, todo el enjambre se va hacia allá y deja de explorar, como se ve en la Figura 2.

**Comparación con descenso por gradiente.** La Figura 3 muestra una corrida de cada método en 2D. En Rosenbrock, GD necesitó 8324 iteraciones (unas 33 300 evaluaciones equivalentes de $f$) para llegar a $f \approx 1.2 \times 10^{-4}$, mientras que PSO llegó a $f \approx 2.7 \times 10^{-5}$ con 1830 evaluaciones. GD es lento aquí porque la tasa de aprendizaje tiene que ser pequeña para no salirse del valle, y con pasos tan pequeños avanza muy despacio a lo largo de él. En Schwefel, GD con tasa de aprendizaje $0.1$ se salió del dominio en pocas iteraciones (parte roja) y terminó en un punto que no es válido. Y aunque se quedara dentro, GD solo puede bajar al mínimo local más cercano a donde empezó, porque Schwefel tiene muchísimos mínimos locales.

## 8. Limitaciones

No ajustamos los parámetros del PSO para cada función; usamos los valores estándar para no favorecer a ningún método. Probablemente un enjambre más grande, o una versión en la que cada partícula solo vea a sus vecinas (*lbest*), atraparía menos veces el mínimo equivocado en Schwefel, pero gastando más evaluaciones. Además, la Figura 3 es una sola corrida y sirve para ilustrar; las conclusiones con números salen de las 30 corridas de la Tabla 2.

## Referencias

<!-- Verificadas el 13/10/2026: las tres existen y los datos coinciden. -->

Clerc, M., & Kennedy, J. (2002). The particle swarm – Explosion, stability, and convergence in a multidimensional complex space. *IEEE Transactions on Evolutionary Computation, 6*(1), 58–73. https://doi.org/10.1109/4235.985692

Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. En *Proceedings of ICNN'95 – International Conference on Neural Networks* (Vol. 4, pp. 1942–1948). IEEE. https://doi.org/10.1109/ICNN.1995.488968

Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer. En *1998 IEEE International Conference on Evolutionary Computation Proceedings* (pp. 69–73). IEEE. https://doi.org/10.1109/ICEC.1998.699146
