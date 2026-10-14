# Datos del TSP y algoritmo genético

**Autora de la sección:** Ana Sofía Gómez Zapata (Persona 4)

<!-- Numeración de figuras y tablas provisional: se ajusta al armar el blog. -->

## 1. Los datos del problema

El recorrido pasa por las 47 capitales de provincia de la España peninsular. Se excluyen Palma, Las Palmas de Gran Canaria y Santa Cruz de Tenerife por ser islas, y Ceuta y Melilla porque no son provincias ni tienen acceso por carretera desde la península.

Las coordenadas de cada capital se obtuvieron con el servicio Nominatim de OpenStreetMap, y las respuestas completas quedaron guardadas en `data/raw/`. A partir de ellas se construyeron cuatro matrices de 47 × 47, que están en `data/` para que los experimentos no dependan de internet. La Tabla 1 resume cómo se calculó cada componente del costo.

**Tabla 1.** *Componentes del costo de viajar entre dos capitales.*

| Componente | Cómo se obtuvo | Valor o fuente |
|---|---|---|
| Distancia $D_{ij}$ | Distancia en línea recta (Haversine, Tierra como esfera de radio 6371 km) multiplicada por 1.28 | Suposición del equipo para aproximar la distancia por carretera |
| Tiempo $T_{ij}$ | $D_{ij}$ / 105 km/h | Velocidad promedio fija, igual para todos los trayectos |
| Peajes $P_{ij}$ | 8 pares de ciudades con peaje (AP-68, AP-66, AP-6, AP-71, AP-46) | Tarifas pendientes de verificar con la fuente oficial 2026 |
| Consumo del carro | SEAT León 1.5 TSI de 130 CV, gasolina | 5.6 L/100 km WLTP (ficha técnica de SEAT y km77) |
| Precio del combustible $p$ | Gasolina 95 | 1.62 €/L (ver limitaciones) |

*Fuente:* elaboración propia; detalle completo, enlaces y fechas de consulta en `data/SOURCES.md`.

Con esto, el costo de ir de la ciudad $i$ a la $j$ es

$$
C_{ij}(V_h) = V_h\,T_{ij} + P_{ij} + D_{ij}\,\frac{5.6}{100}\,p ,
$$

donde $V_h$ es el valor de la hora del vendedor en €/h. Para los experimentos de esta sección usamos $V_h = 20$ €/h (`data/cost_matrix.csv`).

### Limitaciones de los datos

Preferimos decir con claridad qué tan cerca de la realidad están estos datos:

- **Las distancias no son por carretera.** Son una aproximación (línea recta × 1.28). Por ejemplo, Madrid–Barcelona sale en 645.6 km.
- **El tiempo es proporcional a la distancia.** Como la velocidad es la misma en todos los trayectos, el costo queda como $C_{ij} = D_{ij}\left(\frac{V_h}{105} + 0.056\,p\right) + P_{ij}$. Es decir, el valor de la hora y el precio del combustible solo cambian cuánto pesa la distancia frente a los peajes. Esto limita lo que se puede concluir en el estudio del valor de la hora.
- **Faltan peajes.** Solo hay 8 pares con peaje, y el peaje solo se cobra si el trayecto es exactamente entre esas dos ciudades. Hay autopistas que siguen cobrando y no están, como la AP-7 entre Alicante y Cartagena, cuya concesión sigue hasta 2048.
- **El precio del combustible está desactualizado.** La matriz usa 1.62 €/L, pero el precio medio de la gasolina 95 en España era de 1.857 €/L el 1 de octubre de 2026. Para corregirlo basta cambiar un parámetro y regenerar las matrices.

## 2. El algoritmo genético

Un algoritmo genético imita la evolución: se parte de una población de soluciones, se escogen las mejores como padres, se combinan para crear hijos, se les hacen pequeños cambios al azar y se repite durante muchas generaciones. El código está en `src/parte2/ga_tsp.py`.

### Representación y función objetivo

Cada individuo es un recorrido, escrito como una permutación $\pi$ de las 47 ciudades. Su costo es la suma de los costos de todos los trayectos, incluido el regreso a la ciudad de salida:

$$
f(\pi) = \sum_{k=0}^{N-2} C_{\pi(k),\,\pi(k+1)} + C_{\pi(N-1),\,\pi(0)} .
$$

Como cada individuo es una permutación, cada ciudad aparece exactamente una vez, y los operadores están diseñados para que eso se mantenga.

### Operadores

1. **Población inicial:** $N$ recorridos completamente al azar.
2. **Selección por torneo:** se escogen $k$ individuos al azar y gana el de menor costo. Se hace dos veces para obtener dos padres. Con $k = 3$, los recorridos buenos tienen más probabilidad de ser padres, pero los malos también pueden serlo, y eso mantiene la diversidad.
3. **Cruce de orden (OX):** con probabilidad $p_c$, el hijo copia un tramo al azar del primer padre en las mismas posiciones y llena los huecos con las ciudades que faltan, en el orden en que aparecen en el segundo padre. Así hereda pedazos de ruta de los dos y nunca repite ciudades. Si no hay cruce, el hijo es una copia del primer padre.
4. **Mutación por inversión:** con probabilidad $p_m$, se escoge un tramo del recorrido y se invierte su orden. En el TSP esto equivale a quitar dos trayectos y reconectar la ruta al revés (un movimiento 2-opt), lo que sirve para deshacer rutas que se cruzan.
5. **Elitismo:** los $e$ mejores individuos pasan sin cambios a la siguiente generación, así que el mejor costo nunca empeora.

### Conteo de evaluaciones

Cada vez que se crea un recorrido nuevo se calcula su costo una vez. Se evalúa la población inicial y luego, en cada generación, solo los $N - e$ hijos nuevos (los élite ya tienen su costo calculado):

$$
\text{evaluaciones} = N + (N - e)\,G = 300 + 298 \times 800 = 238\,700 .
$$

### Parámetros

**Tabla 2.** *Parámetros del algoritmo genético.*

| Parámetro | Símbolo | Valor | Opción en la línea de comandos |
|---|---|---|---|
| Tamaño de la población | $N$ | 300 | `--pop` |
| Generaciones | $G$ | 800 | `--gens` |
| Tamaño del torneo | $k$ | 3 | `--k` |
| Probabilidad de cruce | $p_c$ | 0.9 | `--pc` |
| Probabilidad de mutación (por individuo) | $p_m$ | 0.2 | `--pm` |
| Individuos élite | $e$ | 2 | `--elite` |
| Semillas | — | 1, 2, …, 30 | — |

*Fuente:* elaboración propia; valores de `src/parte2/ga_tsp.py` y `src/parte2/benchmark_ga_30.py`.

## 3. Experimentos

Hicimos 30 corridas independientes con las semillas 1 a 30 sobre la matriz de costos con $V_h = 20$ €/h. Como línea base usamos la heurística del vecino más cercano: salir de la primera ciudad e ir siempre a la ciudad más barata que falte. Para que la tasa de éxito sea comparable con la del ACO, usamos el mismo criterio: una corrida es exitosa si su costo queda a menos del 3 % del mejor costo encontrado en las 30 corridas.

Todo se repite con un solo comando (unos 4 minutos):

```bash
python src/parte2/benchmark_ga_30.py
```

## 4. Resultados

La Tabla 3 resume las 30 corridas.

**Tabla 3.** *Resultados del algoritmo genético en 30 corridas (costo del recorrido en euros).*

| Mejor | Peor | Media | Desv. est. | Mediana | Éxito (≤ 3 % del mejor) | Evaluaciones por corrida | Vecino más cercano |
|---|---|---|---|---|---|---|---|
| 1740.88 | 1911.16 | 1821.85 | 36.80 | 1831.17 | 16.7 % | 238 700 | 2279.78 |

*Fuente:* elaboración propia, `python src/parte2/benchmark_ga_30.py`. La desviación estándar es la poblacional (divide entre el número de corridas).

El mejor recorrido (semilla 12), empezando en Madrid, es:

> Madrid → Toledo → Ciudad Real → Cáceres → Badajoz → Huelva → Cádiz → Sevilla → Córdoba → Jaén → Málaga → Granada → Almería → Murcia → Alicante → Albacete → Cuenca → Teruel → Valencia → Castellón de la Plana → Tarragona → Barcelona → Girona → Lleida → Huesca → Zaragoza → Soria → Logroño → Pamplona → Donostia-San Sebastián → Vitoria-Gasteiz → Bilbao → Santander → Burgos → Palencia → Valladolid → León → Oviedo → Lugo → A Coruña → Pontevedra → Ourense → Zamora → Salamanca → Ávila → Segovia → Guadalajara → Madrid

Ese recorrido tiene 6121 km, unas 58.3 horas de viaje y 19.70 € en peajes (León–Oviedo por la AP-66 y Málaga–Granada por la AP-46).

<!-- Figura pendiente: la gráfica de convergencia se genera con
     python src/parte2/ga_tsp.py --matrix data/cost_matrix.csv --pop 300 --gens 800 --seed 12 --no-show
     y queda en results/ga_result.png. -->

## 5. Análisis

**Frente a la línea base.** En promedio, el GA encontró recorridos un 20 % más baratos que el vecino más cercano (1821.85 € frente a 2279.78 €), y su peor corrida (1911.16 €) también fue mejor que esa heurística.

**Variabilidad.** Los resultados cambian bastante de una semilla a otra: hay 170 € de diferencia entre la mejor y la peor corrida, y solo 5 de las 30 corridas (16.7 %) quedaron a menos del 3 % de la mejor. Eso indica que el GA tiende a quedarse en recorridos buenos pero distintos, sin llegar siempre al mismo. Es lo esperable en un problema con unos $2.75 \times 10^{57}$ recorridos posibles y con operadores que no incluyen búsqueda local: la mutación por inversión hace un solo cambio 2-opt al azar, en lugar de aplicar todos los que mejoran la ruta.

**Al comparar con el ACO hay que tener cuidado.** Con estos parámetros, el GA hace 238 700 evaluaciones por corrida, mientras que el ACO con 47 hormigas y 80 iteraciones hace 3 760. Comparar solo el costo final no sería justo: para comparar los dos métodos habría que darles el mismo número de evaluaciones o reportar la curva del mejor costo frente al número de evaluaciones.

## 6. Limitaciones

Además de las limitaciones de los datos de la sección 1, no ajustamos los parámetros del GA de forma sistemática; usamos los valores por defecto del código, con una población y un número de generaciones mayores para el problema real de 47 ciudades. Agregar una búsqueda local 2-opt a cada hijo probablemente reduciría la variabilidad entre corridas, a costa de más evaluaciones.
