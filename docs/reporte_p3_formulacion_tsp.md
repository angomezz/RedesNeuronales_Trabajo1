# Formulación del problema del viajero (TSP)

**Autor:** Juan Felipe Miranda Arciniegas (Persona 3), a partir de la versión de Juan José Baena Garcia (Persona 5)

## 1. El problema

Un vendedor tiene que pasar por las $N = 47$ capitales de provincia de la España peninsular, cada una una sola vez, y volver a la ciudad de donde salió. Queremos encontrar el orden de visita que haga el viaje lo más barato posible.

Podemos verlo como un grafo completo: cada ciudad es un nodo, $V = \{0, 1, \dots, N-1\}$, y entre cada par de ciudades $i \neq j$ hay un arco con un costo $C_{ij}$ en euros, que es lo que cuesta ir en carro de $i$ a $j$.

## 2. Costo de ir de una ciudad a otra

Según el enunciado, el costo de cada trayecto es la suma de tres cosas: el tiempo del vendedor, los peajes y el combustible:

$$
C_{ij}(V_h) = \underbrace{V_h \, T_{ij}}_{\text{tiempo}} + \underbrace{P_{ij}}_{\text{peajes}} + \underbrace{D_{ij} \, \frac{c}{100} \, p}_{\text{combustible}}
$$

donde:

- $V_h$ es el valor de la hora del vendedor (€/h). Es el parámetro que se estudia: ¿cambia la ruta si cambia $V_h$?
- $T_{ij}$ es el tiempo de viaje entre $i$ y $j$ en horas (`data/time_matrix.csv`).
- $P_{ij}$ es lo que se paga en peajes en ese trayecto, en euros (`data/toll_matrix.csv`).
- $D_{ij}$ es la distancia por carretera en km (`data/dist_matrix.csv`).
- $c$ es el consumo del carro en litros cada 100 km y $p$ el precio del combustible en €/L. Con el vehículo que escogió el equipo, $c = 5.6$ L/100 km y $p = 1.62$ €/L.

Los valores de cada matriz, sus fuentes y fechas de consulta están en `data/SOURCES.md`.

Suponemos que el costo es simétrico, es decir, $C_{ij} = C_{ji}$: ir de Madrid a Toledo cuesta lo mismo que volver. Revisamos que las cuatro matrices de `data/` cumplan esto. Los costos $C_{ii}$ no se usan, porque nunca se va de una ciudad a ella misma.

## 3. Cómo se representa una solución

Un recorrido se representa como una permutación $\pi = (\pi(0), \pi(1), \dots, \pi(N-1))$ de las ciudades: $\pi(k)$ es la ciudad que se visita en la posición $k$. Como el recorrido es cerrado, después de la última ciudad se vuelve a la primera. Por ejemplo, con 4 ciudades, $\pi = (2, 0, 3, 1)$ significa ir de 2 a 0, de 0 a 3, de 3 a 1 y de 1 de vuelta a 2.

Usar permutaciones garantiza automáticamente que cada ciudad se visita exactamente una vez.

## 4. Función objetivo

El costo de un recorrido es la suma de los costos de todos sus trayectos, incluido el regreso a la ciudad de salida:

$$
f(\pi) = \sum_{k=0}^{N-2} C_{\pi(k),\,\pi(k+1)} \; + \; C_{\pi(N-1),\,\pi(0)}
$$

Y el problema es:

$$
\min_{\pi \in S_N} f(\pi)
$$

donde $S_N$ es el conjunto de todas las permutaciones de $\{0, 1, \dots, N-1\}$.

## 5. ¿Por qué no probar todos los recorridos?

Hay $N!$ permutaciones, pero muchas son el mismo recorrido:

- Da igual en qué ciudad empiece, porque el recorrido es un ciclo: $(2, 0, 3, 1)$ y $(0, 3, 1, 2)$ son el mismo viaje. Eso divide entre $N$.
- Como el costo es simétrico, recorrerlo al revés cuesta lo mismo: $(2, 0, 3, 1)$ y $(2, 1, 3, 0)$ cuestan igual. Eso divide entre 2.

Entonces el número de recorridos distintos es

$$
\frac{(N-1)!}{2} = \frac{46!}{2} \approx 2.75 \times 10^{57}.
$$

Aunque un computador revisara mil millones de recorridos por segundo, tardaría alrededor de $8.7 \times 10^{40}$ años en revisarlos todos. Por eso no se puede resolver por fuerza bruta. Existen métodos exactos más inteligentes (por ejemplo, de ramificación y corte) que sí pueden resolver problemas de este tamaño, pero el enunciado pide usar colonias de hormigas y algoritmos genéticos. Estos métodos no garantizan encontrar el óptimo, pero encuentran recorridos buenos en poco tiempo.

## 6. Qué cuenta como una evaluación

Para comparar los métodos de forma justa, una evaluación de la función objetivo es calcular $f(\pi)$ para un recorrido completo, es decir, sumar sus $N$ costos. En colonias de hormigas, cada hormiga construye un recorrido por iteración, así que se hace una evaluación por hormiga. En el algoritmo genético se evalúa la población inicial y después cada hijo nuevo que se crea; los individuos élite pasan a la siguiente generación sin volver a evaluarse.

## 7. Otra forma de escribirlo (solo como referencia)

El TSP también se puede escribir como un problema de programación lineal entera, con variables binarias $x_{ij}$ que valen 1 si el recorrido va de $i$ a $j$ y 0 si no, más restricciones para que a cada ciudad se entre y se salga una sola vez y para que no se formen ciclos pequeños separados. No usamos esta forma porque nuestros algoritmos trabajan directamente con permutaciones.
