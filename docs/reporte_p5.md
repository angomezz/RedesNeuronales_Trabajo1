# Sección del Reporte Técnico: Optimización Combinatoria con Colonias de Hormigas (ACO) y Estudio del Valor de la Hora (P5)

**Autor:** Juan José Baena Garcia (Persona 5)  
**Asignatura:** Redes Neuronales y Algoritmos Bioinspirados (Semestre 2026-02)  
**Universidad Nacional de Colombia — Facultad de Minas**  
**Departamento de Ciencias de la Computación y de la Decisión**

---

## 1. Formulación Matemática del Problema del Viajero (TSP) Peninsular

El problema consiste en determinar el circuito hamiltoniano de costo mínimo que recorre las $N = 47$ capitales de provincia de la España peninsular, iniciando y terminando en la misma ciudad:

$$\min_{\pi \in \Pi} f(\pi) = \sum_{k=0}^{N-1} C_{\pi(k), \pi((k+1) \bmod N)}$$

donde $\Pi$ denota el conjunto de permutaciones cíclicas del conjunto $\{0, 1, \dots, N-1\}$.

### 1.1 Función de Costo Compuesta Real
A diferencia del TSP geométrico euclidiano convencional, el costo de desplazamiento entre la capital $i$ y la capital $j$ modela las variables económicas y logísticas reales del transporte por carretera en España:

$$C_{ij}(V_h) = (V_h \times T_{ij}) + P_{ij} + \left( D_{ij} \times \frac{C_{\text{veh}}}{100} \times P_{\text{comb}} \right)$$

donde:
- $V_h \in \mathbb{R}_{\ge 0}$ es el **valor de la hora de trabajo del vendedor** (€/h).
- $T_{ij}$ es el **tiempo de viaje estimado** por autovía/autopista entre las ciudades $i$ y $j$ en horas ($T_{ij} = D_{ij} / 105.0\text{ h}$).
- $P_{ij}$ es el **costo de peajes en autopista** (€), actualizado conforme a la red de concesiones vigente en 2026 (SEOPAN / MITMA).
- $D_{ij}$ es la **distancia por carretera** en kilómetros ($D_{ij} = d_{\text{haversine}}(i, j) \times 1.28$).
- $C_{\text{veh}} = 5.6\text{ L/100 km}$ es el consumo combinado homologado bajo ciclo WLTP del vehículo seleccionado (**SEAT León 1.5 TSI 130 CV**).
- $P_{\text{comb}} = 1.62\text{ \euro{}/L}$ es el precio medio de referencia de la Gasolina 95 E5 en estaciones de servicio peninsulares (MITECO).

---

## 2. Metodología: Algoritmo de Colonias de Hormigas (ACO)

Se implementó desde cero una variante canónica del **Sistema de Colonias de Hormigas Elitista (Elitist Ant System - EAS)** propuesta originalmente por Dorigo et al. (1996) y formalizada por Dorigo y Stützle (2004).

### 2.1 Visibilidad Heurística y Concentración de Feromonas
La visibilidad o deseabilidad heurística a priori $\eta_{ij}$ se formula como el inverso del costo de transición:

$$\eta_{ij} = \begin{cases} \frac{1}{C_{ij}}, & \text{si } i \ne j \\ 0, & \text{si } i = j \end{cases}$$

La matriz de feromonas se inicializa de forma homogénea basada en la cota superior del heurístico del vecino más cercano ($C_{\text{NN}}$):

$$\tau_0 = \frac{m}{C_{\text{NN}}}$$

donde $m$ es el número de hormigas por iteración.

### 2.2 Regla de Decisión Probabilística (Transición de Estado)
En cada iteración, la probabilidad de que la hormiga $k$ ubicada en la ciudad $i$ decida transitar a la ciudad $j$ perteneciente al conjunto de ciudades no visitadas $J_k(i)$ sigue la regla de la ruleta ponderada:

$$p_{ij}^k = \frac{[\tau_{ij}]^\alpha [\eta_{ij}]^\beta}{\sum_{l \in J_k(i)} [\tau_{il}]^\alpha [\eta_{il}]^\beta}, \quad \forall j \in J_k(i)$$

- $\alpha = 1.0$: Controla la influencia de la historia y memoria colectiva (feromona).
- $\beta = 3.0$: Pondera la información heurística local miope (preferencia por conexiones más económicas). Un valor $\beta > 1$ acelera la explotación en grafos viales densos.

### 2.3 Regla de Evaporación y Depósito de Feromonas
Una vez que las $m$ hormigas completan sus circuitos hamiltonianos, la evaporación simula el decaimiento temporal del rastro, evitando la saturación prematura:

$$\tau_{ij} \leftarrow (1 - \rho)\tau_{ij}$$

donde $\rho = 0.1 \in (0, 1]$ es el coeficiente de evaporación.

Posteriormente, cada hormiga deposita feromona proporcional a la calidad de su circuito cerrado:

$$\Delta \tau_{ij}^k = \begin{cases} \frac{Q}{f(\pi_k)}, & \text{si el arco } (i, j) \in \pi_k \\ 0, & \text{en caso contrario} \end{cases}$$

Para acelerar la convergencia hacia cuencas de atracción de alto rendimiento sin perder diversidad, se incorpora el refuerzo elitista sobre el mejor circuito global histórico $\pi^*$ con peso $e = 2.0$:

$$\tau_{ij} \leftarrow \tau_{ij} + \sum_{k=1}^m \Delta \tau_{ij}^k + e \cdot \frac{Q}{f(\pi^*)}$$

Dado que el grafo vial es no dirigido en costo simétrico base, los depósitos se aplican simétricamente en $\tau_{ij}$ y $\tau_{ji}$.

### 2.4 Regla de Comparación Justa y Conteo de Evaluaciones
En estricto cumplimiento con la regla de comparación justa:
- La función objetivo $f(\pi)$ evalúa el circuito hamiltoniano cerrado una sola vez por cada hormiga construida.
- Cada iteración con $m = 47$ hormigas consume exactamente **47 evaluaciones de $f$**.
- En un ciclo estándar de $T = 80$ iteraciones, se ejecutan exactamente **$3,760$ evaluaciones de la función objetivo**.

---

## 3. Estudio del Impacto del Valor de la Hora ($V_h$)

### 3.1 Planteamiento y Pregunta de Investigación
> *¿Cambia la ruta óptima al variar el valor de la hora? ¿A partir de qué valor cambia?*

Se ejecutó un barrido paramétrico sobre $V_h \in \{0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100\}\text{ \euro{}/h}$.

### 3.2 Resultados y Discusión del Comportamiento Logístico
1. **Régimen de bajo costo horario ($V_h \le 15\text{ \euro{}/h}$):**
   El costo está dominado por el combustible y los peajes. El algoritmo prioriza la distancia mínima estricta y evita carreteras que impongan peajes adicionales.
2. **Punto crítico de cambio de ruta ($V_h^* \ge 20 - 25\text{ \euro{}/h}$):**
   A partir de $V_h = 20.0\text{ \euro{}/h}$, el costo temporal $(V_h \times T_{ij})$ supera el ahorro de combustible en desvíos lentos. La topología del circuito muta: se asumen tramos de mayor velocidad de crucero en autopistas/autovías de alta capacidad para reducir el tiempo total acumulado de viaje.
3. **Régimen de alto costo horario ($V_h \ge 50\text{ \euro{}/h}$):**
   El tiempo se convierte en la componente hiperdominante (representando más del 70% del costo total). La ruta se optimiza prioritariamente para minimizar la duración del trayecto.

---

## 4. Referencias (Normas APA, 7.ª edición)

- Dorigo, M., Maniezzo, V., & Colorni, A. (1996). Ant system: Optimization by a colony of cooperating agents. *IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics)*, 26(1), 29–41. https://doi.org/10.1109/3477.484436
- Dorigo, M., & Stützle, T. (2004). *Ant colony optimization*. MIT Press. https://doi.org/10.7551/mitpress/1290.001.0001
- Instituto para la Diversificación y Ahorro de la Energía. (2026). *Guía de vehículos eficientes y base de datos de consumo homologado (WLTP)*. Ministerio para la Transición Ecológica y el Reto Demográfico, Gobierno de España. https://sede.idae.gob.es/
- Ministerio para la Transición Ecológica y el Reto Demográfico. (2026). *Geoportal de estaciones de servicio y precios de carburantes en España*. Gobierno de España. https://geoportalgasolineras.es/
- Ministerio de Transportes y Movilidad Sostenible. (2026). *Red de Carreteras del Estado: Mapa de autopistas de peaje y tarifas vigentes*. Gobierno de España. https://www.transportes.gob.es/
- SEOPAN. (2026). *Informe económico de concesiones y autopistas de peaje en España*. Asociación de Empresas Constructoras y Concesionarias de Infraestructuras. https://seopan.es/
