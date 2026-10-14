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
