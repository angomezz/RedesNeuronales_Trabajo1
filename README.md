# Trabajo 1: optimización numérica, metaheurística y combinatoria

Curso Redes neuronales y algoritmos bioinspirados
Universidad Nacional de Colombia, Facultad de Minas 
Departamento de Ciencias de la Computación y de la Decisión · Semestre 2026-02.

## Descripción

- **Parte 1 (optimización numérica):** comparación del descenso por gradiente con métodos
  heurísticos (algoritmos evolutivos, PSO y evolución diferencial) sobre funciones de prueba.
- **Parte 2 (optimización combinatoria):** recorrido cerrado de las 47 capitales de provincia
  de la España peninsular con colonias de hormigas y algoritmos genéticos. El costo de cada
  trayecto es la suma del valor de la hora del vendedor por el tiempo de viaje, el costo de
  los peajes y el costo del combustible.

## Equipo

| Nombre | Correo Institucional |
|---|---|
| Ana Sofía Gómez Zapata | angomezz@unal.edu.co |
| Tomás Gómez Cardona | tgomezc@unal.edu.co|
| juan Felipe Miranda Arciniegas | jumirandaa@unal.edu.co |
| Juan José Baena Garcia | jubaenag@unal.edu.co |
| Sebastian Ocampo Galvis | seocampog@unal.edu.co |

## Videos de contribución individual

En cumplimiento de las indicaciones del trabajo, cada integrante presenta un video de **2 a 3 minutos** en el que explica, en primera persona, sus aportes específicos al proyecto y muestra evidencia concreta de su trabajo, como fragmentos de código, commits, figuras o secciones del reporte.

La presentación reúne los videos de todos los integrantes:

**[Ver presentación con los videos de contribución individual](https://canva.link/m3tak8meiaeh9ky)**

Los aportes expuestos podrán ser objeto de preguntas durante la sustentación del trabajo.

## Estructura del repositorio

## Instalación

Requiere **Python 3.11 o superior** (probado con Python 3.12).

1. Clonar el repositorio y entrar a la carpeta:

   ```bash
   git clone https://github.com/USUARIO/trabajo1-tsp-espana.git
   cd trabajo1-tsp-espana
   ```

2. Crear y activar un entorno virtual:

   ```bash
   python -m venv .venv
   ```

   - Linux o macOS: `source .venv/bin/activate`
   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
   - Windows (CMD): `.venv\Scripts\activate.bat`

3. Instalar las librerías con las versiones fijas:

   ```bash
   pip install -r requirements.txt
   ```

## Ejecución

Todos los comandos se ejecutan desde la raíz del repositorio.

### Algoritmo genético para el TSP

Con ciudades sintéticas (las mismas del notebook visto en clase):

```bash
python scripts/ga_tsp.py --seed 56 --n_cities 40 --pop 200 --gens 500 --no-show
```

Con la matriz de costos real, cuando esté disponible en `data/`:

```bash
python scripts/ga_tsp.py --matrix data/cost_matrix.csv --pop 300 --gens 800 --no-show
```

Parámetros que se pueden cambiar sin editar el código:

| Parámetro | Significado | Valor por defecto |
|---|---|---|
| `--seed` | Semilla aleatoria | 56 |
| `--n_cities` | Número de ciudades sintéticas | 40 |
| `--matrix` | Archivo CSV con la matriz de costos (n × n) | ninguno |
| `--pop` | Tamaño de la población | 200 |
| `--gens` | Número de generaciones | 500 |
| `--k` | Tamaño del torneo | 3 |
| `--pc` | Probabilidad de cruce | 0.9 |
| `--pm` | Probabilidad de mutación | 0.2 |
| `--elite` | Individuos que pasan intactos a la siguiente generación | 2 |
| `--patience` | Generaciones sin mejora para parar (0 = no parar antes) | 0 |
| `--out` | Archivo JSON de salida | `results/ga_result.json` |
| `--no-show` | No abrir ventanas de gráficos | desactivado |

La salida queda en `results/`: un archivo JSON con el mejor recorrido, su costo y el número
de evaluaciones, y una imagen con el recorrido y la curva de convergencia.

### Algoritmo de colonias de hormigas (ACO) para el TSP (Persona 5)

Con ciudades sintéticas:

```bash
python scripts/aco_tsp.py --seed 56 --n_cities 40 --ants 40 --iters 100 --no-show
```

Con la matriz de costos real de España y generación de GIF animado sobre el mapa:

```bash
python scripts/aco_tsp.py --matrix data/cost_matrix.csv --coords data/coordinates.csv --ants 47 --iters 80 --gif --no-show
```

Parámetros que se pueden cambiar sin editar el código:

| Parámetro | Significado | Valor por defecto |
|---|---|---|
| `--seed` | Semilla aleatoria | 56 |
| `--n_cities` | Número de ciudades sintéticas | 40 |
| `--matrix` | Archivo CSV con la matriz de costos (n × n) | ninguno |
| `--coords` | Archivo CSV con coordenadas (id, ciudad, lat, lon) | ninguno |
| `--ants` | Número de hormigas por iteración ($m$) | 47 |
| `--iters` | Número de iteraciones ($T$) | 100 |
| `--alpha` | Peso de la concentración de feromona ($\alpha$) | 1.0 |
| `--beta` | Peso de la visibilidad heurística ($\beta$) | 3.0 |
| `--rho` | Tasa de evaporación de feromonas ($\rho$) | 0.1 |
| `--q` | Constante de depósito de feromona ($Q$) | 100.0 |
| `--elite` | Refuerzo elitista de la mejor solución global ($e$) | 2.0 |
| `--gif` | Generar GIF animado del recorrido sobre el mapa | desactivado |
| `--out` | Archivo JSON de salida | `results/aco_result.json` |
| `--no-show` | No abrir ventanas de gráficos | desactivado |

### Estudio del valor de la hora del vendedor ($V_h$)

Ejecuta el análisis de sensibilidad y genera gráficos de trade-off y descomposición de costos:

```bash
python scripts/study_hourly_value.py --no-show
```

### Arnés de 30 corridas independientes (ACO)

Ejecuta las 30 corridas con semillas del 1 al 30 y reporta tabla estadística completa:

```bash
python scripts/benchmark_aco_30.py --hourly-value 20.0
```

### Coordenadas de las 47 ciudades

Lee `data/cities.csv`, escribe `data/coordinates.csv` y guarda las respuestas crudas en `data/raw/`:

```bash
python scripts/01_build_cities.py
```

Antes de ejecutarlo, editar en el script la línea `user_agent` con un correo de contacto. Alternativamente, `data/coordinates.csv` ya se encuentra validado y versionado en el repositorio.

## Pruebas

```bash
python -m pytest test/ -q
```

## Semillas aleatorias

| Experimento | Semilla por defecto |
|---|---|
| Ciudades sintéticas (iguales a las del notebook de clase) | 56 |
| Algoritmo genético | 56 (configurable con `--seed`) |
| Colonias de hormigas (ACO) | 56 (configurable con `--seed`) |
| Arnés de 30 corridas | 1 a 30 fijas y documentadas |

## Datos y fuentes

Cada dato del TSP está documentado en `data/SOURCES.md` con su fuente y su fecha de consulta.
Los datos que usan los experimentos se guardan en `data/`, de modo que la ejecución no
depende de internet ni de APIs externas.

## Cacería de la alucinación

En cumplimiento de las indicaciones del trabajo, se documentan los hallazgos en los que las herramientas de inteligencia artificial generaron información incorrecta, imprecisa, desactualizada o soluciones subóptimas durante el desarrollo del proyecto.

Los hallazgos se encuentran documentados en archivos Markdown dentro del directorio `docs/`, organizados por integrante:

- `docs/caceria_alucinacion_p1.md`: hallazgos del integrante 1.
- `docs/caceria_alucinacion_p2.md`: hallazgos del integrante 2.
- `docs/caceria_alucinacion_p3.md`: hallazgos del integrante 3.
- `docs/caceria_alucinacion_p4.md`: hallazgos del integrante 4.
- `docs/caceria_alucinacion_p5.md`: hallazgos del integrante 5.

Cada archivo contiene los casos identificados por su respectivo integrante, incluyendo los prompts utilizados, las respuestas relevantes de la IA, la evidencia que demuestra los errores, las correcciones realizadas y las lecciones aprendidas.

Esta documentación permite evidenciar el análisis crítico de las respuestas generadas por inteligencia artificial durante el desarrollo del proyecto, de acuerdo con las categorías de hallazgos establecidas en el enunciado.

## Estado del trabajo

- [x] Lista de las 47 capitales peninsulares
- [x] Algoritmo genético con pruebas (P4)
- [x] Coordenadas verificadas de las 47 capitales (`data/coordinates.csv`)
- [x] Matrices de distancias, tiempos, peajes y costos combinados (`data/`)
- [x] Colonias de hormigas (ACO) con pruebas unitarias (P5)
- [x] Estudio del valor de la hora y punto crítico de cambio de ruta (P5)
- [x] GIF del recorrido sobre el mapa de España (`results/spain_tsp_aco.gif`) (P5)
- [x] Arnés de 30 corridas independientes con estadísticas y tasa de éxito (P5)
- [x] Cacería de la alucinación documentada para P5 (`docs/caceria_alucinacion_p5.md`)
- [ ] Parte 1 (gradiente, evolutivos, PSO y evolución diferencial)
- [ ] Reporte técnico y bibliografía final integrados
