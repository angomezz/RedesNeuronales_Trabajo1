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

### Coordenadas de las 47 ciudades

Requiere conexión a internet. Lee `data/cities.csv`, escribe `data/coordinates.csv` y guarda
las respuestas crudas en `data/raw/`:

```bash
python scripts/01_build_cities.py
```

Antes de ejecutarlo, editar en el script la línea `user_agent` con un correo de contacto.

## Pruebas

```bash
python -m pytest tests/ -q
```

## Semillas aleatorias

| Experimento | Semilla por defecto |
|---|---|
| Ciudades sintéticas (iguales a las del notebook de clase) | 56 |
| Algoritmo genético | 56 (configurable con `--seed`) |

## Datos y fuentes

Cada dato del TSP está documentado en `data/SOURCES.md` con su fuente y su fecha de consulta.
Los datos que usan los experimentos se guardan en `data/`, de modo que la ejecución no
depende de internet ni de APIs externas.

## Estado del trabajo

- [x] Lista de las 47 capitales
- [x] Algoritmo genético con pruebas
- [ ] Coordenadas verificadas
- [ ] Matrices de distancias, tiempos y peajes
- [ ] Colonias de hormigas
- [ ] Parte 1 (gradiente, evolutivos, PSO y evolución diferencial)
- [ ] GIF del recorrido sobre el mapa de España
- [ ] Reporte técnico y bibliografía
