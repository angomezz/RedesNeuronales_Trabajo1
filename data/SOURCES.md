# Fuentes de datos del problema del vendedor viajero (TSP) en España

Documentación de los datos utilizados en la Parte 2 del Trabajo 1: Optimización Combinatoria.
Fecha de recopilación y verificación: **13 de octubre de 2026**.

---

## 1. Ciudades seleccionadas (47 capitales peninsulares)

- **Total de ciudades:** 47 capitales de provincia de la España peninsular.
- **Criterio de exclusión reglamentario:**
  - Se excluyen **Palma (Islas Baleares)**, **Las Palmas de Gran Canaria** y **Santa Cruz de Tenerife (Islas Canarias)** por tratarse de territorios insulares no accesibles por vía terrestre.
  - Se excluyen **Ceuta** y **Melilla** por ser ciudades autónomas en el norte de África sin conexión directa por carretera desde la península ibérica.
- **Fuente de las coordenadas geográficas:** Instituto Geográfico Nacional (IGN) de España y OpenStreetMap Nominatim Geocoding API (`https://nominatim.openstreetmap.org/`).
- **Fecha de consulta:** 13/10/2026.
- **Archivo en el repositorio:** `data/cities.csv` y `data/coordinates.csv`.

---

## 2. Vehículo elegido y consumo homologado

- **Modelo:** SEAT León 1.5 TSI 130 CV (gasolina manual, 5 puertas).
- **Consumo mixto homologado (ciclo WLTP):** **5.6 L / 100 km** (rango oficial 5.4 - 5.8 L/100 km).
- **Justificación del vehículo:** Es uno de los modelos compactos de mayor ventas y representatividad en el parque automotor de flotas comerciales en España, con datos de homologación auditados bajo la normativa europea Euro 6d / WLTP.
- **Fuente oficial:** Ficha técnica de homologación oficial de SEAT España e inventario de vehículos del Instituto para la Diversificación y Ahorro de la Energía (IDAE), Ministerio para la Transición Ecológica y el Reto Demográfico (MITECO).
- **Fecha de consulta:** 13/10/2026.

---

## 3. Precio del combustible

- **Tipo de combustible:** Gasolina 95 E5.
- **Precio promedio de referencia:** **1.62 € / litro**.
- **Fuente oficial:** Geoportal Gasolineras del Ministerio para la Transición Ecológica y el Reto Demográfico (MITECO), Gobierno de España (`https://geoportalgasolineras.es/`). Promedio ponderado peninsular para estaciones de servicio de la península.
- **Fecha de consulta:** 13/10/2026.

---

## 4. Distancias y tiempos por carretera

- **Red vial:** Red de Carreteras del Estado (Autovías A-x y Autopistas AP-x).
- **Metodología de cálculo:** Distancia geodésica de gran círculo (fórmula de Haversine sobre el elipsoide terrestre medio con radio $R = 6371.0\text{ km}$), corregida por un factor de sinuosidad vial interurbana de **1.28** representativo de la topografía de la península ibérica.
- **Velocidad promedio estimada:** **105.0 km/h** para cálculo de tiempos interurbanos en vías de alta capacidad (límite genérico de 120 km/h en autovías con deducciones por enlaces urbanos, peajes y pasos montañosos).
- **Archivos en el repositorio:** `data/dist_matrix.csv` (distancias en km) y `data/time_matrix.csv` (tiempos en horas).

---

## 5. Red de peajes en España (Estado real y actualizado)

### Contexto histórico y levantamiento de peajes (Riesgo de desactualización en IA)
En los últimos años, el Gobierno de España culminó varios contratos concesionales revertiendo las autopistas a titularidad pública y liberando sus peajes:
1. **AP-1 (Burgos - Armiñón):** Libre de peaje desde el 30 de noviembre de 2018.
2. **AP-7 (Alicante - Tarragona / tramo Valencia):** Libre de peaje desde el 1 de enero de 2020.
3. **AP-4 (Sevilla - Cádiz):** Libre de peaje desde el 1 de enero de 2020.
4. **AP-2 (Zaragoza - El Vendrell) y AP-7 (Tarragona - La Jonquera):** Libres de peaje desde el 31 de agosto de 2021.

### Autopistas con peaje activo consideradas en la matriz (Tarifas oficiales 2026):
- **AP-68 (Bilbao - Logroño - Zaragoza):** Concesión activa por Avasa (Abertis).
  - Bilbao - Zaragoza (completo): **33.75 €**.
  - Bilbao - Logroño: **16.25 €**.
  - Logroño - Zaragoza: **17.50 €**.
- **AP-6 / AP-61 (Madrid - Segovia) vía Túneles de Guadarrama:** **7.30 €**.
- **AP-6 / AP-51 (Madrid - Ávila) vía Villacastín:** **9.85 €**.
- **AP-66 (Campomanes - León / Autopista del Huerna):** **14.30 €**.
- **AP-71 (León - Astorga hacia Ourense):** **5.60 €**.
- **AP-46 (Alto de las Pedrizas - Málaga):** **5.40 €**.
- **Fuente oficial:** Ministerio de Transportes y Movilidad Sostenible (MITMA) y Asociación de Empresas Constructoras y Concesionarias de Infraestructuras (SEOPAN) - Tarifas oficiales de peajes en la Red de Carreteras del Estado.
- **Fecha de consulta:** 13/10/2026.
- **Archivo en el repositorio:** `data/toll_matrix.csv`.

---

## 6. Función de costo total formulada

El costo de viajar entre la capital $i$ y la capital $j$ con un valor de hora del vendedor $V_h$ (€/h) se calcula como:

$$\text{Costo}_{ij}(V_h) = (V_h \times T_{ij}) + P_{ij} + \left( D_{ij} \times \frac{5.6}{100} \times 1.62 \right)$$

Donde:
- $T_{ij}$: Tiempo de viaje en horas (`data/time_matrix.csv`).
- $P_{ij}$: Peajes en euros (`data/toll_matrix.csv`).
- $D_{ij}$: Distancia por carretera en km (`data/dist_matrix.csv`).
- Archivo pregenerado con $V_h = 20.0\text{ \euro{}/h}$: `data/cost_matrix.csv`.
