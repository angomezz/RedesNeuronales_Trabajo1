# Fuentes y construcción de los datos del TSP

Responsable de los datos: Persona 4 (Ana Sofía Gómez Zapata).
Revisión de esta documentación: 13 de octubre de 2026.

Este archivo describe exactamente cómo se construyó cada dato que usan los algoritmos, de dónde salió y qué limitaciones tiene. Todos los archivos están guardados en `data/`, así que los experimentos no necesitan internet.

## 1. Ciudades

- **Qué son:** las 47 capitales de provincia de la España peninsular (`data/cities.csv`).
- **Excluidas, según el enunciado:** Palma, Las Palmas de Gran Canaria y Santa Cruz de Tenerife (islas), y Ceuta y Melilla (no son provincias ni tienen acceso por carretera desde la península).

## 2. Coordenadas

- **Fuente:** servicio de geocodificación Nominatim de OpenStreetMap (https://nominatim.openstreetmap.org), datos © colaboradores de OpenStreetMap, licencia ODbL.
- **Cómo:** el script `scripts/01_build_cities.py` consulta cada ciudad y guarda la respuesta completa en `data/raw/geocode_XX.json`; el resultado queda en `data/coordinates.csv`.
- **Fecha de consulta:** 13/10/2026 (fecha en que se agregaron al repositorio).
- **Observación:** para 46 ciudades el punto devuelto es la ciudad como tal. Para Madrid, Nominatim devolvió la Biblioteca Nacional de España (Paseo de Recoletos), que está en el centro de la ciudad, así que no afecta las distancias de forma importante.

## 3. Distancias

- **Cómo se calcularon:** distancia en línea recta sobre la superficie de la Tierra (fórmula de Haversine, tomando la Tierra como una esfera de radio 6371 km), multiplicada por un factor de 1.28 para aproximar la distancia por carretera.
- **Archivo:** `data/dist_matrix.csv` (km). Script: `src/parte2/build_cost_matrices.py`.
- **Limitación importante:** no son distancias reales por carretera. El factor 1.28 es una suposición del equipo y no tiene fuente. Por ejemplo, Madrid–Barcelona sale en 645.6 km.

## 4. Tiempos de viaje

- **Cómo se calcularon:** distancia de la sección 3 dividida entre una velocidad promedio fija de 105 km/h, igual para todos los trayectos.
- **Archivo:** `data/time_matrix.csv` (horas).
- **Limitación importante:** como la velocidad es la misma en todos los trayectos, el tiempo es simplemente proporcional a la distancia. Esto tiene una consecuencia para el estudio del valor de la hora: el costo de cada trayecto queda como

  $$C_{ij} = D_{ij}\left(\frac{V_h}{105} + \frac{5.6}{100}\,p\right) + P_{ij},$$

  así que cambiar $V_h$ (o el precio $p$ del combustible) solo cambia cuánto pesa la distancia frente a los peajes. Con datos reales por carretera, donde unas vías son más rápidas que otras, el efecto del valor de la hora podría ser distinto.

## 5. Peajes

- **Archivo:** `data/toll_matrix.csv` (euros). Solo 8 pares de ciudades tienen peaje:

| Trayecto | Peaje usado (€) | Autopista |
|---|---|---|
| Bilbao – Zaragoza | 33.75 | AP-68 |
| Logroño – Zaragoza | 17.50 | AP-68 |
| Bilbao – Logroño | 16.25 | AP-68 |
| Oviedo – León | 14.30 | AP-66 |
| Madrid – Ávila | 9.85 | AP-6 / AP-51 |
| Madrid – Segovia | 7.30 | AP-6 / AP-61 |
| León – Ourense | 5.60 | AP-71 |
| Málaga – Granada | 5.40 | AP-46 |

- **Pendiente de verificar:** estos valores no tienen enlace a la tarifa oficial de la que salieron. Hay que confirmarlos con las tarifas 2026 publicadas por cada concesionaria o por el Ministerio de Transportes.
- **Autopistas que ya no cobran y por eso tienen peaje 0:** AP-1 Burgos–Armiñón (desde diciembre de 2018); AP-7 Tarragona–Alicante y AP-4 Sevilla–Cádiz (desde el 1 de enero de 2020, al terminar la concesión sin renovarla); AP-2 Zaragoza–El Vendrell y AP-7 Tarragona–La Jonquera (desde septiembre de 2021).
- **Limitaciones:**
  - El peaje solo se asigna al par de ciudades listado. Un trayecto entre otras dos ciudades que en la realidad pase por una de esas autopistas tiene peaje 0 en la matriz.
  - Hay autopistas con peaje que no están en la tabla. Por ejemplo, la AP-7 entre Alicante y Cartagena sigue con concesión hasta 2048. También habría que revisar otras como la AP-6 hacia Valladolid, la AP-9 en Galicia, la AP-8 en el País Vasco y la AP-15 en Navarra.

## 6. Vehículo

- **Modelo:** SEAT León 1.5 TSI de 130 CV (96 kW), gasolina, cambio manual.
- **Consumo usado:** 5.6 L/100 km (ciclo combinado WLTP).
- **Fuentes:**
  - Ficha técnica oficial de SEAT (dossier de prensa del SEAT León 2020): rango WLTP de 5.4 a 6.1 L/100 km para el 1.5 TSI de 130 CV, 5 puertas. https://mundoseat.seat.com/mediacenter_netstor/seat-media-center/global_site/img/PressKits/IPP-SEAT-Leon-2020/Especificaciones-tecnicas-5-puertas.pdf
  - km77, ficha del SEAT León 1.5 TSI 130 CV Xcellence: 5.6 L/100 km combinado WLTP. https://www.km77.com/coches/seat/leon/2020/5-puertas/estandar/leon-15-tsi-96-kw-130-cv-startstop-xcellence/datos
- **Fecha de consulta:** 13/10/2026.

## 7. Combustible

- **Tipo:** gasolina 95 E5.
- **Precio usado en las matrices:** 1.62 €/L.
- **Limitación importante:** este precio está desactualizado. El 1 de octubre de 2026 el precio medio nacional de la gasolina 95 era de 1.857 €/L, según datos oficiales de las estaciones de servicio (Demócrata, 1 de octubre de 2026, https://www.democrata.es/economia/precio-gasolina-hoy-1-octubre-2026/). La fuente oficial para consultarlo es el Geoportal de Gasolineras del Ministerio para la Transición Ecológica (https://geoportalgasolineras.es).
- **Cómo actualizarlo:** cambiar el parámetro `fuel_price` en `build_matrices()` de `src/parte2/build_cost_matrices.py` y volver a generar las matrices.

## 8. Costo total

El costo de ir de la ciudad $i$ a la $j$ es

$$C_{ij}(V_h) = V_h\,T_{ij} + P_{ij} + D_{ij}\,\frac{5.6}{100}\cdot 1.62$$

- **Archivo:** `data/cost_matrix.csv`, generado con $V_h = 20$ €/h.
- Las cuatro matrices son simétricas ($C_{ij} = C_{ji}$) y tienen ceros en la diagonal.
