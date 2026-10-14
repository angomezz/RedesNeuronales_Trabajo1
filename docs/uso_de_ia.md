# Uso de inteligencia artificial

**Responsable de la sección:** Ana Sofía Gómez Zapata (Persona 4), con los prompts aportados por cada integrante.

## 1. Cómo usamos la IA

El enunciado permite y espera el uso de herramientas de IA, pero advierte que se equivocan con total seguridad. Por eso las usamos como apoyo y no como fuente final: todo lo que generaron (código, fórmulas, datos y referencias) lo revisamos antes de dejarlo en el trabajo. Los errores más importantes que encontramos están documentados en la sección "Cacería de la alucinación".

Usamos estas herramientas:

| Herramienta | Quién la usó | Para qué |
|---|---|---|
| Claude (Anthropic) | P3 | Revisar el avance del repositorio contra el enunciado, escribir y depurar código, redactar secciones del reporte y verificar referencias |
| Gemini (Google) | P3 | Generar referencias bibliográficas para la cacería de la alucinación |
| ChatGPT (OpenAI) | P3 | Generar referencias bibliográficas para la cacería de la alucinación |
| GitHub Copilot | P2 | Corregir errores de ejecución en la animación del PSO |
| Claude (Anthropic) | P1 | Proponer código para PSO con PySwarms y corregir la animación del enjambre |
| ChatGPT (OpenAI) | P4 | Consultar datos del mundo real, como el precio del combustible |
| Claude (Anthropic) | P4 | Revisar el algoritmo genético y redactar la sección de datos y del GA |
| Asistente de IA | P5 | Código del ACO y datos de peajes (ver sus prompts abajo) |

## 2. Principales prompts y su impacto

### Persona 1 — Funciones de prueba y descenso por gradiente

| Prompt | Qué hizo la IA | Impacto en el resultado final |
|---|---|---|
| Pedido del siguiente paso de la Parte 1 con una recomendación de código, junto con el código de las funciones de prueba y los contadores de evaluaciones (a Claude). | Propuso implementar PSO con PySwarms y aseguró que contaría bien las evaluaciones, pero el conteo quedaba ambiguo. | Se revisó el conteo de evaluaciones de f; documentado en `docs/caceria_alucinacion_p1.md`. |
| Pedido para cambiar los límites del GIF del PSO, que solo graficaba entre −1 y 1 (a Claude). | Identificó bien la causa, pero el código que propuso tenía invertido el formato de los límites. | La animación se corrigió después de detectar el error; también está en la cacería de P1. |

### Persona 2 — Algoritmo evolutivo y evolución diferencial

| Prompt | Qué hizo la IA | Impacto en el resultado final |
|---|---|---|
| Se pegó el mensaje de error completo de la animación del PSO sobre Rosenbrock (GitHub Copilot). | Propuso reescribir la función de Rosenbrock, cambiar la malla de la gráfica y cambiar el escritor del GIF. | La corrección propuesta era más grande de lo necesario; se documentó como hallazgo en `docs/caceria_alucinacion_p2.md`. |

### Persona 3 — PSO, formulación del TSP y bibliografía

| Prompt | Qué hizo la IA | Impacto en el resultado final |
|---|---|---|
| "Dame 5 referencias en formato APA 7 sobre optimización por enjambre de partículas aplicada al problema del viajero, con DOI" (a Gemini) | Dio 5 referencias que existen, pero una con un autor mal escrito ("Abeda, H. E." en vez de "El Abed, H."). | Es el hallazgo de bibliografía de la cacería (`docs/caceria_alucinacion_p3.md`). |
| "Dame referencias en APA 7 de artículos de 2023 a 2025 que comparen PSO con algoritmos genéticos, con DOI." (a ChatGPT) | Dio 6 referencias; todas existen. | Quedaron documentadas como verificaciones adicionales en la cacería de P3. |

### Persona 4 — Datos del TSP y algoritmo genético

| Prompt | Qué hizo la IA | Impacto en el resultado final |
|---|---|---|
| "¿Cuál es el precio medio de la gasolina 95 en España hoy? Dame la cifra exacta y la fuente." (a ChatGPT) | Dio un precio de 1.42 €/L sin fecha ni enlace directo. | El precio real de octubre de 2026 era mucho mayor; documentado en `docs/caceria_alucinacion_p4.md`, y la sección de datos reporta que el precio de la matriz está desactualizado. |
| Pedido de revisar el algoritmo genético y documentar los datos del TSP (a Claude). | Revisó los operadores del GA, preparó el arnés de 30 corridas y comparó la documentación de los datos con el código. | Se encontró que el antiguo `SOURCES.md` no coincidía con el código (fuente de las coordenadas, distancias no reales por carretera, precio de la gasolina) y se corrigió. |

### Persona 5 — Colonias de hormigas y valor de la hora

| Prompt | Qué hizo la IA | Impacto en el resultado final |
|---|---|---|
| "Escribe una función en Python para actualizar la matriz de feromonas en un algoritmo de Colonias de Hormigas (Ant System) para el TSP, aplicando la evaporación y el depósito de todas las hormigas con los parámetros rho y Q." | Según P5, aplicó la evaporación multiplicando por ρ en lugar de por (1 − ρ). | Se corrigió en `aco_tsp.py`; documentado en `docs/caceria_alucinacion_p5.md`. |
| "¿Cuáles son las autopistas de peaje entre las capitales de Cataluña, Valencia y Aragón para incluirlas en la matriz de costos de un TSP en España?" | Dio peajes de la AP-7 y la AP-2 que ya no existen. | Esos tramos quedaron sin peaje en la matriz; documentado en la cacería de P5. |

## 3. Discusión: qué aportó la IA y qué no

La IA nos ahorró tiempo sobre todo en tres cosas: escribir código base (estructuras de los algoritmos, gráficos y animaciones), encontrar errores de ejecución y redactar. En cambio, fue poco confiable justo en las cosas que más se notan en el trabajo final:

- **Datos del mundo real.** Dio peajes que ya no existen, porque la red de autopistas en España cambió mucho desde 2018. También es fácil que use precios de combustible viejos: el valor que quedó en nuestra matriz (1.62 €/L) está por debajo del precio medio de octubre de 2026 (ver la sección de datos).
- **Números calculados de memoria.** El tamaño del espacio de búsqueda del TSP salió mal por un factor de 10 hasta que lo calculamos con código.
- **Referencias.** Las referencias pueden existir y aun así tener errores pequeños, como un autor mal escrito, que solo se ven comparando campo por campo con la página oficial.
- **Soluciones más grandes de lo necesario.** Ante un error de ejecución, la IA tiende a reescribir más código del que hace falta.

La lección general fue que cualquier cosa que la IA entregue y que se pueda comprobar (un número, un dato, una referencia, una fórmula) hay que comprobarla con una fuente primaria o con un cálculo propio antes de dejarla en el trabajo.
