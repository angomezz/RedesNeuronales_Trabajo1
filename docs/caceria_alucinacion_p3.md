# Cacería de la alucinación — Persona 3 (bibliografía)

**Integrante:** Juan Felipe Miranda Arciniegas (Persona 3)
**Categoría:** Bibliografía
**IA usada:** Gemini (Google), modo "Automático", 13 de octubre de 2026

## Hallazgo: autor mal escrito en una referencia que por lo demás es correcta

### 1. Prompt utilizado (literal)

> Dame 5 referencias en formato APA 7 sobre optimización por enjambre de partículas aplicada al problema del viajero, con DOI

### 2. Parte relevante de la respuesta de la IA

La tercera referencia que dio Gemini fue:

> Elloumi, W., **Abeda, H. E.**, Abraham, A., & Alimi, A. M. (2014). A comparative study of the improvement of performance using a PSO modified by ACO applied to TSP. *Applied Soft Computing*, *25*, 234–241. https://doi.org/10.1016/j.asoc.2014.09.031

Captura de la conversación: `docs/evidencias/p3_gemini_referencias.png`

### 3. Cómo sospechamos que había un error

No lo sospechamos a simple vista: el título, la revista, el año y el DOI se veían bien. Lo encontramos porque decidimos verificar una por una todas las referencias antes de usarlas, buscando cada artículo en su página oficial y en bases de datos como dblp. Al comparar, el segundo autor no coincidía.

### 4. Evidencia

- La página oficial del artículo en ScienceDirect (Elsevier) lista como autores a **Walid Elloumi, Haikal El Abed, Ajith Abraham y Adel M. Alimi**, en *Applied Soft Computing*, volumen 25, diciembre de 2014, páginas 234–241, con el DOI 10.1016/j.asoc.2014.09.031.
- El registro del artículo en dblp (base de datos de publicaciones de computación) muestra los mismos cuatro autores.
- Otros artículos que citan este trabajo lo escriben como "Elloumi, W., El Abed, H., …".

El segundo autor se llama **Haikal El Abed**: su apellido es compuesto ("El Abed"). Gemini lo convirtió en "Abeda" y dejó la "E." como si fuera una segunda inicial del nombre. El resto de la referencia es correcto.

Enlaces de verificación (consultados el 13/10/2026):
- https://www.sciencedirect.com/science/article/abs/pii/S1568494614004840
- https://dblp.uni-trier.de/pid/77/6109.html

### 5. Corrección

Elloumi, W., El Abed, H., Abraham, A., & Alimi, A. M. (2014). A comparative study of the improvement of performance using a PSO modified by ACO applied to TSP. *Applied Soft Computing*, *25*, 234–241. https://doi.org/10.1016/j.asoc.2014.09.031

### 6. Lección aprendida

Que una referencia "exista" no significa que esté bien escrita. Aquí el DOI abría el artículo correcto, así que una revisión rápida la habría dado por buena. El error estaba en un nombre de autor, justo el tipo de detalle que nadie revisa. La verificación que lo evita es comparar **cada campo** de la referencia (todos los autores, año, revista, volumen, páginas y DOI) con la página oficial del artículo, no solo comprobar que el DOI funcione. Además, los apellidos compuestos o de otros idiomas son un punto donde la IA se equivoca con facilidad.

## Verificaciones adicionales

Para buscar este hallazgo verificamos 11 referencias en total: las 5 de Gemini con el prompt de arriba, y 6 de ChatGPT con el prompt *"Dame referencias en APA 7 de artículos de 2023 a 2025 que comparen PSO con algoritmos genéticos, con DOI."* Todas existen. El único error claro fue el de Elloumi et al. (2014).

| IA | Referencia | Resultado |
|---|---|---|
| Gemini | Zheng, Zhang y Yang (2022), *J. of Computational Design and Engineering* | Correcta en todos sus campos. |
| Gemini | Marinakis y Marinaki (2010), *Computers & Operations Research* | Existe; título, autores, año y DOI correctos. Volumen y páginas sin confirmar. |
| Gemini | Elloumi et al. (2014), *Applied Soft Computing* | **Error en el segundo autor** (hallazgo de arriba). |
| Gemini | Goldbarg, Goldbarg y de Souza (2008), capítulo en *Traveling Salesman Problem* (ed. F. Greco) | Correcta; páginas sin confirmar. |
| Gemini | Qasimi (2024), *TechComp Innovations* | Existe; datos correctos. |
| ChatGPT | Asaolu y Adanigbo, *FUOYE J. of Engineering and Technology* | Existe; año, volumen y páginas sin confirmar. |
| ChatGPT | Aldoraibi et al. (2024), *IJACSA* | Correcta. |
| ChatGPT | Mindasari, Sawaluddin y Gultom (2024), *Sinkron* | Correcta. |
| ChatGPT | Papazoglou y Biskas (2023), *Energies* | Correcta. |
| ChatGPT | Wang y Gong (2023), *Emergency Management Science and Technology* | Existe. ChatGPT puso "Article 9"; DOAJ la registra como volumen 3, número 1, páginas 1–10. |
| ChatGPT | Kumar et al. (2023), *IJRITCC* | Existe; título, revista, volumen, páginas y DOI correctos. Autores sin confirmar. |
