# Sistema de Gestión del Conocimiento (KMS) — caso de estudio Softtek

Prototipo de un sistema de gestión del conocimiento aplicado como caso de estudio académico a Softtek (Monterrey, México): clasificación automática de documentos y búsqueda semántica con TF-IDF, más una ontología OWL/SKOS del conocimiento organizacional.

> Los datos de la empresa provienen de fuentes públicas (Softtek Fast Facts 2024 y Wikipedia). Los documentos de ejemplo, las instancias de la ontología y los resultados son ilustrativos; no son datos internos de Softtek.

## Entregables

| Entregable | Dónde está |
| --- | --- |
| Documento de avance (PDF) | [`Softtek_KMS_Reporte.pdf`](Softtek_KMS_Reporte.pdf) |
| Presentación | [`Softtek_KMS_Presentacion.pptx`](Softtek_KMS_Presentacion.pptx) / [`.pdf`](Softtek_KMS_Presentacion.pdf) (también en inglés: `Softtek_KMS_Presentation.*`) |
| Código fuente del prototipo | [`src/`](src/) |
| Scripts de clasificación automática | [`src/classifier.py`](src/classifier.py) |
| Implementación de la búsqueda semántica | [`src/semantic_search.py`](src/semantic_search.py) |
| Documentos usados para pruebas | [`data/documents.csv`](data/documents.csv) (60 etiquetados) y [`data/new_documents.csv`](data/new_documents.csv) (6 sin etiqueta) |
| Ontología | [`ontology/softtek_ontology.ttl`](ontology/softtek_ontology.ttl) |
| Ejercicios prácticos (Temas 1–6) | [`exercises/`](exercises/) |
| Capturas de pantalla | [`docs/capturas/`](docs/capturas/) |
| Salidas de ejecución | [`outputs/`](outputs/) |

## Instalación

Requiere Python 3.10 o superior.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Ejecución

```
.venv/bin/python src/classifier.py                          # entrena, evalúa y clasifica data/new_documents.csv
.venv/bin/python src/classifier.py --text "algún texto"     # clasifica un texto
.venv/bin/python src/semantic_search.py --demo              # consultas de ejemplo
.venv/bin/python src/semantic_search.py "tu pregunta"       # una consulta
.venv/bin/python src/semantic_search.py                     # modo interactivo
.venv/bin/python src/semantic_search.py --demo --baseline   # sin stemming ni expansión (comparación)
.venv/bin/python exercises/ex1_text_preprocessing.py        # ... ex1 a ex6
```

## Cómo funciona

- **Clasificación** (`src/classifier.py`): TF-IDF (unigramas + bigramas, stemming de Porter) con Naive Bayes y regresión logística. La regresión logística logra 0.90 de exactitud en validación cruzada de 5 particiones.
- **Búsqueda semántica** (`src/semantic_search.py`): la consulta se amplía con sinónimos de la ontología (SKOS `altLabel`), se vectoriza con TF-IDF y los documentos se ordenan por similitud coseno. MRR 1.00 en 8 consultas de prueba (0.79 sin la expansión).
- **Ontología** (`ontology/softtek_ontology.ttl`): 13 clases del dominio, 5 clases de documento, 22 relaciones y 372 tripletas; se consulta con SPARQL en `exercises/ex6_ontology_queries.py`.
