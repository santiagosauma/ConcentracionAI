# ConcentracionAI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Issues](https://img.shields.io/github/issues/santiagosauma/ConcentracionAI)

Proyecto de análisis de datos y modelado predictivo sobre el Titanic, con enfoque en interpretabilidad y equidad. Modularizado para facilitar el mantenimiento y la extensión.

## Estructura del proyecto

```
├── src/                  # Código principal modular ejemplo básico
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   ├── visualization.py
│   ├── utils.py
│   └── main.py
├── data/                 # Datasets
├── models/               # Modelos entrenados (.pkl)
├── results/              # Métricas, figuras y tablas
├── notebooks/            # Jupyter Notebooks del proceso completo
├── paper/                # Artículo científico
├── presentation/         # Presentación
├── requirements.txt      # Dependencias
└── README.md             # Documentación
```

## Instalación

1. Clona el repositorio:
	```bash
	git clone https://github.com/santiagosauma/ConcentracionAI.git
	cd ConcentracionAI
	```
2. Instala las dependencias:
	```bash
	pip install -r requirements.txt
	```

## Uso del dashboard interactivo
Ejecuta el pipeline principal:
```bash
python dashboard/run.py
```

## Uso rápido para crear un modelo ejemplo con random forest

Ejecuta el pipeline principal:
```bash
python src/main.py
```

## Uso completo, seguir en orden los ipynb en la carpeta notebooks

## Funcionalidades
- Preprocesamiento de datos
- Entrenamiento y evaluación de modelos (Random Forest, Logistic Regression, SVM, XGBoost)
- Visualización de métricas y feature importances
- Interpretabilidad con SHAP
- Guardado de modelos y resultados

## Requisitos
- Python 3.10+
- Paquetes listados en `requirements.txt`

## Créditos
- Autores: Hector Eduardo Garza Fraga, David Alejandro Lozano Arreola, Luis Santiago Sauma Peñaloza,
Valentino Villegas Martinez, Gerardo Daniel Garcia de la Garza
- Licencia: MIT

## Contribuir
¡Pull requests y sugerencias son bienvenidas!

## Contacto
Para dudas o sugerencias, abre un issue en GitHub.

# ConcentracionAI
Carpetas para las entregas del reto de la Concentración en IA Avanzada
