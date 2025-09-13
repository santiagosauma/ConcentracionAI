# Datos del Proyecto Titanic

Esta carpeta contiene los archivos de datos utilizados en el proyecto de análisis y modelado del Titanic.

## Archivos disponibles

### Titanic-Dataset-Canvas.csv
Archivo base con los datos originales del Titanic. Incluye las siguientes columnas:
- PassengerId: Identificador único de pasajero
- Survived: 0 = No sobrevivió, 1 = Sobrevivió
- Pclass: Clase del boleto (1, 2, 3)
- Name: Nombre completo
- Sex: Género
- Age: Edad
- SibSp: Hermanos/esposos a bordo
- Parch: Padres/hijos a bordo
- Ticket: Número de boleto
- Fare: Tarifa pagada
- Cabin: Cabina asignada
- Embarked: Puerto de embarque (C = Cherbourg, Q = Queenstown, S = Southampton)

### Titanic_Dataset_Imputado.csv
Versión del dataset con imputación de valores faltantes (por ejemplo, edad, cabina, embarque). Útil para análisis donde se requiere un dataset sin valores nulos.

### Titanic_Dataset_Featured.csv
Dataset con ingeniería de características aplicada. Incluye variables adicionales derivadas, como tamaño de familia, títulos extraídos del nombre, agrupaciones de edad, y otras transformaciones para mejorar el modelado.

---
Cada archivo está en formato CSV y puede ser cargado directamente con pandas. Para más detalles sobre la generación de estos archivos, consulta los notebooks en la carpeta `notebooks/`.
