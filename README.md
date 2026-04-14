# 📊 Parcial: Análisis de Attrition con Streamlit

Este proyecto presenta una solución interactiva para el análisis de deserción laboral (Attrition), integrando limpieza de datos, análisis estadístico y pruebas de hipótesis en una interfaz web moderna.

## 🚀 Características

- **Limpieza Automatizada**: Manejo de valores nulos (umbral del 50%), imputación de datos y normalización de textos.
- **Detección de Outliers**: Visualización y cálculo de valores atípicos mediante el método de Rango Intercuartílico (IQR).
- **Análisis Estadístico**: Evaluación de normalidad mediante Histogramas, QQ-Plots y el test de Shapiro-Wilk.
- **Pruebas de Hipótesis**: Verificación de homocedasticidad mediante el test de Levene para comparar varianzas entre grupos.

## 🛠️ Tecnologías Utilizadas

* **Python 3.x**
* **Streamlit**: Para la interfaz de usuario interactiva.
* **Pandas & Numpy**: Procesamiento y manipulación de datos.
* **Matplotlib & Seaborn**: Visualización de datos.
* **Scipy & Statsmodels**: Cálculos estadísticos avanzados.

## 📋 Requisitos Previos

Asegúrate de tener instaladas las dependencias necesarias. Puedes instalarlas ejecutando:

```bash
pip install streamlit pandas numpy matplotlib seaborn scipy statsmodels