import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

st.set_page_config(page_title="Parcial HR Attrition", layout="wide")
st.title("📊 Análisis de Rotación de Empleados (HR Attrition)")

# =========================
# CARGA Y LIMPIEZA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("HR-Employee-Attrition.csv")
    df_before = df.copy()

    # Limpieza
    threshold = len(df) * 0.5
    df = df.dropna(thresh=threshold, axis=1)

    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna("desconocido")

    df.columns = df.columns.str.strip()

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.lower().str.strip()

    # OUTLIERS
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

    return df_before, df

dataAntes, data = load_data()

menu = st.sidebar.radio("Secciones", [
    "EDA",
    "Outliers",
    "Normalidad",
    "Homocedasticidad",
    "Comparación Final"
])

# =========================
# 1. EDA
# =========================
if menu == "EDA":
    st.header("📊 Exploración de Datos")

    col1, col2, col3 = st.columns(3)
    col1.metric("Filas", dataAntes.shape[0])
    col2.metric("Columnas", dataAntes.shape[1])
    col3.metric("Nulos", dataAntes.isnull().sum().sum())

    st.subheader("Datos iniciales")
    st.dataframe(dataAntes.head())

    st.subheader("Resumen estadístico")
    st.dataframe(dataAntes.describe())

# =========================
# 2. OUTLIERS
# =========================
elif menu == "Outliers":
    st.header("📦 Comparación Antes vs Después")

    col = st.selectbox("Selecciona variable", data.select_dtypes(include=np.number).columns)

    limite = dataAntes[col].quantile(0.95)

    fig, ax = plt.subplots(1,2, figsize=(12,5))

    sns.boxplot(y=dataAntes[col], ax=ax[0])
    ax[0].set_title("Antes")
    ax[0].set_ylim(0, limite)

    sns.boxplot(y=data[col], ax=ax[1])
    ax[1].set_title("Después")
    ax[1].set_ylim(0, limite)

    st.pyplot(fig)

    st.info("Se observa reducción de valores extremos tras limpieza")

# =========================
# 3. NORMALIDAD
# =========================
elif menu == "Normalidad":
    st.header("📈 Pruebas de Normalidad")

    col = st.selectbox("Variable", ["MonthlyIncome", "Age", "TotalWorkingYears"])
    var = data[col].dropna()

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.histplot(var, kde=True, stat="density", ax=ax)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sm.qqplot(var, line='s', ax=ax)
        st.pyplot(fig)

    shapiro = stats.shapiro(var.sample(min(len(var), 5000)))

    st.write(f"P-valor Shapiro: {shapiro.pvalue:.5f}")

    if shapiro.pvalue < 0.05:
        st.error("No sigue distribución normal")
    else:
        st.success("Distribución normal")

# =========================
# 4. HOMOCEDASTICIDAD
# =========================
elif menu == "Homocedasticidad":
    st.header("📊 Comparación de grupos")

    g1 = data[data["Attrition"]=="yes"]["MonthlyIncome"]
    g2 = data[data["Attrition"]=="no"]["MonthlyIncome"]

    fig, ax = plt.subplots()
    sns.boxplot(x="Attrition", y="MonthlyIncome", data=data, ax=ax)
    st.pyplot(fig)

    stat, p = stats.levene(g1, g2)

    st.write(f"P-valor Levene: {p:.5f}")

    if p < 0.05:
        st.error("Varianzas diferentes (heterocedasticidad)")
    else:
        st.success("Varianzas iguales")

# =========================
# 5. COMPARACIÓN FINAL
# =========================
elif menu == "Comparación Final":
    st.header("📋 Comparación Dataset")

    comparacion = pd.DataFrame({
        "Métrica": ["Filas", "Columnas"],
        "Antes": [dataAntes.shape[0], dataAntes.shape[1]],
        "Después": [data.shape[0], data.shape[1]]
    })

    st.dataframe(comparacion)

    st.success("Se redujo el dataset eliminando outliers y mejorando calidad")