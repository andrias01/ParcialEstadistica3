import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# Configuración inicial
st.set_page_config(page_title="Parcial - Análisis de Datos", layout="wide")
st.title("📂 Resolución Parcial: Análisis HR Attrition")

# 1. Carga de Datos
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("HR-Employee-Attrition.csv")
    
    # Limpieza: Eliminar columnas con >50% nulos
    limit = len(df) * 0.5
    df = df.dropna(thresh=limit, axis=1)
    
    # Imputación
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna("desconocido")
            
    # Estandarización de textos
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.lower().str.strip()
        
    return df

data = load_and_clean_data()

# Sidebar para los puntos del parcial
menu = st.sidebar.selectbox("Puntos del Parcial", [
    "1. Análisis Exploratorio (EDA)", 
    "2. Tratamiento de Outliers", 
    "3. Análisis Estadístico y Normalidad",
    "4. Prueba de Hipótesis (Homocedasticidad)"
])

if menu == "1. Análisis Exploratorio (EDA)":
    st.header("1. Exploración Inicial de Datos")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Dimensiones del Dataset:**", data.shape)
        st.write("**Nulos totales:**", data.isnull().sum().sum())
    with col2:
        st.write("**Duplicados:**", data.duplicated().sum())

    st.subheader("Muestra de Datos Limpios")
    st.dataframe(data.head(10))
    
    st.subheader("Resumen Estadístico")
    st.write(data.describe())

elif menu == "2. Tratamiento de Outliers":
    st.header("2. Detección de Outliers (Método IQR)")
    
    num_col = st.selectbox("Selecciona columna para ver outliers:", data.select_dtypes(include=[np.number]).columns)
    
    Q1 = data[num_col].quantile(0.25)
    Q3 = data[num_col].quantile(0.75)
    IQR = Q3 - Q1
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR
    
    outliers = data[(data[num_col] < lim_inf) | (data[num_col] > lim_sup)]
    
    st.warning(f"Se detectaron {len(outliers)} outliers en {num_col}")
    
    fig, ax = plt.subplots()
    sns.boxplot(x=data[num_col], ax=ax, color="salmon")
    st.pyplot(fig)

elif menu == "3. Análisis Estadístico y Normalidad":
    st.header("3. Pruebas de Normalidad")
    
    col_est = st.selectbox("Variable a evaluar:", ["monthlyincome", "age", "totalworkingyears"])
    variable = data[col_est].dropna()

    # Gráficos
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Histograma + Curva Normal")
        fig, ax = plt.subplots()
        sns.histplot(variable, kde=True, stat="density", ax=ax)
        st.pyplot(fig)
    
    with c2:
        st.subheader("QQ-Plot")
        fig_qq, ax_qq = plt.subplots()
        sm.qqplot(variable, line='s', ax=ax_qq)
        st.pyplot(fig_qq)

    # Pruebas Formales
    st.subheader("Resultados de Pruebas")
    shapiro_test = stats.shapiro(variable.sample(min(len(variable), 5000)))
    st.write(f"**Shapiro-Wilk P-Value:** {shapiro_test.pvalue:.5f}")
    
    if shapiro_test.pvalue > 0.05:
        st.success("Los datos parecen seguir una distribución normal (No se rechaza H0)")
    else:
        st.error("Los datos NO siguen una distribución normal (Se rechaza H0)")

elif menu == "4. Prueba de Hipótesis (Homocedasticidad)":
    st.header("4. Comparación de Grupos y Homocedasticidad")
    st.info("Objetivo: Evaluar si la varianza del ingreso es igual entre quienes renuncian y quienes no.")

    g_renuncia = data[data['attrition'] == 'yes']['monthlyincome']
    g_permanece = data[data['attrition'] == 'no']['monthlyincome']

    fig, ax = plt.subplots()
    sns.violinplot(x='attrition', y='monthlyincome', data=data, ax=ax)
    st.pyplot(fig)

    # Prueba de Levene
    stat, p_val = stats.levene(g_renuncia, g_permanece)
    st.write(f"**Prueba de Levene (P-Value):** {p_val:.5f}")
    
    if p_val > 0.05:
        st.success("Existe Homocedasticidad (Varianzas iguales)")
    else:
        st.error("Existe Heterocedasticidad (Varianzas diferentes)")