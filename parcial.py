import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
import io
import os

style.use('ggplot')
warnings.filterwarnings('ignore')

st.set_page_config(page_title="HR Attrition - Limpieza de Datos", layout="wide", page_icon="👥")

st.markdown("""
<style>
.metric-card {
    background: #f8f9fa;
    border-left: 4px solid #1f77b4;
    padding: 12px 16px;
    border-radius: 6px;
    margin-bottom: 8px;
}
.section-header {
    background: linear-gradient(90deg, #1f77b4, #2ca02c);
    color: white;
    padding: 8px 16px;
    border-radius: 6px;
    margin: 16px 0 8px 0;
}
</style>
""", unsafe_allow_html=True)

st.title("👥 HR Employee Attrition — Pipeline de Limpieza de Datos")

# =========================
# CARGA DEL DATASET
# =========================
DEFAULT_PATH = "HR-Employee-Attrition.csv"

uploaded_file = st.file_uploader("📂 Subir archivo CSV (opcional)", type=["csv"])

if uploaded_file:
    data_original = pd.read_csv(uploaded_file)
elif os.path.exists(DEFAULT_PATH):
    data_original = pd.read_csv(DEFAULT_PATH)
    st.success(f"✅ Dataset cargado automáticamente desde `{DEFAULT_PATH}`")
else:
    st.info("👆 Sube el archivo CSV para comenzar el análisis.")
    st.stop()

data = data_original.copy()

# Renombrar columnas
data.columns = [
    "Edad","Rotacion","ViajesNegocio","TarifaDiaria","Departamento","DistanciaCasa",
    "Educacion","CampoEducativo","NumeroEmpleados","IdEmpleado","SatisfaccionEntorno",
    "Genero","TarifaHora","Involucramiento","NivelPuesto","RolTrabajo","SatisfaccionLaboral",
    "EstadoCivil","IngresoMensual","TarifaMensual","NumEmpresas","Mayor18","HorasExtra",
    "PorcentajeAumento","Rendimiento","SatisfaccionRelacion","HorasEstandar","NivelStock",
    "AniosTotales","CapacitacionesAnio","BalanceVidaTrabajo","AniosEmpresa","AniosRolActual",
    "AniosDesdeAscenso","AniosConJefe"
]

tabs = st.tabs([
    "📋 Exploración", "🛠️ Nulos & Dupl.", "🔢 Tipos", "🔡 Categóricas",
    "⚠️ Outliers", "🔗 Coherencia", "🆕 Variables", "📊 Rotación", "🧹 Resumen"
])

# ── TAB 1: Exploración ──────────────────────────────────────────────
with tabs[0]:
    st.subheader("📋 Exploración Inicial")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", data_original.shape[0])
    c2.metric("Columnas", data_original.shape[1])
    c3.metric("Nulos totales", int(data_original.isnull().sum().sum()))
    c4.metric("Duplicados", int(data_original.duplicated().sum()))

    st.markdown("**Primeras filas**")
    st.dataframe(data.head(), use_container_width=True)

    st.markdown("**Estructura del dataset**")
    info_df = pd.DataFrame({
        "Columna": data.columns,
        "Tipo": data.dtypes.values,
        "Nulos": data.isnull().sum().values,
        "Únicos": data.nunique().values
    })
    st.dataframe(info_df, use_container_width=True)

    st.markdown("**Resumen estadístico**")
    st.dataframe(data.describe(), use_container_width=True)

# ── TAB 2: Nulos & Duplicados ────────────────────────────────────────
with tabs[1]:
    st.subheader("🛠️ Manejo de Nulos y Duplicados")

    threshold = len(data) * 0.5
    cols_drop = data.columns[data.isnull().sum() > threshold].tolist()
    if cols_drop:
        st.warning(f"Columnas eliminadas por >50% nulos: {cols_drop}")
        data = data.drop(columns=cols_drop)
    else:
        st.success("✅ Ninguna columna supera el 50% de nulos.")

    estrategia_nulos = []
    for col in data.columns:
        n = data[col].isnull().sum()
        if n > 0:
            if data[col].dtype in ['int64', 'float64']:
                data[col] = data[col].fillna(data[col].mean())
                estrategia_nulos.append({"Columna": col, "Estrategia": "Media", "NulosImputados": n})
            else:
                data[col] = data[col].fillna("desconocido")
                estrategia_nulos.append({"Columna": col, "Estrategia": "Desconocido", "NulosImputados": n})

    if estrategia_nulos:
        st.markdown("**Imputaciones realizadas**")
        st.dataframe(pd.DataFrame(estrategia_nulos), use_container_width=True)
    else:
        st.success("✅ No se encontraron nulos para imputar.")

    dup_antes = data.duplicated().sum()
    data = data.drop_duplicates()
    dup_despues = data.duplicated().sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Duplicados antes", int(dup_antes))
    c2.metric("Duplicados después", int(dup_despues))
    c3.metric("Eliminados", int(dup_antes - dup_despues))

# ── TAB 3: Tipos ─────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("🔢 Corrección de Tipos de Datos")

    conversiones = []
    for col in data.columns:
        if data[col].dtype == 'object':
            try:
                data[col] = pd.to_numeric(data[col])
                conversiones.append({"Columna": col, "Convertido": True})
            except:
                conversiones.append({"Columna": col, "Convertido": False})

    st.markdown("**Conversiones a numérico**")
    st.dataframe(pd.DataFrame(conversiones), use_container_width=True)

    st.markdown("**Tipos finales**")
    tipos_df = pd.DataFrame({"Columna": data.columns, "TipoFinal": data.dtypes.values})
    st.dataframe(tipos_df, use_container_width=True)

# ── TAB 4: Categóricas ───────────────────────────────────────────────
with tabs[3]:
    st.subheader("🔡 Normalización de Valores Categóricos")
    categoricas = data.select_dtypes(include=['object']).columns

    antes = {col: data[col].unique().tolist() for col in categoricas}

    for col in categoricas:
        data[col] = (
            data[col].str.lower().str.strip()
            .str.replace(r'[áàä]', 'a', regex=True)
            .str.replace(r'[éèë]', 'e', regex=True)
            .str.replace(r'[íìï]', 'i', regex=True)
            .str.replace(r'[óòö]', 'o', regex=True)
            .str.replace(r'[úùü]', 'u', regex=True)
            .str.replace(r'[ñ]', 'n', regex=True)
        )

    despues = {col: data[col].unique().tolist() for col in categoricas}

    comparacion = pd.DataFrame({
        "Columna": list(categoricas),
        "Valores Antes": [str(antes[c]) for c in categoricas],
        "Valores Después": [str(despues[c]) for c in categoricas]
    })
    st.dataframe(comparacion, use_container_width=True)

# ── TAB 5: Outliers ──────────────────────────────────────────────────
with tabs[4]:
    st.subheader("⚠️ Detección y Eliminación de Outliers (IQR)")
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns

    rangos = []
    for col in num_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        rangos.append({
            "Columna": col, "Q1": round(Q1, 2), "Q3": round(Q3, 2), "IQR": round(IQR, 2),
            "Lím. Inferior": round(Q1 - 1.5 * IQR, 2), "Lím. Superior": round(Q3 + 1.5 * IQR, 2)
        })

    rangos_df = pd.DataFrame(rangos)

    outliers_lista = []
    for _, row in rangos_df.iterrows():
        col = row["Columna"]
        n = ((data[col] < row["Lím. Inferior"]) | (data[col] > row["Lím. Superior"])).sum()
        if n > 0:
            outliers_lista.append({"Columna": col, "Outliers": int(n)})

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Rangos IQR**")
        st.dataframe(rangos_df, use_container_width=True)
    with col2:
        st.markdown("**Columnas con outliers**")
        if outliers_lista:
            out_df = pd.DataFrame(outliers_lista).sort_values("Outliers", ascending=False)
            st.dataframe(out_df, use_container_width=True)
            fig, ax = plt.subplots(figsize=(5, max(3, len(out_df) * 0.4)))
            ax.barh(out_df["Columna"], out_df["Outliers"], color="#1f77b4")
            ax.set_xlabel("Cantidad de outliers")
            st.pyplot(fig)
        else:
            st.success("No se detectaron outliers.")

    filas_antes_outliers = data.shape[0]
    for _, row in rangos_df.iterrows():
        col = row["Columna"]
        data = data[(data[col] >= row["Lím. Inferior"]) & (data[col] <= row["Lím. Superior"])]
    filas_despues_outliers = data.shape[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Filas antes", filas_antes_outliers)
    c2.metric("Filas después", filas_despues_outliers)
    c3.metric("Eliminadas", filas_antes_outliers - filas_despues_outliers)

# ── TAB 6: Coherencia ────────────────────────────────────────────────
with tabs[5]:
    st.subheader("🔗 Validación de Coherencia Lógica")
    filas_antes_logico = data.shape[0]

    val_df = pd.DataFrame({
        "Variable": ["Edad", "DistanciaCasa", "IngresoMensual", "NivelPuesto", "AniosTotales"],
        "MinEsperado": [18, 0, 1, 1, 0], "MaxEsperado": [65, 9999, 9999999, 5, 40],
        "MinReal": [data[c].min() for c in ["Edad","DistanciaCasa","IngresoMensual","NivelPuesto","AniosTotales"]],
        "MaxReal": [data[c].max() for c in ["Edad","DistanciaCasa","IngresoMensual","NivelPuesto","AniosTotales"]]
    })
    st.markdown("**Rangos lógicos**")
    st.dataframe(val_df, use_container_width=True)

    data = data[
        (data["Edad"].between(18, 65)) & (data["DistanciaCasa"] >= 0) &
        (data["IngresoMensual"] > 0) & (data["NivelPuesto"] >= 1) & (data["AniosTotales"] >= 0)
    ]

    inc_df = pd.DataFrame({
        "Regla": ["AniosEmpresa <= AniosTotales", "AniosRolActual <= AniosEmpresa"],
        "FilasInconsistentes": [
            int((data["AniosEmpresa"] > data["AniosTotales"]).sum()),
            int((data["AniosRolActual"] > data["AniosEmpresa"]).sum())
        ]
    })
    st.markdown("**Inconsistencias entre columnas**")
    st.dataframe(inc_df, use_container_width=True)

    data = data[(data["AniosEmpresa"] <= data["AniosTotales"]) & (data["AniosRolActual"] <= data["AniosEmpresa"])]
    filas_despues_logico = data.shape[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Filas antes", filas_antes_logico)
    c2.metric("Filas después", filas_despues_logico)
    c3.metric("Eliminadas", filas_antes_logico - filas_despues_logico)

# ── TAB 7: Variables derivadas ────────────────────────────────────────
with tabs[6]:
    st.subheader("🆕 Variables Derivadas")

    data["GrupoEdad"] = pd.cut(data["Edad"], bins=[18,30,40,50,60,100],
                               labels=["joven","adulto","maduro","senior","mayor"])
    data["RangoSalario"] = pd.cut(data["IngresoMensual"], bins=[0,3000,6000,10000,999999],
                                  labels=["bajo","medio","alto","muy_alto"])
    data["TieneHorasExtra"] = data["HorasExtra"].apply(lambda x: "si" if x == "yes" else "no")

    desc_df = pd.DataFrame({
        "Variable": ["GrupoEdad", "RangoSalario", "TieneHorasExtra"],
        "Descripción": [
            "Grupo de edad (joven/adulto/maduro/senior/mayor)",
            "Rango salarial mensual (bajo/medio/alto/muy_alto)",
            "Indicador de horas extra (si/no)"
        ]
    })
    st.dataframe(desc_df, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        fig, ax = plt.subplots()
        data["GrupoEdad"].value_counts().plot.bar(ax=ax, color="#1f77b4")
        ax.set_title("Distribución GrupoEdad"); ax.set_xlabel(""); plt.xticks(rotation=30)
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots()
        data["RangoSalario"].value_counts().plot.bar(ax=ax, color="#2ca02c")
        ax.set_title("Distribución RangoSalario"); ax.set_xlabel(""); plt.xticks(rotation=30)
        st.pyplot(fig)
    with c3:
        fig, ax = plt.subplots()
        data["TieneHorasExtra"].value_counts().plot.pie(ax=ax, autopct="%1.1f%%",
                                                         colors=["#ff7f0e","#1f77b4"])
        ax.set_title("Horas Extra"); ax.set_ylabel("")
        st.pyplot(fig)

# ── TAB 8: Rotación ───────────────────────────────────────────────────
with tabs[7]:
    st.subheader("📊 Análisis de Rotación")

    rotacion_df = pd.crosstab(data["GrupoEdad"], data["Rotacion"]).reset_index()
    rotacion_df["total"] = rotacion_df.get("no", 0) + rotacion_df.get("yes", 0)
    rotacion_df["rotacion(%)"] = (rotacion_df.get("yes", 0) / rotacion_df["total"] * 100).round(2)

    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("**Tabla de rotación por grupo de edad**")
        st.dataframe(rotacion_df, use_container_width=True)
    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        groups = rotacion_df["GrupoEdad"].astype(str)
        x = np.arange(len(groups))
        w = 0.35
        ax.bar(x - w/2, rotacion_df.get("no", 0), w, label="No rotó", color="#2ca02c")
        ax.bar(x + w/2, rotacion_df.get("yes", 0), w, label="Rotó", color="#d62728")
        ax.set_xticks(x); ax.set_xticklabels(groups)
        ax.set_title("Rotación por grupo de edad"); ax.legend()
        st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.bar(groups, rotacion_df["rotacion(%)"], color="#ff7f0e")
    ax2.set_ylabel("% Rotación"); ax2.set_title("Tasa de rotación (%) por grupo de edad")
    for i, v in enumerate(rotacion_df["rotacion(%)"]):
        ax2.text(i, v + 0.3, f"{v}%", ha='center', fontsize=9)
    st.pyplot(fig2)

# ── TAB 9: Resumen global ──────────────────────────────────────────────
with tabs[8]:
    st.subheader("🧹 Resumen Global del Proceso")

    resumen_df = pd.DataFrame({
        "Etapa": [
            "Dataset original",
            "Tras eliminar columnas con >50% nulos",
            "Tras eliminar duplicados",
            "Tras eliminar outliers",
            "Dataset final (tras limpieza lógica)"
        ],
        "Filas restantes": [
            data_original.shape[0],
            data_original.shape[0],
            data_original.shape[0] - dup_antes,
            filas_despues_outliers,
            filas_despues_logico
        ]
    })
    resumen_df["Filas eliminadas vs anterior"] = resumen_df["Filas restantes"].diff().fillna(0).astype(int) * -1

    st.dataframe(resumen_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    colores = ["#1f77b4","#aec7e8","#ffbb78","#ff7f0e","#d62728"]
    ax.bar(range(len(resumen_df)), resumen_df["Filas restantes"], color=colores)
    ax.set_xticks(range(len(resumen_df)))
    ax.set_xticklabels(resumen_df["Etapa"], rotation=15, ha='right', fontsize=8)
    ax.set_ylabel("Filas"); ax.set_title("Evolución del dataset a lo largo del pipeline")
    for i, v in enumerate(resumen_df["Filas restantes"]):
        ax.text(i, v + 5, str(v), ha='center', fontsize=9)
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("**Dataset final — primeras filas**")
    st.dataframe(data.head(), use_container_width=True)

    csv_buffer = io.StringIO()
    data.to_csv(csv_buffer, index=False)
    st.download_button("⬇️ Descargar dataset limpio (.csv)", data=csv_buffer.getvalue(),
                       file_name="HR_Attrition_limpio.csv", mime="text/csv")