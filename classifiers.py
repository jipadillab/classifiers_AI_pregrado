import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# Modelos
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Plotly para gráficos interactivos
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Clasificadores ML | Interactivo", layout="wide")
st.title("⚙️ Laboratorio Interactivo de Clasificación ML (Plotly)")
st.markdown("""
Explora el pipeline de Machine Learning ajustando hiperparámetros en tiempo real. 
Pasa el cursor sobre los gráficos para ver el detalle de los datos y haz zoom en las fronteras de decisión.
""")

# --- SIDEBAR: CONFIGURACIÓN DE DATOS Y MODELO ---
st.sidebar.header("1. Selección de Datos")
dataset_name = st.sidebar.selectbox(
    "Elige el Dataset:",
    ("Vino (Real, Multiclase)", "Iris (Real, Multiclase)", "Cáncer de Mama (Real, Binario)", "Moons (Sintético, Ruido)", "Circles (Sintético, Concéntrico)")
)

st.sidebar.header("2. Pipeline: Partición y CV")
test_size = st.sidebar.slider("Tamaño del Test Set (%)", 10, 50, 30, step=5) / 100.0
cv_folds = st.sidebar.slider("Folds para Cross-Validation (K)", 2, 10, 5, step=1)

st.sidebar.header("3. Selección de Modelo")
classifier_name = st.sidebar.selectbox(
    "Elige el Clasificador:",
    ("KNN", "SVM", "Decision Tree", "Random Forest", "LDA", "Naive Bayes")
)

# --- FUNCIÓN PARA CARGAR DATOS ---
@st.cache_data
def load_data(name):
    if name == "Vino (Real, Multiclase)":
        data = datasets.load_wine()
    elif name == "Iris (Real, Multiclase)":
        data = datasets.load_iris()
    elif name == "Cáncer de Mama (Real, Binario)":
        data = datasets.load_breast_cancer()
    elif name == "Moons (Sintético, Ruido)":
        X, y = datasets.make_moons(n_samples=300, noise=0.2, random_state=42)
        return X, y, ["Feature 1", "Feature 2"], ["Clase 0", "Clase 1"]
    else: # Circles
        X, y = datasets.make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)
        return X, y, ["Feature 1", "Feature 2"], ["Clase 0", "Clase 1"]
    
    return data.data, data.target, data.feature_names, data.target_names

X, y, feature_names, target_names = load_data(dataset_name)

# --- PREPROCESO Y REDUCCIÓN DE DIMENSIONALIDAD (PCA) ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if X.shape[1] > 2:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    st.sidebar.info("💡 PCA aplicado: Dimensionalidad reducida a 2D para visualización.")
else:
    X_pca = X_scaled

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=test_size, random_state=42, stratify=y)

# --- EDA (Análisis Exploratorio) con Plotly ---
with st.expander("📊 Ver Análisis Exploratorio de Datos (EDA)"):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write(f"**Dataset shape:** {X.shape[0]} muestras, {X.shape[1]} características originales.")
        st.write(f"**Clases objetivo:** {len(np.unique(y))} clases ({', '.join(target_names)}).")
    with col2:
        df_y = pd.DataFrame({'Clase': [target_names[i] if i < len(target_names) else str(i) for i in y]})
        fig_eda = px.histogram(df_y, x='Clase', color='Clase', title="Distribución de Clases", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_eda.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_eda, use_container_width=True)

# --- HIPERPARÁMETROS DINÁMICOS ---
st.sidebar.header(f"⚙️ Hiperparámetros: {classifier_name}")

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        params["K"] = st.sidebar.slider("Número de Vecinos (K)", 1, 50, 5)
    elif clf_name == "SVM":
        params["C"] = st.sidebar.select_slider("Regularización (C)", options=[0.01, 0.1, 1, 10, 100], value=1)
        params["kernel"] = st.sidebar.selectbox("Kernel", ("rbf", "linear", "poly"))
    elif clf_name == "Decision Tree":
        params["max_depth"] = st.sidebar.slider("Profundidad Máxima", 1, 20, 5)
    elif clf_name == "Random Forest":
        params["n_estimators"] = st.sidebar.slider("Número de Árboles", 10, 200, 50, step=10)
        params["max_depth"] = st.sidebar.slider("Profundidad Máxima", 1, 20, 5)
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        return KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        return SVC(C=params["C"], kernel=params["kernel"], probability=True)
    elif clf_name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=params["max_depth"], random_state=42)
    elif clf_name == "Random Forest":
        return RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=42)
    elif clf_name == "LDA":
        return LinearDiscriminantAnalysis()
