import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay

# Modelos
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Clasificadores ML | Pipeline & Deploy", layout="wide")
st.title("⚙️ Laboratorio Interactivo de Clasificación ML")
st.markdown("""
Esta aplicación recorre el pipeline de Machine Learning: **Datos -> Preproceso -> EDA -> Training -> Test -> Deploy**.
Ajusta los hiperparámetros en el menú lateral para observar en tiempo real los fenómenos de *Overfitting* (sobreajuste) y *Underfitting* (subajuste).
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
# Estandarizamos los datos (vital para KNN y SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reducimos a 2D para poder graficar fronteras de decisión en la web
if X.shape[1] > 2:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    st.sidebar.info("💡 Se aplicó PCA para reducir las características a 2 dimensiones y permitir la visualización de fronteras.")
else:
    X_pca = X_scaled

# Split dinámico
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=test_size, random_state=42, stratify=y)

# --- EDA (Análisis Exploratorio) ---
with st.expander("📊 Ver Análisis Exploratorio de Datos (EDA)"):
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Dataset shape:** {X.shape[0]} muestras, {X.shape[1]} características.")
        st.write(f"**Clases objetivo:** {len(np.unique(y))} clases ({', '.join(target_names)}).")
    with col2:
        fig_eda, ax_eda = plt.subplots(figsize=(5, 3))
        sns.countplot(x=y, palette="viridis", ax=ax_eda)
        ax_eda.set_title("Distribución de Clases")
        st.pyplot(fig_eda)

# --- HIPERPARÁMETROS DINÁMICOS Y MODELOS ---
st.sidebar.header(f"⚙️ Hiperparámetros: {classifier_name}")

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("Número de Vecinos (K)", 1, 50, 5)
        params["K"] = K
        st.sidebar.markdown("*Tip: K=1 causa Overfitting. K=50 causa Underfitting.*")
    elif clf_name == "SVM":
        C = st.sidebar.select_slider("Regularización (C)", options=[0.01, 0.1, 1, 10, 100], value=1)
        kernel = st.sidebar.selectbox("Kernel", ("rbf", "linear", "poly"))
        params["C"] = C
        params["kernel"] = kernel
        st.sidebar.markdown("*Tip: C alto ajusta más el margen (Overfitting).*")
    elif clf_name == "Decision Tree":
        max_depth = st.sidebar.slider("Profundidad Máxima", 1, 20, 5)
        params["max_depth"] = max_depth
        st.sidebar.markdown("*Tip: Profundidad=20 causa Overfitting. Profundidad=1 causa Underfitting.*")
    elif clf_name == "Random Forest":
        n_estimators = st.sidebar.slider("Número de Árboles", 10, 200, 50, step=10)
        max_depth = st.sidebar.slider("Profundidad Máxima", 1, 20, 5)
        params["n_estimators"] = n_estimators
        params["max_depth"] = max_depth
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
    else:
        return GaussianNB()

clf = get_classifier(classifier_name, params)

# --- TRAINING Y VALIDACIÓN CRUZADA ---
# Cross Validation en Training Data
kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X_train, y_train, cv=kf, scoring='accuracy')

# Fit final
clf.fit(X_train, y_train)

# Predicciones
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)

# --- VISUALIZACIÓN DE RESULTADOS ---
col_res1, col_res2 = st.columns((1, 2))

with col_res1:
    st.subheader("📈 Métricas de Evaluación")
    st.write(f"**Precisión Entrenamiento (Train):** {acc_train:.3f}")
    st.write(f"**Precisión Prueba (Test):** {acc_test:.3f}")
    st.write(f"**Validación Cruzada ({cv_folds}-Fold Mean):** {cv_scores.mean():.3f} (± {cv_scores.std():.3f})")
    
    # Gráfico de barras comparativo Train vs Test
    fig_acc, ax_acc = plt.subplots(figsize=(4, 3))
    ax_acc.bar(["Train", "Test", "CV Mean"], [acc_train, acc_test, cv_scores.mean()], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax_acc.set_ylim(0, 1.1)
    ax_acc.set_ylabel("Accuracy")
    st.pyplot(fig_acc)

with col_res2:
    st.subheader("🗺️ Fronteras de Decisión (Espacio 2D PCA)")
    
    fig_bound, ax_bound = plt.subplots(figsize=(8, 5))
    
    # Dibujar la frontera de decisión
    DecisionBoundaryDisplay.from_estimator(
        clf, X_pca, response_method="predict",
        cmap=plt.cm.RdYlBu, alpha=0.4, ax=ax_bound, xlabel='Componente Principal 1', ylabel='Componente Principal 2'
    )
    
    # Dibujar los puntos
    # Train = Círculos, Test = Triángulos
    scatter_train = ax_bound.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdYlBu, edgecolors='k', marker='o', label='Train Data')
    scatter_test = ax_bound.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdYlBu, edgecolors='white', marker='^', s=100, label='Test Data')
    
    ax_bound.set_title(f"Clasificador: {classifier_name} sobre {dataset_name}")
    ax_bound.legend(loc="best")
    st.pyplot(fig_bound)

# --- EXPLICACIÓN DE LOS ALGORITMOS (Teoría integrada) ---
st.markdown("---")
st.subheader(f"🧠 ¿Cómo funciona el modelo seleccionado ({classifier_name})?")

if classifier_name == "KNN":
    st.write("El **K-Nearest Neighbors** no 'aprende' una ecuación. Simplemente memoriza los datos de entrenamiento. Para clasificar un nuevo punto, busca los 'K' puntos más cercanos y vota. Si K es muy bajo (ej. 1), memoriza el ruido creando fronteras súper complejas (*Overfitting*). Si K es muy alto, ignora los patrones y divide el espacio de forma genérica (*Underfitting*).")
elif classifier_name == "SVM":
    st.write("Las **Máquinas de Vectores de Soporte** buscan el hiperplano (línea o curva en 2D) que maximice el margen entre las clases. Usando el 'Kernel Trick' (como RBF), pueden deformar el espacio para separar datos no lineales. El parámetro 'C' castiga los errores: un C alto fuerza al modelo a no equivocarse en el set de entrenamiento (*Overfitting*), mientras que un C bajo permite un margen más suave (*Underfitting*).")
elif classifier_name == "Decision Tree":
    st.write("Los **Árboles de Decisión** hacen preguntas en cascada (ej. ¿El componente 1 es > 0.5?). Si no limitamos su profundidad (`max_depth`), el árbol hará suficientes preguntas hasta aislar perfectamente cada punto de entrenamiento en su propia región rectángular, fallando catastróficamente con datos nuevos (*Overfitting*).")
elif classifier_name == "Random Forest":
    st.write("El **Bosque Aleatorio** soluciona el problema de los árboles individuales creando cientos de ellos, cada uno entrenado con una muestra aleatoria diferente de los datos. Al final, promedian sus resultados. Es extremadamente robusto y difícil (aunque no imposible) de sobreajustar.")
elif classifier_name == "LDA":
    st.write("El **Análisis Discriminante Lineal** asume que los datos siguen una distribución Gaussiana. Intenta proyectar los datos buscando maximizar la distancia entre las medias de las clases y minimizar la dispersión dentro de la misma clase. Solo traza fronteras lineales, por lo que sufre de *Underfitting* natural si los datos son muy complejos (como el dataset Moons o Circles).")
elif classifier_name == "Naive Bayes":
    st.write("Basado en el Teorema de Bayes, asume de manera 'ingenua' que todas las variables son estadísticamente independientes entre sí. Es increíblemente rápido y funciona muy bien como un modelo base o en procesamiento de texto, pero sus fronteras suelen ser suaves y menos precisas en conjuntos de datos fuertemente correlacionados.")
