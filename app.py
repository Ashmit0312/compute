import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, HDBSCAN
from sklearn.metrics import silhouette_score,davies_bouldin_score,calinski_harabasz_score
import seaborn as sns
import matplotlib.pyplot as plt
import hdbscan


st.set_page_config(page_title="Interactive Clustering Dashboard", layout="wide")
st.title("🗣️GooGooGaGa Predictor ")
st.markdown("Experiment with encoders, scalers, algorithms, and visualize clusters in 3D PCA space!")


uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df.drop("CustomerID",axis=1)
    st.write("### Preview of Dataset")
    st.dataframe(df.head())

    # Separate numeric and categorical columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not num_cols and not cat_cols:
        st.error(" No valid numeric or categorical columns found for processing.")
        st.stop()

    
    st.sidebar.header("⚙️ Preprocessing Options")

    scaler_choice = st.sidebar.selectbox(
        "Choose Feature Scaler:",
        ['None', 'StandardScaler', 'MinMaxScaler', 'RobustScaler']
    )

    # Initialize chosen scaler
    if scaler_choice == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_choice == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif scaler_choice == 'RobustScaler':
        scaler = RobustScaler()
    else:
        scaler = 'passthrough'

    # Create column transformer
    ct = make_column_transformer(
        (OneHotEncoder(sparse_output = False), cat_cols),
        (scaler, num_cols),
        remainder='drop'
    )

    ct.set_output(transform="pandas")

    
    st.sidebar.header(" Clustering Algorithm")

    algo_choice = st.sidebar.selectbox(
        "Select Algorithm:",
        ['K-Means', 'DBSCAN', 'Agglomerative', 'HDBSCAN']
    )
 
    params = {}

    if algo_choice == 'K-Means':
        params['n_clusters'] = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)
        model = KMeans(n_clusters=params['n_clusters'], random_state=42)

    elif algo_choice == 'DBSCAN':
        params['eps'] = st.sidebar.slider("Epsilon (eps)", 0.1, 10.0, 0.5)
        params['min_samples'] = st.sidebar.slider("Min Samples", 1, 20, 5)
        model = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])

    elif algo_choice == 'Agglomerative':
        params['n_clusters'] = st.sidebar.slider("Number of Clusters", 2, 10, 3)
        model = AgglomerativeClustering(n_clusters=params['n_clusters'])

    elif algo_choice == 'HDBSCAN':
        params['min_cluster_size'] = st.sidebar.slider("Min Cluster Size", 2, 50, 5)
        params['min_samples'] = st.sidebar.slider("Min Samples", 2, 50, 5)
        model = HDBSCAN(min_cluster_size=params['min_cluster_size'],min_samples=params['min_samples'])

    
    pipe = make_pipeline(ct,PCA(n_components=3))


    # PCA for visualization
    
    X_pca = pipe.fit_transform(df)
    pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2', 'PCA3'])

    
    labels = model.fit_predict(X_pca)
    pca_df['Cluster'] = labels.astype(str)

    
    st.subheader("📊 Performance Metrics")
    mask =labels!=-1
    if mask.sum()>0:
        sil_score = silhouette_score(X_pca[mask], labels[mask])
        st.metric("Silhouette Score", f"{sil_score:.3f}")
        dbi_score = davies_bouldin_score(X_pca[mask], labels[mask])
        st.metric("Davie Bouldin Score", f"{dbi_score:.3f}")
        ch_score = calinski_harabasz_score(X_pca[mask], labels[mask])
        st.metric("Calinski Harabasz Score", f"{ch_score:.3f}")
    else:
        st.warning("⚠️ Silhouette Score cannot be computed (only one cluster or all noise points).")

    
    st.subheader(" Seaborn PCA Visualization (3D)")
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
            X_pca[:,0], X_pca[:,1],X_pca[:,2],
            c=labels, cmap='viridis', s=50, alpha=0.7
    )
    ax.set_title(f"{algo_choice} Clustering (PCA1 vs PCA2 vs PCA3)", fontsize=14)
    st.pyplot(fig)

else:
    st.info("👆 Upload a CSV file to get started.")