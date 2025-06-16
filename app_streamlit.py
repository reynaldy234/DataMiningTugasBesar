import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report, silhouette_score

st.set_page_config(page_title="Dashboard Analisis Diabetes", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# Load & encode dataset ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes_prediction_dataset.csv")  # ≈100 000 baris

    gender_le = LabelEncoder()
    smoking_le = LabelEncoder()

    df["gender"] = gender_le.fit_transform(df["gender"])
    df["smoking_history"] = smoking_le.fit_transform(df["smoking_history"])

    return df, gender_le, smoking_le

df, gender_le, smoking_le = load_data()

# Fitur & target
features = [
    "age",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
    "gender",
    "hypertension",
    "heart_disease",
    "smoking_history",
]
X = df[features]
y = df["diabetes"]

# Standarisasi & split 80‑20
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

# Train logistic regression sekali saja
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar & halaman ───────────────────────────────────────────────────────────
st.sidebar.title("Navigasi")
menu = st.sidebar.radio(
    "Pilih Halaman",
    ["Beranda", "Clustering (K-Means)", "Regresi Logistik", "Prediksi Diabetes"],
)

st.title("📊 Dashboard Analisis Risiko Diabetes")

# ─────────────────────────────────────────────────────────────────────────────
# 1‒ BERANDA ──────────────────────────────────────────────────────────────────
if menu == "Beranda":
    st.header("📁 Deskripsi Dataset")
    st.dataframe(df.head())
    st.markdown(f"Jumlah Data: **{df.shape[0]}**")
    st.markdown(f"Jumlah Fitur: **{len(df.columns)}**")

    # Statistik deskriptif dasar
    desc = df.describe().T.round(2)

    # Tambahkan missing values
    missing = df.isnull().sum()
    desc["Missing Values"] = missing

    # Tampilkan tabel deskriptif
    st.subheader("📋 Statistik Deskriptif & Missing Values")
    st.dataframe(desc.style.format("{:.2f}"))

    # Boxplot
    st.subheader("📦 Boxplot untuk Deteksi Outlier")
    numeric_cols = ["age", "bmi", "blood_glucose_level", "HbA1c_level"]
    for col in numeric_cols:
        fig_box, ax_box = plt.subplots()
        sns.boxplot(x=df[col], ax=ax_box, color='lightblue')
        ax_box.set_title(f"Boxplot: {col}")
        st.pyplot(fig_box)

    # Histogram fitur numerik
    st.subheader("📉 Distribusi Fitur Numerik Tertentu")
    selected_cols = numeric_cols
    ncols = 2
    nrows = (len(selected_cols) + ncols - 1) // ncols

    plt.figure(figsize=(12, 4 * nrows))
    df[selected_cols].hist(bins=30, layout=(nrows, ncols))
    plt.tight_layout()
    st.pyplot(plt.gcf())

# ─────────────────────────────────────────────────────────────────────────────
# 2‒ CLUSTERING ───────────────────────────────────────────────────────────────
elif menu == "Clustering (K-Means)":
    st.header("🔍 Analisis Cluster (K-Means)")

    # Elbow method
    st.subheader("Metode Elbow untuk Menentukan Jumlah Cluster Optimal")
    wcss = []
    for i in range(1, 11):
        kmeans_tmp = KMeans(n_clusters=i, random_state=0)
        kmeans_tmp.fit(X_scaled)
        wcss.append(kmeans_tmp.inertia_)

    fig_elbow, ax_elbow = plt.subplots()
    ax_elbow.plot(range(1, 11), wcss, marker="o")
    ax_elbow.set_xlabel("Jumlah Cluster")
    ax_elbow.set_ylabel("WCSS")
    ax_elbow.set_title("Metode Elbow")
    st.pyplot(fig_elbow)

    # Silhouette method dengan sample 10k
    st.subheader("Silhouette Score (Sample 10 000 Data)")
    sil_scores = []
    for k in range(2, 11):
        kmeans_tmp = KMeans(n_clusters=k, random_state=42)
        labels_tmp = kmeans_tmp.fit_predict(X_scaled)
        score_tmp = silhouette_score(X_scaled, labels_tmp, sample_size=10000, random_state=42)
        sil_scores.append(score_tmp)

    fig_sil, ax_sil = plt.subplots()
    ax_sil.plot(range(2, 11), sil_scores, marker='o')
    ax_sil.set_xlabel('Jumlah Cluster (k)')
    ax_sil.set_ylabel('Silhouette Score')
    ax_sil.set_title('Silhouette Method (10k Sample)')
    st.pyplot(fig_sil)

    best_k = np.argmax(sil_scores) + 2
    best_score = sil_scores[np.argmax(sil_scores)]
    if best_score >= 0.50:
        quality = 'struktur cluster **kuat** (kompak & terpisah jelas).'
    elif best_score >= 0.25:
        quality = 'struktur cluster **sedang** (ada overlap ringan).'
    else:
        quality = 'struktur cluster **lemah** (banyak overlap).' 

    st.markdown(f"Silhouette tertinggi = **{best_score:.3f}** pada **k = {best_k}** → {quality}")

    # Slider cluster
    n_clusters = st.sidebar.slider("Jumlah Cluster", 2, 6, best_k)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df["Cluster"] = cluster_labels

    # PCA 2D plot
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    cluster_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    cluster_df["Cluster"] = cluster_labels

    st.subheader("Visualisasi PCA per Cluster")
    fig_pca, ax_pca = plt.subplots()
    sns.scatterplot(data=cluster_df, x="PC1", y="PC2", hue="Cluster", palette="Set2", ax=ax_pca)
    ax_pca.set_title("PCA Clustering")
    st.pyplot(fig_pca)

    # Rangkuman cluster
    st.subheader("📊 Karakteristik Rata‑rata per Cluster")
    cluster_summary = df.groupby("Cluster")[features + ["diabetes"]].mean().reset_index()
    cluster_counts = df["Cluster"].value_counts().reset_index()
    cluster_counts.columns = ["Cluster", "Jumlah Individu"]
    cluster_summary = pd.merge(cluster_summary, cluster_counts, on="Cluster")
    cluster_summary["Persentase Diabetes"] = df.groupby("Cluster")["diabetes"].mean().values * 100
    st.dataframe(cluster_summary.style.format({"Persentase Diabetes": "{:.2f}%"}))

# ─────────────────────────────────────────────────────────────────────────────
# 3‒ REGRESI LOGISTIK (test set) ──────────────────────────────────────────────
elif menu == "Regresi Logistik":
    st.header("📈 Evaluasi Model Regresi Logistik – Test Set")

    y_prob = log_reg.predict_proba(X_test)[:, 1]
    y_pred = log_reg.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    st.subheader("ROC Curve")
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"AUC = {auc
