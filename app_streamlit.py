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
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report

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
X_scaled = scaler.fit_transform(X)  # skala penuh: dipakai clustering & pra‑split
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

    # Histogram semua kolom numerik (kecuali target)
    st.subheader("📉 Distribusi Fitur Numerik")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "diabetes" in num_cols:
        num_cols.remove("diabetes")

    ncols = 2
    nrows = (len(num_cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        axes[i].hist(df[col], bins=30, color="skyblue", edgecolor="black")
        axes[i].set_title(f"Distribusi {col}")
    # Sembunyikan sumbu kosong jika jumlah kolom ganjil
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

# ─────────────────────────────────────────────────────────────────────────────
# 2‒ CLUSTERING ───────────────────────────────────────────────────────────────
elif menu == "Clustering (K-Means)":
    st.header("🔍 Analisis Cluster (K-Means)")

    # Elbow method
    st.subheader("Metode Elbow untuk Menentukan Jumlah Cluster Optimal")
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    fig_elbow, ax_elbow = plt.subplots()
    ax_elbow.plot(range(1, 11), wcss, marker="o")
    ax_elbow.set_xlabel("Jumlah Cluster")
    ax_elbow.set_ylabel("WCSS")
    ax_elbow.set_title("Metode Elbow")
    st.pyplot(fig_elbow)

    # Jalankan K‑Means dengan slider
    n_clusters = st.sidebar.slider("Jumlah Cluster", 2, 6, 3)
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
    sns.scatterplot(
        data=cluster_df,
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette="Set2",
        ax=ax_pca,
    )
    ax_pca.set_title("PCA Clustering")
    st.pyplot(fig_pca)

    # Rangkuman cluster
    st.subheader("📊 Karakteristik Rata‑rata per Cluster")
    cluster_summary = (
        df.groupby("Cluster")[features + ["diabetes"]].mean().reset_index()
    )
    cluster_counts = df["Cluster"].value_counts().reset_index()
    cluster_counts.columns = ["Cluster", "Jumlah Individu"]
    cluster_summary = pd.merge(cluster_summary, cluster_counts, on="Cluster")
    cluster_summary["Persentase Diabetes"] = (
        df.groupby("Cluster")["diabetes"].mean().values * 100
    )
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
    ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    ax_roc.plot([0, 1], [0, 1], "--", color="gray")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("Kurva ROC – Test Set")
    ax_roc.legend()
    st.pyplot(fig_roc)

    # Confusion‑matrix heat‑map
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax_cm,
        xticklabels=["Pred Non‑Diabetes", "Pred Diabetes"],
        yticklabels=["Actual Non‑Diabetes", "Actual Diabetes"],
    )
    ax_cm.set_xlabel("Prediksi")
    ax_cm.set_ylabel("Aktual")
    st.pyplot(fig_cm)

    # Laporan klasifikasi
    st.subheader("Laporan Klasifikasi")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))

# ─────────────────────────────────────────────────────────────────────────────
# 4‒ PREDIKSI ONLINE ──────────────────────────────────────────────────────────
elif menu == "Prediksi Diabetes":
    st.header("🧪 Prediksi Risiko Diabetes Baru")
    st.write("Masukkan data untuk memprediksi kemungkinan diabetes:")

    age = st.number_input("Umur", 0, 120, 30)
    bmi = st.number_input("BMI", 10.0, 50.0, 22.0)
    hba1c = st.number_input("HbA1c Level", 3.0, 15.0, 5.5)
    glucose = st.number_input("Kadar Glukosa Darah", 50, 300, 100)
    gender = st.selectbox("Jenis Kelamin", gender_le.classes_.tolist())
    hypertension = st.selectbox("Hipertensi", ["Tidak", "Ya"])
    heart_disease = st.selectbox("Penyakit Jantung", ["Tidak", "Ya"])
    smoking = st.selectbox("Riwayat Merokok", smoking_le.classes_.tolist())

    if st.button("Prediksi"):
        input_array = np.array([
            [
                age,
                bmi,
                hba1c,
                glucose,
                gender_le.transform([gender])[0],
                1 if hypertension == "Ya" else 0,
                1 if heart_disease == "Ya" else 0,
                smoking_le.transform([smoking])[0],
            ]
        ])
        input_scaled = scaler.transform(input_array)
        pred = log_reg.predict(input_scaled)[0]
        prob = log_reg.predict_proba(input_scaled)[0][1]

        st.subheader("Hasil Prediksi")
        st.markdown(f"**Prediksi:** {'Diabetes' if pred == 1 else 'Tidak Diabetes'}")
        st.markdown(f"**Probabilitas:** {prob:.2f}")
