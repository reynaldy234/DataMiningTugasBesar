import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi halaman
st.set_page_config(page_title="Dashboard Analisis Diabetes", layout="wide")

# Fitur yang digunakan dalam model
features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level',
            'gender', 'hypertension', 'heart_disease', 'smoking_history']

# Fungsi untuk load data dan preprocessing
@st.cache_data
def load_data_model():
    df = pd.read_csv("diabetes_prediction_dataset.csv")

    gender_encoder = LabelEncoder()
    smoking_encoder = LabelEncoder()

    df['gender'] = gender_encoder.fit_transform(df['gender'])
    df['smoking_history'] = smoking_encoder.fit_transform(df['smoking_history'])

    X = df[features]
    y = df['diabetes']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    return df, model, scaler, X_scaled, y, gender_encoder, smoking_encoder

# Load data dan model
df, model, scaler, X_scaled, y, gender_encoder, smoking_encoder = load_data_model()

# Sidebar Navigasi
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman", ["📊 Beranda", "🔍 Clustering (K-Means)", "📈 Regresi Logistik", "🧪 Prediksi Diabetes"])

# Judul Halaman
st.title("Dashboard Analisis Risiko Diabetes")

# Halaman Beranda
if menu == "📊 Beranda":
    st.header("Tentang Dataset")
    st.write("""
    Dataset ini digunakan untuk menganalisis dan memprediksi risiko diabetes berdasarkan informasi kesehatan pasien seperti umur, BMI, riwayat merokok, dan lain-lain.
    """)
    st.dataframe(df.head())
    st.markdown(f"**Jumlah Data:** {df.shape[0]}")
    st.markdown(f"**Fitur:** {', '.join(df.columns)}")

# Halaman Clustering
elif menu == "🔍 Clustering (K-Means)":
    st.header("Clustering Pasien Berdasarkan Kondisi Kesehatan")

    st.markdown("""
    Clustering atau pengelompokan dilakukan untuk melihat pola kelompok pasien yang memiliki karakteristik kesehatan yang mirip.
    Anda bisa mengatur jumlah kelompok (cluster) yang ingin dibentuk.
    """)

    # Sidebar untuk pengaturan jumlah cluster
    num_clusters = st.sidebar.slider("Jumlah Cluster", min_value=2, max_value=10, value=3)

    # Elbow Method
    distortions = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X_scaled)
        distortions.append(kmeans.inertia_)

    st.subheader("Elbow Method: Menentukan Jumlah Cluster yang Optimal")
    fig1, ax1 = plt.subplots()
    ax1.plot(K_range, distortions, marker='o')
    ax1.set_xlabel("Jumlah Cluster (k)")
    ax1.set_ylabel("Inertia (Jarak Total ke Titik Tengah Cluster)")
    ax1.set_title("Grafik Elbow Method")
    st.pyplot(fig1)

    # Clustering dengan jumlah cluster dari slider
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # PCA untuk visualisasi
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X_scaled)
    cluster_df = pd.DataFrame(reduced, columns=["PC1", "PC2"])
    cluster_df["Cluster"] = cluster_labels

    st.subheader(f"Visualisasi Clustering (k = {num_clusters})")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=cluster_df, x='PC1', y='PC2', hue='Cluster', palette='Set1', ax=ax2)
    ax2.set_title("Visualisasi Cluster Pasien dengan PCA")
    st.pyplot(fig2)

# Halaman Regresi Logistik
elif menu == "📈 Regresi Logistik":
    st.header("Evaluasi Model Prediksi Diabetes")

    st.markdown("""
    Model ini menggunakan Regresi Logistik untuk memprediksi apakah seorang pasien berisiko menderita diabetes.
    Berikut adalah hasil evaluasinya:
    """)

    y_prob = model.predict_proba(X_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    auc = roc_auc_score(y, y_prob)

    st.subheader("Kurva ROC")
    fig3, ax3 = plt.subplots()
    ax3.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    ax3.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.set_title("Receiver Operating Characteristic (ROC)")
    ax3.legend()
    st.pyplot(fig3)

    y_pred = model.predict(X_scaled)
    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))
    st.subheader("Confusion Matrix")
    st.text(confusion_matrix(y, y_pred))

# Halaman Prediksi
elif menu == "🧪 Prediksi Diabetes":
    st.header("Prediksi Risiko Diabetes Berdasarkan Data Anda")

    st.markdown("Masukkan informasi Anda untuk melihat apakah Anda berisiko diabetes.")

    # Form input
    age = st.number_input("Umur", min_value=0, max_value=120, value=30)
    bmi = st.number_input("BMI (Indeks Massa Tubuh)", min_value=10.0, max_value=50.0, value=22.0)
    hba1c = st.number_input("Tingkat HbA1c", min_value=3.0, max_value=15.0, value=5.5)
    glucose = st.number_input("Kadar Glukosa Darah", min_value=50, max_value=300, value=100)
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female", "Other"])
    hypertension = st.selectbox("Riwayat Hipertensi", ["Tidak", "Ya"])
    heart_disease = st.selectbox("Riwayat Penyakit Jantung", ["Tidak", "Ya"])
    smoking_history = st.selectbox("Riwayat Merokok", ["never", "current", "former", "ever", "not current", "No Info"])

    if st.button("Prediksi Sekarang"):
        gender_enc = gender_encoder.transform([gender])[0]
        smoking_enc = smoking_encoder.transform([smoking_history])[0]
        hypertension_enc = 1 if hypertension == "Ya" else 0
        heart_disease_enc = 1 if heart_disease == "Ya" else 0

        input_data = pd.DataFrame([[age, bmi, hba1c, glucose, gender_enc,
                                    hypertension_enc, heart_disease_enc, smoking_enc]],
                                  columns=features)

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.subheader("Hasil Prediksi")
        if prediction == 1:
            st.markdown(f"🛑 **Anda kemungkinan memiliki risiko diabetes.**")
        else:
            st.markdown(f"✅ **Anda kemungkinan tidak memiliki risiko diabetes.**")

        st.markdown(f"**Probabilitas:** {probability:.2f}")
