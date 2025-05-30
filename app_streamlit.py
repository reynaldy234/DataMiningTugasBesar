import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report

st.set_page_config(page_title="Dashboard Analisis Diabetes", layout="wide")

# Global fitur
features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level',
            'gender', 'hypertension', 'heart_disease', 'smoking_history']

# Load data dan model
@st.cache_data
def load_data_model():
    df = pd.read_csv("diabetes_prediction_dataset.csv")

    # Label Encoding konsisten
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

    # KMeans Clustering
    kmeans = KMeans(n_clusters=3, random_state=0)
    clusters = kmeans.fit_predict(X_scaled)

    # PCA untuk visualisasi
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    cluster_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
    cluster_df['Cluster'] = clusters

    return df, model, scaler, cluster_df, gender_encoder, smoking_encoder

# Load
df, model, scaler, cluster_df, gender_encoder, smoking_encoder = load_data_model()

# Sidebar navigasi
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman", ["Beranda", "Clustering (K-Means)", "Regresi Logistik", "Prediksi Diabetes"])

# Judul
st.title("📊 Dashboard Analisis Risiko Diabetes")

# Halaman Beranda
if menu == "Beranda":
    st.header("Deskripsi Dataset")
    st.dataframe(df.head())
    st.write("Jumlah data:", df.shape[0])
    st.write("Fitur:", df.columns.tolist())

# Halaman Clustering
elif menu == "Clustering (K-Means)":
    st.header("🔍 Visualisasi Clustering (K-Means)")
    fig, ax = plt.subplots()
    sns.scatterplot(data=cluster_df, x='PC1', y='PC2', hue='Cluster', palette='Set1', ax=ax)
    ax.set_title("PCA Projection dari Cluster")
    st.pyplot(fig)

# Halaman Regresi Logistik
elif menu == "Regresi Logistik":
    st.header("📈 Evaluasi Model Regresi Logistik")
    X = scaler.transform(df[features])
    y = df['diabetes']
    y_prob = model.predict_proba(X)[:, 1]

    fpr, tpr, _ = roc_curve(y, y_prob)
    auc = roc_auc_score(y, y_prob)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

    y_pred = model.predict(X)
    st.subheader("Classification Report:")
    st.text(classification_report(y, y_pred))
    st.subheader("Confusion Matrix:")
    st.text(confusion_matrix(y, y_pred))

# Halaman Prediksi
elif menu == "Prediksi Diabetes":
    st.header("🧪 Prediksi Risiko Diabetes")
    st.write("Masukkan informasi kesehatan berikut untuk memprediksi kemungkinan diabetes.")

    # Input dari user
    age = st.number_input("Umur", min_value=0, max_value=120, value=30)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
    hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
    glucose = st.number_input("Kadar Glukosa Darah", min_value=50, max_value=300, value=100)
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female", "Other"])
    hypertension = st.selectbox("Hipertensi", ["Tidak", "Ya"])
    heart_disease = st.selectbox("Penyakit Jantung", ["Tidak", "Ya"])
    smoking_history = st.selectbox("Riwayat Merokok", ["never", "current", "former", "ever", "not current", "No Info"])

    if st.button("Prediksi"):
        # Encode input
        gender_enc = gender_encoder.transform([gender])[0]
        smoking_enc = smoking_encoder.transform([smoking_history])[0]
        hypertension_enc = 1 if hypertension == "Ya" else 0
        heart_disease_enc = 1 if heart_disease == "Ya" else 0

        input_df = pd.DataFrame([[age, bmi, hba1c, glucose, gender_enc,
                                  hypertension_enc, heart_disease_enc, smoking_enc]],
                                columns=features)

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.subheader("Hasil Prediksi")
        st.write(f"Prediksi Diabetes: **{'Ya' if prediction == 1 else 'Tidak'}**")
        st.write(f"Probabilitas: **{probability:.2f}**")
