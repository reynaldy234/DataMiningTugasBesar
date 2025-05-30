import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Dashboard Analisis Diabetes", layout="wide")

@st.cache_data
def load_data_model():
    df = pd.read_csv("diabetes_prediction_dataset.csv")

    # Encoding categorical
    df['gender'] = LabelEncoder().fit_transform(df['gender'])
    df['smoking_history'] = LabelEncoder().fit_transform(df['smoking_history'])

    features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'gender',
                'hypertension', 'heart_disease', 'smoking_history']

    X = df[features]
    y = df['diabetes']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    # Default KMeans dengan 3 cluster (bisa diubah nanti)
    kmeans = KMeans(n_clusters=3, random_state=0)
    clusters = kmeans.fit_predict(X_scaled)

    # PCA untuk visualisasi
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    cluster_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
    cluster_df['Cluster'] = clusters

    return df, model, scaler, cluster_df, clusters

df, model, scaler, cluster_df, clusters = load_data_model()

# Sidebar - Menu dan Pengaturan KMeans
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman", ["Beranda", "Clustering (K-Means)", "Regresi Logistik", "Prediksi Diabetes"])

st.sidebar.markdown("---")
st.sidebar.header("Pengaturan KMeans")
n_clusters = st.sidebar.slider("Jumlah Cluster (K)", min_value=2, max_value=10, value=3, step=1)

if menu == "Clustering (K-Means)":
    # Jalankan KMeans ulang dengan jumlah cluster dari sidebar
    features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'gender',
                'hypertension', 'heart_disease', 'smoking_history']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    cluster_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
    cluster_df['Cluster'] = clusters

    st.header("🔍 Visualisasi Clustering (K-Means)")
    fig, ax = plt.subplots()
    sns.scatterplot(data=cluster_df, x='PC1', y='PC2', hue='Cluster', palette='Set1', ax=ax)
    ax.set_title(f"PCA Projection dari {n_clusters} Cluster")
    st.pyplot(fig)

    # Tabel Karakteristik rata-rata fitur asli per cluster
    df['Cluster'] = clusters
    cluster_summary = df.groupby('Cluster')[features + ['diabetes']].mean().round(2)
    st.subheader("Tabel Karakteristik Rata-Rata Per Cluster")
    st.dataframe(cluster_summary)

    # Fungsi interpretasi risiko berdasarkan HbA1c dan Blood Glucose
    def interpret_risk(row):
        glucose = row['blood_glucose_level']
        hba1c = row['HbA1c_level']
        if glucose >= 126 or hba1c >= 6.5:
            return "Risiko Tinggi (Diabetes)"
        elif 100 <= glucose < 126 or 5.7 <= hba1c < 6.5:
            return "Risiko Sedang (Prediabetes)"
        else:
            return "Risiko Rendah (Normal)"

    cluster_summary['Interpretasi Risiko'] = cluster_summary.apply(interpret_risk, axis=1)

    st.subheader("Interpretasi Risiko Diabetes per Cluster")
    for idx, row in cluster_summary.iterrows():
        st.markdown(f"**Cluster {idx}**:")
        st.write(f"- Rata-rata Kadar Gula Darah: {row['blood_glucose_level']} mg/dL")
        st.write(f"- Rata-rata HbA1c Level: {row['HbA1c_level']}%")
        st.write(f"- Rata-rata Probabilitas Diabetes di Cluster: {row['diabetes']:.2f}")
        st.markdown(f"➡️ **{row['Interpretasi Risiko']}**")
        st.write("---")

elif menu == "Beranda":
    st.title("📊 Dashboard Analisis Risiko Diabetes")
    st.header("Deskripsi Dataset")
    st.dataframe(df.head())
    st.write("Jumlah data:", df.shape[0])
    st.write("Fitur:", df.columns.tolist())

elif menu == "Regresi Logistik":
    st.header("📈 Evaluasi Model Regresi Logistik")
    features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'gender', 'hypertension', 'heart_disease', 'smoking_history']
    X_scaled = scaler.transform(df[features])
    y = df['diabetes']
    y_prob = model.predict_proba(X_scaled)[:, 1]

    from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report

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

    y_pred = model.predict(X_scaled)
    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    st.write(cm)

elif menu == "Prediksi Diabetes":
    st.header("🧪 Prediksi Risiko Diabetes")
    st.write("Masukkan informasi kesehatan berikut untuk memprediksi kemungkinan diabetes.")

    age = st.number_input("Umur", min_value=0, max_value=120, value=30)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
    hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
    glucose = st.number_input("Kadar Glukosa Darah", min_value=50, max_value=300, value=100)
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female", "Other"])
    hypertension = st.selectbox("Hipertensi", ["Tidak", "Ya"])
    heart_disease = st.selectbox("Penyakit Jantung", ["Tidak", "Ya"])
    smoking_history = st.selectbox("Riwayat Merokok", ["never", "current", "former", "ever", "not current", "No Info"])

    if st.button("Prediksi"):
        gender_enc = LabelEncoder().fit(["Male", "Female", "Other"]).transform([gender])[0]
        smoking_enc = LabelEncoder().fit(["never", "current", "former", "ever", "not current", "No Info"]).transform([smoking_history])[0]
        hypertension_enc = 1 if hypertension == "Ya" else 0
        heart_disease_enc = 1 if heart_disease == "Ya" else 0

        input_df = pd.DataFrame([[age, bmi, hba1c, glucose, gender_enc, hypertension_enc, heart_disease_enc, smoking_enc]],
                                columns=features)

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.subheader("Hasil Prediksi")
        st.write(f"Prediksi Diabetes: **{'Ya' if prediction == 1 else 'Tidak'}**")
        st.write(f"Probabilitas: **{probability:.2f}**")
