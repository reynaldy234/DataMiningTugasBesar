import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report

st.set_page_config(page_title="Dashboard Analisis Diabetes", layout="wide")

# Load dan siapkan data
@st.cache_data

def load_data():
    df = pd.read_csv("diabetes_prediction_dataset.csv")
    df['gender'] = LabelEncoder().fit_transform(df['gender'])
    df['smoking_history'] = LabelEncoder().fit_transform(df['smoking_history'])
    return df

df = load_data()

features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level',
            'gender', 'hypertension', 'heart_disease', 'smoking_history']
X = df[features]
y = df['diabetes']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sidebar
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman", ["Beranda", "Clustering (K-Means)", "Regresi Logistik", "Prediksi Diabetes"])

st.title("📊 Dashboard Analisis Risiko Diabetes")

if menu == "Beranda":
    st.header("📁 Deskripsi Dataset")
    st.dataframe(df.head())
    st.markdown(f"Jumlah Data: **{df.shape[0]}**")
    st.markdown(f"Jumlah Fitur: **{len(df.columns)}**")

elif menu == "Clustering (K-Means)":
    st.header("🔍 Analisis Cluster (K-Means)")

    n_clusters = st.sidebar.slider("Jumlah Cluster", min_value=2, max_value=6, value=3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    cluster_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
    cluster_df['Cluster'] = cluster_labels

    df['Cluster'] = cluster_labels

    st.subheader("Visualisasi PCA per Cluster")
    fig, ax = plt.subplots()
    sns.scatterplot(data=cluster_df, x='PC1', y='PC2', hue='Cluster', palette='Set2', ax=ax)
    ax.set_title("Visualisasi PCA dari Clustering")
    st.pyplot(fig)

    st.subheader("📊 Karakteristik Rata-rata per Cluster")
    cluster_summary = df.groupby('Cluster')[features + ['diabetes']].mean().reset_index()
    cluster_counts = df['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Jumlah Individu']
    cluster_summary = pd.merge(cluster_summary, cluster_counts, on='Cluster')

    cluster_summary['Persentase Diabetes'] = df.groupby('Cluster')['diabetes'].mean().values * 100

    st.dataframe(cluster_summary.style.format({"Persentase Diabetes": "{:.2f}%"}))

    st.subheader("🧠 Interpretasi per Cluster")
    for _, row in cluster_summary.iterrows():
        cluster = int(row['Cluster'])
        hba1c = row['HbA1c_level']
        glucose = row['blood_glucose_level']
        diabetes_pct = row['Persentase Diabetes']

        st.markdown(f"### Cluster {cluster}")
        risk_level = "rendah"
        explanations = []

        if hba1c > 6.4:
            risk_level = "tinggi"
            explanations.append("HbA1c rata-rata melebihi 6.4, yang menunjukkan kemungkinan diabetes")
        elif hba1c > 5.7:
            risk_level = "sedang"
            explanations.append("HbA1c rata-rata antara 5.7 dan 6.4, yang menunjukkan pradiabetes")

        if glucose > 125:
            if risk_level != "tinggi":
                risk_level = "tinggi"
            explanations.append("Kadar glukosa rata-rata di atas 125 mg/dL, mengindikasikan diabetes")
        elif glucose > 100:
            if risk_level == "rendah":
                risk_level = "sedang"
            explanations.append("Kadar glukosa rata-rata antara 100-125 mg/dL, mengindikasikan pradiabetes")

        if not explanations:
            explanations.append("HbA1c dan kadar glukosa berada dalam batas normal")

        st.markdown(f"- **Persentase Diabetes:** {diabetes_pct:.2f}%")
        st.markdown(f"- **Tingkat Risiko:** {risk_level.capitalize()} berdasarkan:")
        for ex in explanations:
            st.markdown(f"  - {ex}")

elif menu == "Regresi Logistik":
    st.header("📈 Evaluasi Model Regresi Logistik")

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = model.predict(X_scaled)

    fpr, tpr, _ = roc_curve(y, y_prob)
    auc = roc_auc_score(y, y_prob)

    st.subheader("ROC Curve")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Kurva ROC")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Confusion Matrix dan Laporan Klasifikasi")
    cm = confusion_matrix(y, y_pred)
    cm_df = pd.DataFrame(cm, index=["Non-Diabetes", "Diabetes"], columns=["Prediksi Non-Diabetes", "Prediksi Diabetes"])
    st.dataframe(cm_df)

    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.2f}"))

elif menu == "Prediksi Diabetes":
    st.header("🧪 Prediksi Risiko Diabetes Baru")
    st.write("Masukkan data berikut untuk memprediksi kemungkinan diabetes:")

    age = st.number_input("Umur", 0, 120, 30)
    bmi = st.number_input("BMI", 10.0, 50.0, 22.0)
    hba1c = st.number_input("HbA1c Level", 3.0, 15.0, 5.5)
    glucose = st.number_input("Kadar Glukosa Darah", 50, 300, 100)
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female", "Other"])
    hypertension = st.selectbox("Hipertensi", ["Tidak", "Ya"])
    heart_disease = st.selectbox("Penyakit Jantung", ["Tidak", "Ya"])
    smoking = st.selectbox("Riwayat Merokok", ["never", "current", "former", "ever", "not current", "No Info"])

    if st.button("Prediksi"):
        input_data = pd.DataFrame([[age, bmi, hba1c, glucose,
                                     LabelEncoder().fit(["Male", "Female", "Other"]).transform([gender])[0],
                                     1 if hypertension == "Ya" else 0,
                                     1 if heart_disease == "Ya" else 0,
                                     LabelEncoder().fit(["never", "current", "former", "ever", "not current", "No Info"]).transform([smoking])[0]]],
                                    columns=features)

        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        st.subheader("Hasil Prediksi")
        st.markdown(f"**Prediksi:** {'Diabetes' if pred == 1 else 'Tidak Diabetes'}")
        st.markdown(f"**Probabilitas:** {prob:.2f}")
