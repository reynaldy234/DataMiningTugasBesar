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

st.set_page_config(page_title="Dashboard Analisis Diabetes", layout="wide")

@st.cache_data
def load_data_model():
    df = pd.read_csv("diabetes_prediction_dataset.csv")
    df['gender'] = LabelEncoder().fit_transform(df['gender'])
    df['smoking_history'] = LabelEncoder().fit_transform(df['smoking_history'])
    features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'gender', 'hypertension', 'heart_disease', 'smoking_history']
    X = df[features]
    y = df['diabetes']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    return df, model, scaler, features, X_scaled, y

df, model, scaler, features, X_scaled, y = load_data_model()

st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman", ["Beranda", "Clustering (K-Means)", "Regresi Logistik", "Prediksi Diabetes"])

st.title("📊 Dashboard Analisis Risiko Diabetes")

if menu == "Beranda":
    st.header("📁 Deskripsi Dataset")
    st.dataframe(df.head())
    st.write(f"Jumlah data: {df.shape[0]}")
    st.write(f"Fitur: {df.columns.tolist()}")

elif menu == "Clustering (K-Means)":
    st.header("🔍 Analisis Clustering (K-Means)")

    k = st.sidebar.slider("Jumlah Cluster", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)

    df['Cluster'] = clusters
    cluster_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
    cluster_df['Cluster'] = clusters

    fig, ax = plt.subplots()
    sns.scatterplot(data=cluster_df, x='PC1', y='PC2', hue='Cluster', palette='Set1', ax=ax)
    ax.set_title("Visualisasi PCA dari Clustering")
    st.pyplot(fig)

    st.subheader("📋 Karakteristik Rata-Rata per Cluster")
    cluster_summary = df.groupby('Cluster')[features + ['diabetes']].mean().round(2)
    cluster_summary['Persentase_Diabetes'] = (df.groupby('Cluster')['diabetes'].mean() * 100).round(2)

    def interpret_risk(row):
        diabetes_percent = row['Persentase_Diabetes']
        if row['blood_glucose_level'] >= 126 or row['HbA1c_level'] >= 6.5:
            risk = "Risiko Tinggi"
            reason = "Karena kadar gula darah rata-rata \u2265 126 mg/dL atau HbA1c \u2265 6.5% \n(ambang batas diagnosa diabetes menurut standar medis)."
        elif 100 <= row['blood_glucose_level'] < 126 or 5.7 <= row['HbA1c_level'] < 6.5:
            risk = "Risiko Sedang"
            reason = "Karena kadar gula darah rata-rata antara 100-125 mg/dL atau HbA1c antara 5.7% sampai < 6.5%, \nmenandakan kondisi prediabetes."
        else:
            risk = "Risiko Rendah"
            reason = "Karena kadar gula darah rata-rata < 100 mg/dL dan HbA1c < 5.7%, \nmenunjukkan kondisi gula darah normal."
        return pd.Series([risk, reason])

    cluster_summary[['Interpretasi Risiko', 'Penjelasan']] = cluster_summary.apply(interpret_risk, axis=1)

    st.dataframe(cluster_summary)

    st.subheader("🧾 Interpretasi Tiap Cluster")
    for i, row in cluster_summary.iterrows():
        st.markdown(f"### Cluster {i}")
        st.write(f"**Risiko Diabetes:** {row['Interpretasi Risiko']}")
        st.write(f"**Persentase Diabetes:** {row['Persentase_Diabetes']}%")
        st.write(f"**Penjelasan:** {row['Penjelasan']}")
        st.markdown("---")

elif menu == "Regresi Logistik":
    st.header("📈 Evaluasi Model Regresi Logistik")
    from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report

    y_prob = model.predict_proba(X_scaled)[:, 1]
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
    cm = confusion_matrix(y, y_pred)
    st.subheader("📊 Confusion Matrix")
    st.write(pd.DataFrame(cm, index=["Aktual Negatif", "Aktual Positif"], columns=["Pred Negatif", "Pred Positif"]))

    st.subheader("📄 Classification Report")
    report = classification_report(y, y_pred, output_dict=True)
    st.write(pd.DataFrame(report).transpose())

elif menu == "Prediksi Diabetes":
    st.header("🧪 Prediksi Risiko Diabetes")
    st.write("Masukkan informasi kesehatan Anda untuk memprediksi kemungkinan risiko diabetes:")

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
