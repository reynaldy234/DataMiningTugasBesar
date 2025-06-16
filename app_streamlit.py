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

    # Tambahan histogram per fitur (kecuali diabetes)
    st.subheader("📊 Distribusi Setiap Fitur")
    feature_cols = [c for c in df.columns if c != 'diabetes']
    for col in feature_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_xlabel(col)
        ax.set_ylabel("Frekuensi")
        ax.set_title(f"Histogram {col}")
        st.pyplot(fig)
        plt.close(fig)

elif menu == "Clustering (K-Means)":
    st.header("🔍 Analisis Cluster (K-Means)")

    # Elbow Method untuk menentukan jumlah cluster optimal
    st.subheader("Metode Elbow untuk Menentukan Jumlah Cluster Optimal")
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    fig_elbow, ax_elbow = plt.subplots()
    ax_elbow.plot(range(1, 11), wcss, marker='o')
    ax_elbow.set_xlabel('Jumlah Cluster')
    ax_elbow.set_ylabel('WCSS (Within-Cluster Sum of Squares)')
    ax_elbow.set_title('Metode Elbow')
    st.pyplot(fig_elbow)

    # Slider jumlah cluster
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

    st.subheader("🧐 Interpretasi per Cluster")

    mean_hba1c = cluster_summary['HbA1c_level'].mean()
    mean_glucose = cluster_summary['blood_glucose_level'].mean()

    for _, row in cluster_summary.iterrows():
        cluster = int(row['Cluster'])
        hba1c = row['HbA1c_level']
        glucose = row['blood_glucose_level']
        diabetes_pct = row['Persentase Diabetes']

        st.markdown(f"### Cluster {cluster}")

        explanations = []

        if hba1c > mean_hba1c:
            hba1c_level = "tinggi"
            explanations.append(f"Rata-rata HbA1c {hba1c:.2f} lebih tinggi dari rata-rata semua cluster ({mean_hba1c:.2f})")
        elif hba1c > mean_hba1c - 0.3:
            hba1c_level = "sedang"
            explanations.append(f"Rata-rata HbA1c {hba1c:.2f} sedikit di bawah rata-rata semua cluster")
        else:
            hba1c_level = "rendah"
            explanations.append(f"Rata-rata HbA1c {hba1c:.2f} jauh di bawah rata-rata")

        if glucose > mean_glucose:
            glucose_level = "tinggi"
            explanations.append(f"Rata-rata glukosa {glucose:.2f} lebih tinggi dari rata-rata semua cluster ({mean_glucose:.2f})")
        elif glucose > mean_glucose - 5:
            glucose_level = "sedang"
            explanations.append(f"Rata-rata glukosa {glucose:.2f} sedikit di bawah rata-rata")
        else:
            glucose_level = "rendah"
            explanations.append(f"Rata-rata glukosa {glucose:.2f} jauh di bawah rata-rata")

        st.markdown(f"- **Persentase Diabetes:** {diabetes_pct:.2f}%")
        st.markdown(f"- **Tingkat HbA1c:** {hba1c_level.capitalize()}")
        st.markdown(f"- **Tingkat Glukosa:** {glucose_level.capitalize()}")
        st.markdown(f"- **Interpretasi:** Risiko diabetes pada cluster ini sebesar {diabetes_pct:.2f}%.")
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
        model = LogisticRegression(max_iter=1000)
        model.fit(X_scaled, y)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        st.subheader("Hasil Prediksi")
        st.markdown(f"**Prediksi:** {'Diabetes' if pred == 1 else 'Tidak Diabetes'}")
        st.markdown(f"**Probabilitas:** {prob:.2f}")
