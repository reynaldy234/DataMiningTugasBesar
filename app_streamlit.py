# Tampilkan missing values
st.subheader("❗ Missing Values")
missing_df = df.isnull().sum().reset_index()
missing_df.columns = ["Fitur", "Jumlah Missing"]
missing_df = missing_df[missing_df["Jumlah Missing"] > 0]

if missing_df.empty:
    st.success("Tidak ada missing values dalam dataset.")
else:
    st.dataframe(missing_df)

# Visualisasi boxplot untuk fitur numerik
st.subheader("📦 Deteksi Outlier (Boxplot Fitur Numerik)")
num_cols = ["age", "bmi", "blood_glucose_level", "HbA1c_level"]
fig_box, axes = plt.subplots(1, len(num_cols), figsize=(16, 4))

for i, col in enumerate(num_cols):
    sns.boxplot(y=df[col], ax=axes[i], color="skyblue")
    axes[i].set_title(col)
    axes[i].set_ylabel("")

st.pyplot(fig_box)
