import streamlit as st
import pandas as pd
import joblib
import os
import gdown
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Add your Google Drive file IDs here
model_ids = {
    "Random Forest": "1NGCNYiRrq4kN5u0XJfUe7Hk2Fh6K6gVK",
    "SVM": "1NDmKK6U8PHkWFk0zKSleQOC6vbId1C0d",
    "Neural Network": "1NBjolLQERJJOhwmkLtTJLiMhky-eCvcO"
}
# Function to download a file from Google Drive

def download_model_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        
#  Load all models after downloading

@st.cache_resource
def load_models():
    models = {}
    for name, file_id in model_ids.items():
        model_filename = f"{name}_model.pkl"
        download_model_from_drive(file_id, model_filename)
        models[name] = joblib.load(model_filename)
    return models

#  Title and Description
st.title("🌐 Cloud-Based Intrusion Detection System (IDS)")
st.write("""
Upload a CSV file with network traffic data. This app uses machine learning models
to detect unauthorized access and identify specific attack types.
""")

# Load models
models = load_models()

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload a test CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.subheader("📄 Data Preview")
        st.dataframe(data.head())
        
# Choose target column (label like attack_cat)
        target_column = st.selectbox("🎯 Select the label column", data.columns)

        X = pd.get_dummies(data.drop(columns=[target_column]))
        y = data[target_column]

        #  Choose which models to test
        selected_models = st.multiselect(
            "🤖 Choose models to test", list(models.keys()), default=list(models.keys())
        )

       #  Run predictions and display results
        if st.button("🚀 Run Predictions"):
            for name in selected_models:
                st.markdown(f"### 🔍 Results for `{name}`")
                model = models[name]

                # Align columns
                model_input_cols = model.feature_names_in_
                for col in model_input_cols:
                    if col not in X.columns:
                        X[col] = 0
                X_aligned = X[model_input_cols]

                # Predict
                preds = model.predict(X_aligned)

                # Result DataFrame
                result_df = X.copy()
                result_df["Actual"] = y
                result_df["Prediction"] = preds

                # ----------------------
                # 📊 Prediction Table
                # ----------------------
                st.subheader("📊 Prediction Results Table")
                st.dataframe(result_df)

                # ----------------------
                # 🥧 Pie Chart
                # ----------------------
                st.subheader("📈 Prediction Distribution")
                labels, counts = np.unique(preds, return_counts=True)
                fig, ax = plt.subplots()
                ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)

                # ----------------------
                # 📉 Confusion Matrix
                # ----------------------
                st.subheader("📉 Confusion Matrix (Actual vs Predicted)")
                cm = confusion_matrix(y, preds, labels=np.unique(y))
                cm_df = pd.DataFrame(cm, index=np.unique(y), columns=np.unique(y))

                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                st.pyplot(fig_cm)

                # 🔒 Unauthorized detection summary
                unauthorized = sum(result_df["Prediction"] != "Normal")
                st.info(f"🔐 {unauthorized} unauthorized access attempts detected out of {len(result_df)} samples.")

        # ------------------------------------------------------
        # 🔁 Simulate Real-Time Detection (Row by Row)
        # ------------------------------------------------------
        st.markdown("---")
        st.markdown("### 🔁 Simulate Real-Time Detection")

        realtime_index = st.number_input(
            "Enter row number to test (starting from 0)",
            min_value=0,
            max_value=len(data) - 1,
            value=0
        )

        row_data = data.iloc[realtime_index:realtime_index + 1].drop(columns=[target_column])
        row_label = data.iloc[realtime_index][target_column]
        row_data = pd.get_dummies(row_data)

        for name in selected_models:
            model = models[name]
            model_input_cols = model.feature_names_in_
            for col in model_input_cols:
                if col not in row_data.columns:
                    row_data[col] = 0
            row_data = row_data[model_input_cols]

            pred = model.predict(row_data)[0]
            st.success(f"**Model: {name}** → Prediction: `{pred}` | Actual: `{row_label}`")

    except Exception as e:
        st.error(f"❌ Error while processing: {e}")
