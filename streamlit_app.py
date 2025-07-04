import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 🔹 Title and Instructions
# -----------------------------
st.title("Cloud-Based Intrusion Detection System (IDS)")
st.write("""
Upload a CSV file with network data and choose a machine learning model 
to detect unauthorized access. This system uses models trained on the UNSW-NB15 dataset.
""")

# -----------------------------
# 🔹 Load Models
# -----------------------------
@st.cache_resource
def load_models():
    models = {
        "Random Forest": joblib.load("Random Forest_model.pkl"),
        "SVM": joblib.load("SVM_model.pkl"),
        "Neural Network": joblib.load("Neural Network_model.pkl")
    }
    return models

models = load_models()

# -----------------------------
# 🔹 Upload CSV File
# -----------------------------
uploaded_file = st.file_uploader("Upload your test CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Load uploaded data
        data = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data Preview")
        st.write(data.head())

        # -----------------------------
        # 🔹 Target column selection
        # -----------------------------
        target_column = st.selectbox("Select the target column (label)", data.columns)

        # Split features and label
        X = pd.get_dummies(data.drop(columns=[target_column]))
        y = data[target_column]

        # -----------------------------
        # 🔹 Model Selection
        # -----------------------------
        selected_models = st.multiselect(
            "Choose models to test", list(models.keys()), default=list(models.keys())
        )

        # -----------------------------
        # 🔹 Prediction Section
        # -----------------------------
        if st.button("Run Predictions"):
            for name in selected_models:
                st.markdown(f"### 🔍 Results for `{name}`")
                model = models[name]

                # Align input columns with model training
                model_input_cols = model.feature_names_in_
                missing_cols = set(model_input_cols) - set(X.columns)
                for col in missing_cols:
                    X[col] = 0
                X_aligned = X[model_input_cols]

                # Predict
                preds = model.predict(X_aligned)

                # Combine results into a dataframe
                result_df = X.copy()
                result_df["Actual"] = y
                result_df["Prediction"] = preds

                # Show results table
                st.subheader("📊 Prediction Results Table")
                st.dataframe(result_df)

                # Show pie chart
                st.subheader("📈 Prediction Distribution Pie Chart")
                labels, counts = np.unique(preds, return_counts=True)
                fig, ax = plt.subplots()
                ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)

                # Summary
                unauthorized = sum(result_df["Prediction"] != "Normal")
                total = len(result_df)
                st.info(f"🔒 {unauthorized} unauthorized access attempts detected out of {total} samples.")

        # -----------------------------
        # 🔹 Real-Time Detection Simulation
        # -----------------------------
        st.markdown("---")
        st.markdown("### 🔁 Simulate Real-Time Detection (One Row at a Time)")

        realtime_index = st.number_input("Enter row number to simulate (starting at 0)", min_value=0, max_value=len(data)-1, value=0)

        # Extract and preprocess one row
        row_data = data.iloc[realtime_index:realtime_index+1].drop(columns=[target_column])
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
            st.success(f"**Model: {name}** — Prediction: `{pred}` | Actual: `{row_label}`")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
