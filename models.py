# %% 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# %% 


# %%
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return data
    except FileNotFoundError:
        logging.error("File not found. Please check the file path.")
        return None
    except pd.errors.EmptyDataError:
        logging.error("No data found in the file.")
        return None
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None
# %%

# %%
def preprocess_data(data):
    try:
        if "attack_cat" not in data.columns:
            raise ValueError("The column 'attack_cat' is missing in the dataset.")
        
        X = data.drop(columns=["attack_cat"])
        y = data["attack_cat"]

        # Convert all categorical columns to numeric
        X = pd.get_dummies(X)

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y  # Stratify to maintain class distribution
        )
        logging.info("Data preprocessed successfully using attack_cat labels.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"An error occurred during preprocessing: {e}")
        return None, None, None, None


# %%

# %%
def train_models(X_train, y_train):
    models = {
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "Neural Network": MLPClassifier(max_iter=500)
    }
    
    trained_models = {}
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
            logging.info(f"{name} model trained successfully.")
        except Exception as e:
            logging.error(f"An error occurred while training {name}: {e}")
    
    return trained_models

# %%

#%%
def evaluate_models(models, X_test, y_test):
    for name, model in models.items():
        try:
            predictions = model.predict(X_test)
            print(f"Evaluation for {name}:")
            print(confusion_matrix(y_test, predictions))
            # Use zero_division parameter to handle undefined precision
            print(classification_report(y_test, predictions, zero_division=0))
        except Exception as e:
            logging.error(f"An error occurred while evaluating {name}: {e}")
# %%
# %%
def save_models(models):
    for name, model in models.items():
        try:
            joblib.dump(model, f"{name}_model.pkl")
            logging.info(f"{name} model saved successfully.")
        except Exception as e:
            logging.error(f"An error occurred while saving {name}: {e}")
# %%

# %%
def save_test_sample(X_test, y_test, output_path="test_sample.csv", sample_size=50):
    try:
        test_sample = X_test.copy()
        test_sample["attack_cat"] = y_test.values
        sample = test_sample.sample(n=sample_size, random_state=42)
        sample.to_csv(output_path, index=False)
        logging.info(f"Sample test data saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save test sample: {e}")
# %%
def main(file_path):
    data = load_data(file_path)
    if data is not None:
        X_train, X_test, y_train, y_test = preprocess_data(data)
        if X_train is not None:
            models = train_models(X_train, y_train)
            evaluate_models(models, X_test, y_test)
            save_models(models)
            save_test_sample(X_test, y_test)

if __name__ == "__main__":
    main("data/UNSW_NB15_training-set.csv")
# %%