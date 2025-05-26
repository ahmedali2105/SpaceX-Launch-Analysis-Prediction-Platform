import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import sys

# Add the parent directory to the system path to allow importing data_fetcher
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_fetcher import load_and_preprocess_data, PROCESSED_DATA_PATH

# Define the path for saving the model
MODELS_DIR = 'models'
MODEL_PATH = os.path.join(MODELS_DIR, 'spacex_launch_prediction_model.joblib')
ENCODER_PATH = os.path.join(MODELS_DIR, 'feature_columns.joblib') # To save columns for consistent prediction

def train_model(df_model):
    """
    Trains a classification model (Logistic Regression and RandomForest) on the preprocessed data.
    Evaluates the models and saves the best performing one.
    """
    if df_model.empty or 'launch_success' not in df_model.columns:
        print("DataFrame is empty or 'launch_success' column is missing. Cannot train model.")
        return None, None

    X = df_model.drop('launch_success', axis=1)
    y = df_model['launch_success']

    # Store feature columns for consistent prediction later
    feature_columns = X.columns.tolist()
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Initialize models
    log_reg_model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    models = {
        "Logistic Regression": log_reg_model,
        "Random Forest": rf_model
    }

    best_model = None
    best_accuracy = -1
    best_model_name = ""

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] # Probability of success

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = name

        except Exception as e:
            print(f"Error training {name}: {e}")

    if best_model:
        print(f"\nBest model: {best_model_name} with Accuracy: {best_accuracy:.4f}")
        # Create models directory if it doesn't exist
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        joblib.dump(feature_columns, ENCODER_PATH) # Save feature columns
        print(f"Best model saved to {MODEL_PATH}")
        print(f"Feature columns saved to {ENCODER_PATH}")
        return best_model, feature_columns
    else:
        print("No model was successfully trained.")
        return None, None

def load_model_and_features():
    """
    Loads the trained model and feature columns.
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            feature_columns = joblib.load(ENCODER_PATH)
            print("Model and feature columns loaded successfully.")
            return model, feature_columns
        except Exception as e:
            print(f"Error loading model or feature columns: {e}")
            return None, None
    else:
        print("Model or feature columns not found. Please train the model first.")
        return None, None

if __name__ == "__main__":
    # This block runs when model_trainer.py is executed directly
    print("Running model_trainer.py directly...")
    df_display, df_model = load_and_preprocess_data()
    if not df_model.empty:
        trained_model, feature_cols = train_model(df_model)
        if trained_model:
            print("\nModel training complete.")
        else:
            print("\nModel training failed.")
    else:
        print("No preprocessed data available for training.")
