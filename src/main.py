import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from knn_predictor import MyKNearestNeighbors

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("--- Main script execution started ---")

    try:
        # Creating Synthetic User Data
        logger.info("Generating synthetic data...")
        np.random.seed(42)
        # [Age, Time_on_Site, Estimated_Income]
        X_class0 = np.random.normal(loc=[25, 5, 40000], scale=[5, 2, 10000], size=(100, 3))
        X_class1 = np.random.normal(loc=[45, 15, 75000], scale=[8, 4, 15000], size=(100, 3))
        X = np.vstack([X_class0, X_class1])
        y = np.array([0]*100 + [1]*100)

        logger.info("Splitting and scaling data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # Fit the scaler on the training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # transform the test data
        X_test_scaled = scaler.transform(X_test)
        logger.info(f"Data split: {X_train_scaled.shape[0]} training, {X_test_scaled.shape[0]} test samples.")

        logger.info("--- 1. Training Model (k=5) ---")
        model = MyKNearestNeighbors(k=5)
        model.fit(X_train_scaled, y_train)

        # Predict on test set
        y_pred_test = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred_test)

        logger.info(f"Test Set Accuracy (k=5): {accuracy:.4f}")
        print(f"\n[INFO] Model is trained. Test Set Accuracy: {accuracy:.4f}\n")

        # Check prediction on new data
        logger.info("Predicting on new data")

        # User 1: [Age: 30, Time: 8 min, Income: $45,000] (Should be 'No Click' - 0)
        # User 2: [Age: 50, Time: 20 min, Income: $80,000] (Should be 'Click' - 1)
        X_new = np.array([
            [30, 8, 45000],
            [50, 20, 80000]
        ])

        X_new_scaled = scaler.transform(X_new)

        logger.info(f"New data scaled using the trained scaler.")

        # Now, predict
        new_predictions = model.predict(X_new_scaled)

        # --- 5. Display the New Predictions ---
        labels = {0: "No Click", 1: "Click"}

        print("--- New User Predictions ---")
        for i, user_data in enumerate(X_new):
            prediction_label = labels[new_predictions[i]]
            print(f"  User with data {user_data} -> Predicted: {prediction_label} ({new_predictions[i]})")

        logger.info(f"Predictions complete: {new_predictions}")

    except Exception as e:
        logger.exception("Main execution FAILED unexpectedly.")
        print(f"\n[FAILURE] Main script failed: {e}\n")