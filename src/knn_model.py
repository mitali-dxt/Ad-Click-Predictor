import numpy as np
import logging
import sys
from collections import Counter 

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("knn_predictor.log", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class MyKNearestNeighbors:
    """
    Implements the k-Nearest Neighbors algorithm from scratch.
    """

    def __init__(self, k=3):
        """Initializes the model with the 'k' value."""
        logger.info(f"Initializing k-NN model with k={k}")
        try:
            if k <= 0:
                raise ValueError("k is not a positive integer.")
            self.k = k
            self.X_train = None
            self.y_train = None
            logger.info(f"Model initialized with k as {self.k}.")
        except ValueError as e:
            logger.exception(f"Failed to initialize model: {e}")
            raise

    def _predict_single(self, x_new):
        """Helper function to predict one new sample."""

        # 1. Calculate Euclidean Distances 
        distances = [np.linalg.norm(x_new - x_train) for x_train in self.X_train]

        # 2. Get k-Nearest Neighbors
        k_nearest_indices = np.argsort(distances)[:self.k]

        # 3. Get the Labels
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]

        # 4. Vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def fit(self, X, y):
        """
        "Trains" the k-NN model.
        """
        logger.info(f"Starting fit() on data with X shape {X.shape}")
        try:
            if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
                raise TypeError("Input must be NumPy arrays.")
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"Mismatch: X has {X.shape[0]} samples, y has {y.shape[0]} samples.")
            if self.k > X.shape[0]:
                raise ValueError(f"k ({self.k}) cannot be greater than the number "
                                 f"of training samples ({X.shape[0]}).")

            self.X_train = X
            self.y_train = y
            logger.info(f"fit() complete. Stored {X.shape[0]} training samples.")

        except (ValueError, TypeError) as e:
            logger.exception(f"Error during fit(): {e}")
            raise
        except Exception as e:
            logger.exception(f"An unexpected error occurred during fit(): {e}")
            raise

    def predict(self, X_new):
        """
        Predicts class labels for new data.
        """
        logger.info(f"Starting predict() on {X_new.shape[0]} new samples.")
        try:
            if self.X_train is None or self.y_train is None:
                raise RuntimeError("Model has not been trained yet.")

            if X_new.ndim == 1:
                X_new = X_new.reshape(1, -1)

            if X_new.shape[1] != self.X_train.shape[1]:
                raise ValueError(f"Input data has {X_new.shape[1]} features, "
                                 f"but model was trained on {self.X_train.shape[1]} features.")

            if self.X_train.shape[0] > 5000:
                logger.warning(f"Predict is iterating over {self.X_train.shape[0]} samples. "
                               "This may be slow.")

            predictions = []
            for x in X_new:
                try:
                    predictions.append(self._predict_single(x))
                except Exception as e:
                    logger.exception(f"Failed to predict sample {x}. Appending None.")
                    predictions.append(None)

            logger.info(f"predict() complete. Returning {len(predictions)} predictions.")
            return np.array(predictions)

        except (ValueError, RuntimeError) as e:
            logger.exception(f"Error during predict(): {e}")
            raise