"""
classifier.py

This module implements the One-Class Classification stage of the Hybrid 
Statistical-Autoencoder Model for IoT Node Authentication.

It accepts latent representations (embeddings) from the Autoencoder and 
performs anomaly detection to distinguish between legitimate devices 
and impostors.

Supported Algorithms:
1. OC-SVM (One-Class Support Vector Machine) - Proposed Baseline
2. Robust Gaussian (Elliptic Envelope / Mahalanobis with MCD) - Proposed Baseline
3. Isolation Forest - Proposed Baseline
4. ECOD (Empirical Cumulative Distribution-based Outlier Detection) - SOTA Addition (2024)
5. Deep SVDD (Deep Support Vector Data Description) - SOTA Addition (2024)

Dependencies: scikit-learn, pyod, joblib, numpy
"""

import numpy as np
import joblib
import logging
from abc import ABC, abstractmethod
from typing import Union, Optional

# Scikit-learn implementations for baselines
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.base import BaseEstimator

# PyOD implementations for State-of-the-Art methods (2024-2025)
# Using PyOD is recommended for advanced methods not present in sklearn
try:
    from pyod.models.ecod import ECOD
    from pyod.models.deep_svdd import DeepSVDD
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False
    logging.warning("PyOD library not found. ECOD and DeepSVDD will be unavailable. "
                    "Install via `pip install pyod`.")

# Configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaseOneClassClassifier(ABC):
    """
    Abstract Base Class for all one-class classifiers to ensure consistent API.
    This adheres to the Strategy Pattern, allowing easy swapping of algorithms.
    """
    
    @abstractmethod
    def train(self, X: np.ndarray):
        """
        Train the model on legitimate latent vectors.
        
        Args:
            X (np.ndarray): Latent vectors of shape (n_samples, latent_dim).
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Returns:
            np.ndarray: +1 for Legitimate, -1 for Anomaly/Impostor
        """
        pass

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Return raw anomaly scores.
        Convention: Higher scores indicate higher likelihood of being an anomaly.
        """
        pass

    def save(self, filepath: str):
        """Persist the model using joblib."""
        try:
            joblib.dump(self.model, filepath)
            logging.info(f"Model saved to {filepath}")
        except Exception as e:
            logging.error(f"Failed to save model: {e}")

    def load(self, filepath: str):
        """Load a persisted model."""
        try:
            self.model = joblib.load(filepath)
            logging.info(f"Model loaded from {filepath}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")


class OCSVMClassifier(BaseOneClassClassifier):
    """
    Wrapper for Scikit-Learn One-Class SVM.
    Effective for non-linear boundaries in latent space.
    
    Parameters from proposal:
    - Kernel: RBF (Radial Basis Function)
    - Nu: Controls the fraction of outliers (tunable)
    """
    def __init__(self, kernel: str = 'rbf', nu: float = 0.1, gamma: str = 'scale'):
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        self.name = "OC-SVM"

    def train(self, X: np.ndarray):
        logging.info(f"Training {self.name} on {X.shape} samples...")
        self.model.fit(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # sklearn returns 1 for inlier, -1 for outlier
        return self.model.predict(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        # decision_function returns positive for inliers, negative for outliers.
        # We invert it so higher values indicate anomalies for consistency.
        return -self.model.decision_function(X)


class IsolationForestClassifier(BaseOneClassClassifier):
    """
    Wrapper for Scikit-Learn Isolation Forest.
    Efficient for high-dimensional latent spaces and resistant to overfitting.
    
    Why use it:
    - Handles high-dimensional latent vectors better than distance-based methods.
    - O(n log n) complexity makes it faster than OC-SVM (O(n^3)).
    """
    def __init__(self, n_estimators: int = 100, contamination: Union[float, str] = 0.1, random_state: int = 42):
        self.model = IsolationForest(n_estimators=n_estimators, 
                                     contamination=contamination, 
                                     random_state=random_state,
                                     n_jobs=-1) # Use all CPU cores
        self.name = "Isolation Forest"

    def train(self, X: np.ndarray):
        logging.info(f"Training {self.name} on {X.shape} samples...")
        self.model.fit(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        # decision_function returns negative anomaly score. 
        # Inverting so higher = more anomalous.
        return -self.model.decision_function(X)


class GaussianClassifier(BaseOneClassClassifier):
    """
    Wrapper for Scikit-Learn EllipticEnvelope (Robust Mahalanobis Distance).
    Uses Minimum Covariance Determinant (MCD) for robustness against noise.
    
    Why use it:
    - Standard Gaussian models fail if training data has noise.
    - MCD robustly estimates parameters even with 50% contamination.
    """
    def __init__(self, contamination: float = 0.1, support_fraction: float = None):
        # assume_centered=False means we estimate the mean (center) as well.
        self.model = EllipticEnvelope(contamination=contamination, 
                                      support_fraction=support_fraction,
                                      assume_centered=False)
        self.name = "Robust Gaussian (MCD)"

    def train(self, X: np.ndarray):
        logging.info(f"Training {self.name} on {X.shape} samples...")
        self.model.fit(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        # Returns Mahalanobis distance directly
        return self.model.mahalanobis(X)


class ECODClassifier(BaseOneClassClassifier):
    """
    Wrapper for PyOD ECOD (Empirical Cumulative Distribution-based Outlier Detection).
    SOTA (2022-2025) method: Parameter-free, fast, and highly effective for embeddings.
    
    Why use it:
    - Does not assume a distribution (unlike Gaussian).
    - No hyperparameters to tune (unlike OC-SVM).
    - Extremely fast training and inference.
    """
    def __init__(self, contamination: float = 0.1):
        if not PYOD_AVAILABLE:
            raise ImportError("PyOD is required for ECOD.")
        self.model = ECOD(contamination=contamination)
        self.name = "ECOD"

    def train(self, X: np.ndarray):
        logging.info(f"Training {self.name} on {X.shape} samples...")
        self.model.fit(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # PyOD returns 0 for inliers, 1 for outliers.
        # We map this to sklearn standard: 1 for Legitimate, -1 for Anomaly
        labels = self.model.predict(X)
        return np.where(labels == 0, 1, -1)

    def score(self, X: np.ndarray) -> np.ndarray:
        # PyOD decision_function returns raw anomaly scores (higher = outlier)
        return self.model.decision_function(X)


class DeepSVDDClassifier(BaseOneClassClassifier):
    """
    Wrapper for Deep SVDD (using PyOD implementation).
    Maps data into a minimum volume hypersphere.
    
    Why use it:
    - Jointly optimizes the embedding and the boundary (if using end-to-end).
    - Here, applied to fixed embeddings to define a tight spherical boundary.
    """
    def __init__(self, contamination: float = 0.1, use_ae: bool = False, epochs: int = 50):
        if not PYOD_AVAILABLE:
            raise ImportError("PyOD is required for Deep SVDD.")
        # Note: PyOD's DeepSVDD can use an internal AE for initialization.
        # Since we already have embeddings, we train a lightweight SVDD head.
        self.model = DeepSVDD(contamination=contamination, 
                              use_ae=use_ae, 
                              epochs=epochs, 
                              verbose=0)
        self.name = "Deep SVDD"

    def train(self, X: np.ndarray):
        logging.info(f"Training {self.name} on {X.shape} samples...")
        self.model.fit(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        labels = self.model.predict(X)
        return np.where(labels == 0, 1, -1)

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.model.decision_function(X)


class ClassifierFactory:
    """
    Factory class to instantiate classifiers based on configuration names.
    This allows easy switching between models in the main pipeline.
    """
    @staticmethod
    def get_classifier(name: str, **kwargs) -> BaseOneClassClassifier:
        name = name.lower()
        if name == 'ocsvm':
            return OCSVMClassifier(**kwargs)
        elif name == 'isolation_forest':
            return IsolationForestClassifier(**kwargs)
        elif name == 'gaussian':
            return GaussianClassifier(**kwargs)
        elif name == 'ecod':
            return ECODClassifier(**kwargs)
        elif name == 'deep_svdd':
            return DeepSVDDClassifier(**kwargs)
        else:
            raise ValueError(f"Unknown classifier type: {name}")

# --- Example Usage Logic ---
if __name__ == "__main__":
    # Simulate Latent Vectors (e.g., from the trained Autoencoder)
    # 64 dimensions as per proposal
    rng = np.random.RandomState(42)
    
    # Generate synthetic 'legitimate' data (clustered near 0)
    X_train_latent = 0.3 * rng.randn(1000, 64) 
    
    # Generate test data: mix of legitimate and random noise (impostors)
    X_test_latent = np.vstack([
        0.3 * rng.randn(200, 64),               # Legitimate test
        np.random.uniform(low=-4, high=4, size=(50, 64)) # Impostor/Anomalies
    ])

    # 1. Initialize Classifier (Swap 'ocsvm' for 'ecod' or 'deep_svdd')
    classifier_type = 'ocsvm' 
    print(f"--- Initializing {classifier_type} ---")
    
    try:
        clf = ClassifierFactory.get_classifier(classifier_type, nu=0.05)

        # 2. Train
        clf.train(X_train_latent)

        # 3. Predict
        predictions = clf.predict(X_test_latent)
        scores = clf.score(X_test_latent)

        # 4. Evaluation
        n_legitimate_test = 200
        n_impostors = 50
        
        legit_preds = predictions[:n_legitimate_test]
        impostor_preds = predictions[n_legitimate_test:]

        print(f"Legitimate Accuracy: {np.sum(legit_preds == 1) / n_legitimate_test:.2%}")
        print(f"Impostor Detection Rate: {np.sum(impostor_preds == -1) / n_impostors:.2%}")

        # 5. Save Model
        clf.save("auth_model_v1.pkl")
        
    except Exception as e:
        logging.error(f"Execution failed: {e}")