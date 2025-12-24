"""
metrics.py

Comprehensive Evaluation Module for Hybrid Autoencoder-Based IoT Authentication.
Implements standard biometric metrics (FAR, FRR, EER), advanced visualization (DET),
probability calibration (ECE), and Inductive Conformal Prediction (ICP).

This module is designed to support the research requirements of COMP3020,
integrating state-of-the-art methods for uncertainty quantification.

Author:
Date: October 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, Union
from sklearn.metrics import (
    roc_curve, auc, f1_score, precision_recall_curve, 
    confusion_matrix, DetCurveDisplay, brier_score_loss
)
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import brentq
from scipy.interpolate import interp1d

class IoTAuthenticationMetrics:
    """
    A unified class for evaluating One-Class Classifiers in an IoT security context.
    Manages metric calculation, visualization, and statistical calibration.
    """

    def __init__(self, legitimate_label: int = 1, anomaly_label: int = -1):
        """
        Initialize the metrics engine.

        Args:
            legitimate_label (int): The label used for legitimate devices (default: 1).
            anomaly_label (int): The label used for impostors (default: -1).
        """
        self.legit_label = legitimate_label
        self.anom_label = anomaly_label
        
        # Storage for Conformal Prediction Calibration Set
        self.calibration_scores = None 
        self.calibrator_model = None # For probability calibration (Platt/Isotonic)

    def _binarize_labels(self, y_true: np.ndarray) -> np.ndarray:
        """
        Internal utility to map custom labels to binary {0, 1} for sklearn.
        
        Mapping:
            Legitimate (Target) -> 1
            Anomaly (Impostor)  -> 0
            
        Args:
            y_true: Array of ground truth labels.
            
        Returns:
            np.ndarray: Binary labels.
        """
        return np.array([1 if y == self.legit_label else 0 for y in y_true])

    def compute_far_frr(self, y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> Dict[str, float]:
        """
        Compute False Acceptance Rate (FAR) and False Rejection Rate (FRR) 
        at a specific decision threshold.

        Args:
            y_true: Ground truth labels.
            y_scores: Anomaly scores (Assumption: Higher score = More Likelihood of Legitimate).
                      *NOTE*: If using Reconstruction Error where High = Anomaly, 
                      negate the scores before passing to this function.
            threshold: The cutoff value. Scores >= threshold are classified as Legitimate.

        Returns:
            dict: Dictionary containing FAR, FRR, TP, FN, FP, TN.
        """
        # Generate predictions based on threshold
        y_pred = np.where(y_scores >= threshold, self.legit_label, self.anom_label)
        
        # Calculate Confusion Matrix elements manually for clarity
        # True Positive: Legit classified as Legit
        tp = np.sum((y_pred == self.legit_label) & (y_true == self.legit_label))
        # False Negative: Legit classified as Anomaly (False Rejection)
        fn = np.sum((y_pred == self.anom_label) & (y_true == self.legit_label))
        # False Positive: Anomaly classified as Legit (False Acceptance)
        fp = np.sum((y_pred == self.legit_label) & (y_true == self.anom_label))
        # True Negative: Anomaly classified as Anomaly
        tn = np.sum((y_pred == self.anom_label) & (y_true == self.anom_label))
        
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        return {
            "FAR": far, 
            "FRR": frr, 
            "TP": int(tp), 
            "FN": int(fn), 
            "FP": int(fp), 
            "TN": int(tn)
        }

    def compute_eer(self, y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
        """
        Compute the Equal Error Rate (EER) using high-precision interpolation.
        
        The EER is the point where FAR = FRR. Since empirical data is discrete,
        an exact intersection often doesn't exist. This function interpolates
        between the closest points to approximate the true EER.

        Args:
            y_true: Ground truth labels.
            y_scores: Decision scores (Higher = Legitimate).

        Returns:
            eer (float): The approximated Equal Error Rate.
            threshold (float): The threshold at which EER occurs.
        """
        y_bin = self._binarize_labels(y_true)
        
        # Compute ROC points
        fpr, tpr, thresholds = roc_curve(y_bin, y_scores, pos_label=1)
        fnr = 1 - tpr
        
        # Find the index where the difference between FPR and FNR is minimized
        # This gives us a starting point for interpolation
        abs_diff = np.abs(fpr - fnr)
        min_idx = np.nanargmin(abs_diff)
        
        # If perfect overlap exists
        if abs_diff[min_idx] == 0:
            return fpr[min_idx], thresholds[min_idx]
            
        # Interpolation Strategy:
        # We create continuous functions of FPR and FNR with respect to the threshold
        # Then solve for threshold t where FPR(t) - FNR(t) = 0
        try:
            # Handle edge cases in thresholds (sklearn adds an arbitrary first threshold)
            valid_indices = np.where(np.isfinite(thresholds))
            if len(valid_indices) < 2:
                return 0.5, 0.0 # Fallback for degenerate cases
                
            interp_fpr = interp1d(thresholds[valid_indices], fpr[valid_indices], kind='linear', fill_value="extrapolate")
            interp_fnr = interp1d(thresholds[valid_indices], fnr[valid_indices], kind='linear', fill_value="extrapolate")
            
            # Define the function to find the root of
            def difference(t):
                return interp_fpr(t) - interp_fnr(t)
            
            # Define search bounds around the rough estimate
            # We look at the neighbors of the min_idx
            t_low = thresholds[min(min_idx + 1, len(thresholds)-1)]
            t_high = thresholds[max(min_idx - 1, 0)]
            
            # Ensure proper ordering for brentq
            if t_low > t_high:
                t_low, t_high = t_high, t_low
                
            # If bounds collapse (duplicate thresholds), widen slightly
            if t_low == t_high:
                t_low -= 1e-6
                t_high += 1e-6
                
            optimal_threshold = brentq(difference, t_low, t_high)
            eer_val = interp_fpr(optimal_threshold)
            
        except Exception as e:
            print(f" EER Interpolation failed: {e}. Returning discrete approximation.")
            eer_val = (fpr[min_idx] + fnr[min_idx]) / 2
            optimal_threshold = thresholds[min_idx]
            
        return float(eer_val), float(optimal_threshold)

    def plot_det_curve(self, y_true: np.ndarray, y_scores: np.ndarray, title: str = "DET Curve"):
        """
        Plot the Detection Error Trade-off (DET) Curve.
        
        Uses Log-Log scale to visualize performance in high-security (low FAR) regions.
        This provides a superior view compared to ROC for biometric applications.
        """
        y_bin = self._binarize_labels(y_true)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        DetCurveDisplay.from_predictions(y_bin, y_scores, ax=ax, name="Hybrid AE Model")
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("False Acceptance Rate (FAR) - Log Scale", fontsize=12)
        ax.set_ylabel("False Rejection Rate (FRR) - Log Scale", fontsize=12)
        ax.grid(which='both', linestyle='--', linewidth=0.5)
        plt.show()

    # ==========================================================
    # State-of-the-Art Method: Inductive Conformal Prediction
    # ==========================================================
    
    def fit_conformal(self, calibration_scores: np.ndarray):
        """
        Fit the Conformal Predictor.
        
        Stores the non-conformity scores of a held-out calibration set.
        Crucial: These scores must come from LEGITIMATE samples not seen during training.
        
        Args:
            calibration_scores: 1D array of anomaly scores (e.g., reconstruction error).
                                Assumption: HIGHER score = MORE ANOMALOUS.
        """
        self.calibration_scores = np.sort(np.asarray(calibration_scores))
        
    def predict_conformal(self, test_scores: Union[float, np.ndarray], epsilon: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Inductive Conformal Prediction to detect anomalies with statistical guarantees.
        
        Args:
            test_scores: Anomaly scores of new test samples (Higher = Anomalous).
            epsilon: Significance level (e.g., 0.05 for 95% confidence).
                     This bounds the False Rejection Rate at epsilon.
                     
        Returns:
            is_anomaly: Boolean array (True if p-value < epsilon).
            p_values: The calculated p-values for each sample.
        """
        if self.calibration_scores is None:
            raise ValueError("Conformal calibration set not found. Run fit_conformal() first.")
            
        test_scores = np.asarray(test_scores)
        if test_scores.ndim == 0:
            test_scores = test_scores[np.newaxis]
            
        n_cal = len(self.calibration_scores)
        p_values =
        
        # Calculate p-value for each test score
        # p(x) = (|{alpha_cal >= alpha_test}| + 1) / (n_cal + 1)
        # Using searchsorted for efficiency O(log N) instead of O(N) linear scan
        # searchsorted finds indices where elements should be inserted to maintain order.
        # Since calibration_scores is sorted ascending, elements to the right are >=.
        
        indices = np.searchsorted(self.calibration_scores, test_scores, side='left')
        n_higher = n_cal - indices
        
        p_values = (n_higher + 1) / (n_cal + 1)
        is_anomaly = p_values < epsilon
        
        return is_anomaly, p_values

    # ==========================================================
    # State-of-the-Art Method: Expected Calibration Error (ECE)
    # ==========================================================

    def calibrate_scores(self, y_val_true: np.ndarray, y_val_scores: np.ndarray):
        """
        Fit a probability calibration model (Isotonic Regression) to map 
        raw distances/errors to  probabilities.
        """
        y_bin = self._binarize_labels(y_val_true)
        self.calibrator_model = IsotonicRegression(out_of_bounds='clip')
        self.calibrator_model.fit(y_val_scores, y_bin)

    def compute_ece(self, y_true: np.ndarray, y_scores: np.ndarray, n_bins: int = 15) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        Measures how well the predicted probabilities match the empirical accuracy.
        Lower ECE = Better calibrated trust scores.
        
        Args:
            y_true: Ground truth labels.
            y_scores: Raw model scores (will be converted to probs if calibrator exists,
                      else assumed to be probabilities).
            n_bins: Number of bins for partitioning.
            
        Returns:
            float: The ECE value.
        """
        y_bin = self._binarize_labels(y_true)
        
        # Convert to probabilities if a calibrator has been fitted
        if self.calibrator_model is not None:
            probs = self.calibrator_model.predict(y_scores)
        else:
            # Assume scores are already normalized to 
            probs = y_scores
            if np.max(probs) > 1.0 or np.min(probs) < 0.0:
                print(" Scores outside  range and no calibrator fitted. ECE may be invalid.")
                probs = (probs - np.min(probs)) / (np.max(probs) - np.min(probs)) # Min-Max Fallback

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total_samples = len(y_bin)
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i+1]
            
            # Select samples in this bin
            if i == n_bins - 1:
                mask = (probs >= bin_lower) & (probs <= bin_upper)
            else:
                mask = (probs >= bin_lower) & (probs < bin_upper)
                
            n_in_bin = np.sum(mask)
            
            if n_in_bin > 0:
                current_acc = np.mean(y_bin[mask])
                current_conf = np.mean(probs[mask])
                ece += (n_in_bin / total_samples) * np.abs(current_acc - current_conf)
                
        return ece

# ==========================================================
# Example Usage Workflow
# ==========================================================
if __name__ == "__main__":
    print("Initializing IoT Authentication Metrics Module...")
    
    # 1. Simulate Data (as per RadioML context)
    # Legitimate (Device 1): High OCSVM scores (close to centroid)
    # Impostor (Device 6): Low OCSVM scores (far from centroid)
    rng = np.random.RandomState(42)
    
    # Generate 1000 Legit scores (Normal distribution centered at 5.0)
    legit_scores = rng.normal(loc=5.0, scale=1.0, size=1000)
    # Generate 1000 Impostor scores (Normal distribution centered at -2.0)
    impostor_scores = rng.normal(loc=-2.0, scale=2.0, size=1000)
    
    # Concatenate for testing
    y_test_scores = np.concatenate([legit_scores, impostor_scores])
    y_test_labels = np.concatenate([np.ones(1000), -1 * np.ones(1000)]) # 1 vs -1
    
    evaluator = IoTAuthenticationMetrics(legitimate_label=1, anomaly_label=-1)
    
    # 2. Compute Basic Metrics (EER)
    eer, thresh = evaluator.compute_eer(y_test_labels, y_test_scores)
    print(f"\n--- Standard Biometric Metrics ---")
    print(f"Equal Error Rate (EER): {eer*100:.2f}%")
    print(f"Optimal Threshold (at EER): {thresh:.4f}")
    
    # 3. Compute Metrics at Specific Operating Point (e.g., Threshold=2.0)
    op_metrics = evaluator.compute_far_frr(y_test_labels, y_test_scores, threshold=2.0)
    print(f"At threshold=2.0: FAR={op_metrics*100:.2f}%, FRR={op_metrics*100:.2f}%")
    
    # 4. State-of-the-Art: Conformal Prediction
    print(f"\n--- Conformal Prediction (ICP) ---")
    # Simulate a held-out Calibration Set (Legitimate devices only)
    # Note: For Conformal, we usually use 'Non-Conformity' scores (Lower = Better).
    # If using OCSVM scores (Higher = Better), we negate them.
    cal_scores = -1 * rng.normal(loc=5.0, scale=1.0, size=500) 
    
    evaluator.fit_conformal(cal_scores)
    
    # Test on a new sample (e.g., score = 0.0, which is ambiguous)
    # Remember to negate if using the same logic
    test_val = -1 * 0.0 
    is_anom, p_val = evaluator.predict_conformal(test_val, epsilon=0.05)
    print(f"Test Sample (Score=0.0): p-value = {p_val:.4f}")
    print(f"Is Anomaly at 95% confidence? {'Yes' if is_anom else 'No'}")
    
    # 5. State-of-the-Art: Calibration Error
    print(f"\n--- Trust Calibration (ECE) ---")
    # First, fit the calibrator on validation data
    val_scores = np.concatenate([rng.normal(5,1,500), rng.normal(-2,2,500)])
    val_labels = np.concatenate([np.ones(500), -1*np.ones(500)])
    evaluator.calibrate_scores(val_labels, val_scores)
    
    ece = evaluator.compute_ece(y_test_labels, y_test_scores)
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print("(Lower ECE indicates the probability scores are more reliable)")