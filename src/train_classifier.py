"""
train_classifier.py
Phase 2: Training the One-Class Models for RF Fingerprinting Authentication.

This module implements a Hybrid Statistical-Autoencoder framework for Open Set Recognition (OSR).
It incorporates the newest research methods (2024-2025) including:
1. Extreme Value Theory (EVT) via Peaks-Over-Threshold (POT) for robust thresholding.
2. Empirical-Cumulative-distribution-based Outlier Detection (ECOD).
3. Standard One-Class SVM (OC-SVM) as a baseline.

Requirements:
    - torch, numpy, scipy, scikit-learn, pyod, matplotlib
    - A pre-trained Autoencoder model (simulated via mock class if not provided)

Author: [Expert Persona]
Date: December 2025
Reference: COMP3020 Project / Advanced Research on Open Set Recognition
"""

import os
import pickle
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import genpareto

# PyOD Libraries for modern Anomaly Detection
from pyod.models.ecod import ECOD
from pyod.models.ocsvm import OCSVM  # Wrapper around sklearn OC-SVM
from sklearn.preprocessing import StandardScaler

# ==========================================
# Configuration & Hyperparameters
# ==========================================
CONFIG = {
    'random_seed': 42,
    'latent_dim': 64,
    'input_len': 256,   # 2 channels * 128 samples flattened
    'batch_size': 128,
    'tail_size': 0.1,   # EVT: Fraction of data to consider as 'tail' (top 10%)
    'evt_frr': 0.01,    # EVT: Target False Rejection Rate (1%)
    'contamination': 0.01, # PyOD: Expected proportion of outliers in training (noise)
    'model_save_dir': './saved_models/',
    'viz_save_dir': './visualizations/'
}

# Ensure reproducibility
torch.manual_seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])

# ==========================================
# 1. Mock Autoencoder Definition (Phase 1)
# ==========================================
class RFAutoencoder(nn.Module):
    """
    A placeholder for the Deep Autoencoder trained in Phase 1.
    In a real deployment, import this from the Phase 1 `models.py`.
    """
    def __init__(self, input_len=256, latent_dim=64):
        super(RFAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_len, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, latent_dim),
            nn.Tanh() # Latent space bounded [-1, 1]
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, input_len),
            nn.Tanh() 
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

# ==========================================
# 2. EVT / Peaks-Over-Threshold Class
# ==========================================
class EVTModeler:
    """
    Implements Extreme Value Theory (EVT) using the Peaks-Over-Threshold (POT) method.
    Fits a Generalized Pareto Distribution (GPD) to the tail of reconstruction errors.
    """
    def __init__(self, tail_size=0.1):
        self.tail_size = tail_size
        self.threshold_u = None     # The boundary between 'body' and 'tail'
        self.params_gpd = None      # (shape_xi, loc, scale_sigma)
        self.decision_thresh = None # The final operational threshold

    def fit(self, scores):
        """
        Fits the GPD to the top `tail_size` fraction of scores.
        Args:
            scores (np.array): 1D array of reconstruction errors (MSE).
        """
        # 1. Determine 'u': The threshold where the tail begins
        # We use the empirical quantile.
        self.threshold_u = np.quantile(scores, 1 - self.tail_size)
        
        # 2. Extract Exceedances: Data points strictly greater than u
        exceedances = scores[scores > self.threshold_u]
        
        # 3. Fit Generalized Pareto Distribution
        # Note: We fit to (x - u), so loc is effectively 0 for the fit function.
        # scipy.stats.genpareto returns (shape, loc, scale)
        # We fix location at 0 because we are modeling the *excess* over u.
        shape_xi, loc, scale_sigma = genpareto.fit(exceedances - self.threshold_u, floc=0)
        
        self.params_gpd = (shape_xi, self.threshold_u, scale_sigma)
        
        print(f" Fitted GPD on {len(exceedances)} exceedances.")
        print(f"      Threshold u: {self.threshold_u:.5f}")
        print(f"      Shape (xi):  {shape_xi:.5f} (Positive=Heavy Tail, Negative=Bounded)")
        print(f"      Scale (sig): {scale_sigma:.5f}")

    def calculate_threshold(self, target_frr=0.01):
        """
        Calculates the decision threshold for a specific False Rejection Rate.
        
        The formula is derived from the GPD survival function:
        P(X > x) = P(X > u) * P(X > x | X > u)
        FRR = tail_size * (1 + xi * (x - u) / sigma)^(-1/xi)
        
        Solving for x gives the threshold.
        """
        xi, u, sigma = self.params_gpd
        
        # If target FRR is greater than tail size, EVT doesn't apply (in the body).
        if target_frr > self.tail_size:
            print(" Target FRR > Tail Size. Returning empirical quantile.")
            return np.quantile(scores, 1 - target_frr)

        # Inverse calculation
        # Term inside bracket
        term = (target_frr / self.tail_size) ** (-xi) - 1
        calculated_threshold = u + (sigma / xi) * term
        
        self.decision_thresh = calculated_threshold
        return calculated_threshold

    def predict(self, scores):
        """Binary classification based on the calculated threshold."""
        if self.decision_thresh is None:
            raise ValueError("Run calculate_threshold() first.")
        # 1 = Anomaly (Reject), 0 = Normal (Accept) - matching PyOD convention
        return (scores > self.decision_thresh).astype(int)

# ==========================================
# 3. Training Manager Class
# ==========================================
class OneClassTrainer:
    def __init__(self, ae_model):
        self.ae = ae_model
        self.ae.eval() # Set to evaluation mode
        self.scaler = StandardScaler()
        
        # Initialize Models
        self.evt = EVTModeler(tail_size=CONFIG['tail_size'])
        self.ecod = ECOD(contamination=CONFIG['contamination'])
        self.ocsvm = OCSVM(kernel='rbf', contamination=CONFIG['contamination'])

    def extract_features(self, dataloader):
        """Helper to get Latent Z and Reconstruction Error from AE."""
        latents = []
        errors = []
        
        print("[Phase 2] Extracting features from legitimate data...")
        with torch.no_grad():
            for batch in dataloader:
                # Handle batch - could be tensor, list, or tuple
                if isinstance(batch, (list, tuple)):
                    x = batch[0]  # Get first element if batch is list/tuple
                else:
                    x = batch
                # Ensure x is a tensor
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x, dtype=torch.float32)
                x_flat = x.view(x.size(0), -1)
                
                # Forward pass
                recon, z = self.ae(x_flat)
                
                # Compute MSE per sample
                # loss shape: (batch_size, )
                loss = torch.mean((x_flat - recon) ** 2, dim=1)
                
                latents.append(z.cpu().numpy())
                errors.append(loss.cpu().numpy())
                
        return np.concatenate(latents), np.concatenate(errors)

    def train(self, dataloader):
        # 1. Feature Extraction
        Z_train, mse_train = self.extract_features(dataloader)
        
        # 2. Preprocessing
        # ECOD handles raw data well, but SVM needs scaling
        Z_scaled = self.scaler.fit_transform(Z_train)
        
        # 3. Train EVT Model (on 1D Errors)
        print("\n--- Training EVT Model ---")
        self.evt.fit(mse_train)
        evt_thresh = self.evt.calculate_threshold(target_frr=CONFIG['evt_frr'])
        print(f" EVT Decision Threshold (FRR={CONFIG['evt_frr']}): {evt_thresh:.6f}")
        
        # 4. Train ECOD (on Multidimensional Latents)
        print("\n--- Training ECOD Model ---")
        # ECOD is parameter-free, fitting is fast
        self.ecod.fit(Z_train) 
        print(f" ECOD Internal Threshold: {self.ecod.threshold_:.4f}")
        
        # 5. Train OC-SVM (Baseline)
        print("\n--- Training OC-SVM Baseline ---")
        self.ocsvm.fit(Z_scaled)
        print(f" OC-SVM fit complete.")
        
        # 6. Visualization
        self.visualize_training(mse_train, evt_thresh)
        
        return Z_train, mse_train

    def visualize_training(self, scores, threshold):
        """Plot the error distribution and the EVT cut-off."""
        if not os.path.exists(CONFIG['viz_save_dir']):
            os.makedirs(CONFIG['viz_save_dir'])
            
        plt.figure(figsize=(10, 6))
        sns.histplot(scores, bins=50, kde=True, color='blue', label='Legitimate Scores')
        plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'EVT Threshold (FRR {CONFIG["evt_frr"]})')
        plt.title('Reconstruction Error Distribution with EVT Boundary')
        plt.xlabel('Mean Squared Error (MSE)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(CONFIG['viz_save_dir'], 'evt_distribution.png'))
        plt.close()
        print(f"[Viz] Saved distribution plot to {CONFIG['viz_save_dir']}")

    def save_state(self):
        """Serialize all models for inference into a single joblib file."""
        if not os.path.exists(CONFIG['model_save_dir']):
            os.makedirs(CONFIG['model_save_dir'])
        
        # Prepare EVT state
        evt_state = {
            'threshold_u': self.evt.threshold_u,
            'params': self.evt.params_gpd,
            'decision_thresh': self.evt.decision_thresh
        }
        
        # Save all classifier components in a single joblib file
        classifier_model = {
            'scaler': self.scaler,
            'evt_model': evt_state,
            'ecod_model': self.ecod,
            'ocsvm_model': self.ocsvm
        }
        
        classifier_path = os.path.join(CONFIG['model_save_dir'], 'classifier_model.joblib')
        joblib.dump(classifier_model, classifier_path)
        
        print(f"\n[INFO] Classifier model saved to {classifier_path}")
        print(f"      Contains: scaler, evt_model, ecod_model, ocsvm_model")
    
    @staticmethod
    def load_state(model_dir=None):
        """Load all classifier components from a single joblib file."""
        if model_dir is None:
            model_dir = CONFIG['model_save_dir']
        
        classifier_path = os.path.join(model_dir, 'classifier_model.joblib')
        
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Classifier model not found at {classifier_path}")
        
        classifier_model = joblib.load(classifier_path)
        print(f"[INFO] Loaded classifier model from {classifier_path}")
        
        return classifier_model

# ==========================================
# 4. Main Execution
# ==========================================
def generate_mock_data(samples=5000):
    """
    Generates synthetic I/Q data for demonstration.
    Returns: TensorDataset
    """
    # Simulate Gaussian I/Q samples with a specific 'device' signature
    # Shape: [N, 2, 128] -> flattened to [N, 256]
    X = np.random.normal(0, 1, (samples, 256)).astype(np.float32)
    
    # Inject a deterministic fingerprint (e.g., DC offset + Scale imbalance)
    X[:, ::2] *= 1.1  # I-imbalance
    X += 0.05         # LO leakage
    
    return TensorDataset(torch.from_numpy(X))

if __name__ == "__main__":
    print("=== Starting Phase 2: One-Class Model Training ===")
    
    # 1. Load Pre-trained AE
    # In real usage: ae = RFAutoencoder(); ae.load_state_dict(...)
    ae_model = RFAutoencoder()
    print("Autoencoder initialized (Mock).")
    
    # 2. Load Data
    # In real usage: Load RadioML.pkl file here
    print("Generating synthetic legitimate data...")
    dataset = generate_mock_data(samples=5000)
    train_loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    # 3. Execute Training Pipeline
    trainer = OneClassTrainer(ae_model)
    trainer.train(train_loader)
    
    # 4. Save
    trainer.save_state()
    print("=== Phase 2 Complete ===")