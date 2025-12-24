import h5py
import numpy as np
import os
import gc
import argparse
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
# Get the project root directory
# File structure: ML Project/src/utils/preprocessing
# SCRIPT_DIR = ML Project/src/utils (1 level up from file)
# Need to go up 2 more levels: utils -> src -> ML Project
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Go up 2 levels from utils
RAW_FILE_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw.hdf5')  # Input file
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')    # Output directory

# SNR Threshold: Train only on signals with SNR >= 0 dB to ensure
# the Autoencoder learns clear features, not noise.
MIN_TRAIN_SNR = 0  

# Frame Slicing: Slice 1024 samples into 8 frames of 128.
# This increases data volume and reduces input dimensionality for the AE.
SLICE_LENGTH = 128  

# Virtual Device Definition:
# We treat specific modulation classes as "Legitimate" devices.
# Classes 0-4 are typically digital modulations in RadioML.
# 0:OOK, 1:4ASK, 2:8ASK, 3:BPSK, 4:QPSK
LEGITIMATE_CLASSES = [0, 1, 2, 3, 4]

# Random Seed for scientific reproducibility
SEED = 42

def load_dataset_metadata(file_path):
    """
    Opens the HDF5 file and retrieves metadata without loading the full
    dataset into RAM. This is crucial for large files (20GB+).
    """
    file_path = os.path.abspath(file_path)  # Ensure absolute path
    if not os.path.exists(file_path):
        # Provide helpful error message with calculated paths
        print(f"[ERROR] Calculated file path: {file_path}")
        print(f"[ERROR] Project root: {PROJECT_ROOT}")
        print(f"[ERROR] Script directory: {SCRIPT_DIR}")
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    
    print(f"[INFO] Inspecting dataset: {file_path}")
    hf = h5py.File(file_path, 'r')
    
    # Verify keys exist
    keys = list(hf.keys())
    print(f"[INFO] Found HDF5 Keys: {keys}")
    
    # Get shapes
    n_samples = hf['X'].shape
    print(f"[INFO] Total Samples: {n_samples}")
    
    return hf, n_samples

def process_chunk(hf, indices, slice_len=128):
    """
    Loads a specific chunk of data based on indices, performs slicing,
    and applies Instance Normalization.
    """
    # 1. Load specific indices (requires sorting for h5py efficiency)
    indices = np.sort(indices)
    
    # Note: h5py fancy indexing can be slow. For max speed, we might load
    # larger blocks and filter in memory, but here we prioritize RAM safety.
    X_chunk = hf['X'][indices] # Shape: (N, 1024, 2)
    
    # 2. Sequential Slicing (Data Augmentation)
    # Reshape (N, 1024, 2) -> (N, 8, 128, 2) -> (N*8, 128, 2)
    n_samples, orig_len, n_channels = X_chunk.shape
    n_slices = orig_len // slice_len
    
    X_sliced = X_chunk[:, :n_slices*slice_len, :].reshape(
        n_samples, n_slices, slice_len, n_channels
    )
    X_flat = X_sliced.reshape(-1, slice_len, n_channels)
    
    # 3. Instance Normalization (L2 Norm per sample)
    # Calculate energy of each 128-sample frame
    # Shape: (N*8, 1, 1)
    energies = np.linalg.norm(X_flat, axis=(1, 2), keepdims=True)
    
    # Avoid division by zero
    energies[energies == 0] = 1.0
    
    X_norm = X_flat / energies
    
    return X_norm

def main():
    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[INFO] Created output directory: {OUTPUT_DIR}")

    # 1. Open File & Load Metadata
    hf, n_samples = load_dataset_metadata(RAW_FILE_PATH)
    
    # 2. Load Labels (Y) and SNR (Z) first to determine filter masks
    # These are small enough to fit in RAM (approx 200MB)
    print("[INFO] Loading Labels (Y) and SNR (Z) for filtering...")
    Y_all = hf['Y'][:] 
    Z_all = hf['Z'][:]
    
    # Convert One-Hot Y to indices
    Y_indices = np.argmax(Y_all, axis=1)
    Z_values = np.squeeze(Z_all)
    
    # 3. Define Masks
    # Mask 1: Signal Quality (SNR >= 0dB)
    snr_mask = Z_values >= MIN_TRAIN_SNR
    
    # Mask 2: Legitimate Identity (Classes 0-4)
    legit_mask = np.isin(Y_indices, LEGITIMATE_CLASSES)
    
    # 4. Partition Indices
    # Training Data: Must be Legitimate AND High SNR
    train_indices = np.where(snr_mask & legit_mask)[0]
    
    # Anomaly Data: All samples from unauthorized classes (regardless of SNR)
    # We use these to test if the model rejects them.
    # We verify detection across ALL SNRs, even low ones.
    anomaly_indices = np.where(~legit_mask)[0]
    
    print(f"[INFO] Selected {len(train_indices)} samples for Legitimate Training (High SNR).")
    print(f"[INFO] Selected {len(anomaly_indices)} samples for Anomaly Testing.")
    
    # 5. Process Data in Batches to avoid OOM
    # For this script, we'll assume we can load the filtered legitimate set 
    # if it's < 5GB. If not, we would loop.
    
    print("[INFO] Processing Legitimate Data...")
    X_legit_processed = process_chunk(hf, train_indices, SLICE_LENGTH)
    
    print("[INFO] Processing Anomaly Data (Subsampling 50k for efficiency)...")
    # We randomly subsample anomaly data to keep file sizes manageable for the user
    np.random.seed(SEED)
    if len(anomaly_indices) > 50000:
        anomaly_indices_sub = np.random.choice(anomaly_indices, 50000, replace=False)
    else:
        anomaly_indices_sub = anomaly_indices
        
    X_anomaly_processed = process_chunk(hf, anomaly_indices_sub, SLICE_LENGTH)
    
    # 6. Train/Val/Test Split on Legitimate Data
    # 70% Train, 15% Val, 15% Test
    print("[INFO] Splitting datasets...")
    X_train, X_temp = train_test_split(X_legit_processed, test_size=0.3, random_state=SEED)
    X_val, X_test_legit = train_test_split(X_temp, test_size=0.5, random_state=SEED)
    
    # 7. Save to Disk
    # Using.npz for compressed storage
    save_file = os.path.join(OUTPUT_DIR, 'radioml_2018_processed.npz')
    print(f"[INFO] Saving to {save_file}...")
    
    np.savez_compressed(
        save_file,
        X_train=X_train,          # For training Autoencoder
        X_val=X_val,              # For validation loss monitoring
        X_test_legit=X_test_legit,# For testing True Positive Rate (TPR)
        X_test_anomaly=X_anomaly_processed # For testing False Positive Rate (FPR)
    )
    
    hf.close()
    print(" Preprocessing complete. Data ready for Hybrid Model.")

if __name__ == "__main__":
    main()