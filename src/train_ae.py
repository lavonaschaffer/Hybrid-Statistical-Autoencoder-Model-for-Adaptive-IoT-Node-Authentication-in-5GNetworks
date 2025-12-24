import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import argparse
from tqdm import tqdm

# Import model definition
import sys
from pathlib import Path
# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.autoencoder import RFAutoencoder

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'batch_size': 256,
    'epochs': 50,
    'lr': 0.001,
    'latent_dim': 64,
    'processed_data_path': 'data/processed/radioml_2018_processed.npz',
    'save_path': 'saved_models/ae_weights.pth',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # SOTA Hyperparameters
    'use_contrastive': True,  # Enable Joint Contrastive Learning
    'lambda_contrastive': 0.1, # Weight for contrastive loss
    'temperature': 0.07       # SimCLR temperature parameter
}

# ==========================================
# UTILITIES: SOTA AUGMENTATION & LOSS
# ==========================================

def rf_augment(signal_batch):
    """
    Physics-aware augmentation for RF signals.
    Applies random phase rotation and additive noise to create 'views' 
    of the same fingerprint for Contrastive Learning.
    
    Args:
        signal_batch (Tensor): Shape (B, 2, 128)
    Returns:
        Tensor: Augmented signals
    """
    B, C, L = signal_batch.shape
    device = signal_batch.device
    
    # 1. Random Phase Rotation (Simulating Oscillator Drift)
    # Treating (I, Q) as complex numbers: (I + jQ) * e^(j*theta)
    # theta ~ Uniform(0, 2pi)
    theta = torch.rand(B, device=device) * 2 * np.pi
    cos_t = torch.cos(theta).view(B, 1)  # Shape: (B, 1) for broadcasting with (B, L)
    sin_t = torch.sin(theta).view(B, 1)  # Shape: (B, 1) for broadcasting with (B, L)
    
    I = signal_batch[:, 0, :]  # Shape: (B, L)
    Q = signal_batch[:, 1, :]  # Shape: (B, L)
    
    I_rot = I * cos_t - Q * sin_t  # Broadcasting: (B, L) * (B, 1) -> (B, L)
    Q_rot = I * sin_t + Q * cos_t  # Broadcasting: (B, L) * (B, 1) -> (B, L)
    
    rotated_signal = torch.stack([I_rot, Q_rot], dim=1)  # Shape: (B, 2, L)
    
    # 2. Additive White Gaussian Noise (AWGN)
    noise = torch.randn_like(rotated_signal) * 0.05
    
    return rotated_signal + noise

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (SimCLR).
    Forces latent vectors of augmented versions of the same signal to be close,
    while pushing away others in the batch.
    """
    def __init__(self, temperature=0.07):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        """
        z_i, z_j: Projections of two augmented views of the same batch.
        Shape: (Batch_Size, Projection_Dim)
        """
        batch_size = z_i.shape[0]  # Get the batch size (first dimension)
        
        # Concatenate representations
        z = torch.cat([z_i, z_j], dim=0) # (2B, Dim)
        
        # Similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature # (2B, 2B)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim_matrix.masked_fill_(mask, -float('inf'))
        
        # Positive pairs are (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(z.device)
        
        loss = self.criterion(sim_matrix, labels)
        return loss

# ==========================================
# TRAINING PIPELINE
# ==========================================

def load_data():
    """Load preprocessed.npz files."""
    if not os.path.exists(CONFIG['processed_data_path']):
        raise FileNotFoundError("Processed data not found. Run preprocessing.py first.")
    
    print(f"[INFO] Loading data from {CONFIG['processed_data_path']}...")
    data = np.load(CONFIG['processed_data_path'])
    # We only train on 'X_train' which contains Legitimate devices (Class 1-5)
    X_train = torch.FloatTensor(data['X_train'])
    X_val = torch.FloatTensor(data['X_val'])
    
    # Channel First: (N, 128, 2) -> (N, 2, 128) for PyTorch Conv1d
    X_train = X_train.permute(0, 2, 1)
    X_val = X_val.permute(0, 2, 1)
    
    print(f"[INFO] Train shape: {X_train.shape}")
    return X_train, X_val

def train_model():
    # 1. Setup Data
    X_train, X_val = load_data()
    
    train_dataset = TensorDataset(X_train)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    val_dataset = TensorDataset(X_val)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    # 2. Setup Model & Optimizer
    model = RFAutoencoder(latent_dim=CONFIG['latent_dim']).to(CONFIG['device'])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    # Loss Functions
    mse_criterion = nn.MSELoss()
    contrastive_criterion = NTXentLoss(temperature=CONFIG['temperature'])

    print(f"[INFO] Starting training on {CONFIG['device']}...")
    print(f"       Method: {'Joint Contrastive + Reconstruction' if CONFIG['use_contrastive'] else 'Reconstruction Only'}")

    # 3. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss_accum = 0.0
        mse_accum = 0.0
        cont_accum = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for batch in progress_bar:
            # Handle batch - TensorDataset returns tuple/list, extract tensor
            if isinstance(batch, (list, tuple)):
                x_clean = batch[0].to(CONFIG['device'])
            else:
                x_clean = batch.to(CONFIG['device'])
            
            # --- Forward Pass ---
            if CONFIG['use_contrastive']:
                # Generate two augmented views for Contrastive Learning
                x_aug1 = rf_augment(x_clean)
                x_aug2 = rf_augment(x_clean)
                
                # Get reconstruction, latent, and projection for both
                # Note: We reconstruct the CLEAN input, but use AUGMENTED for contrastive
                recon_clean, _, _ = model(x_clean, return_projection=True)
                _, _, proj1 = model(x_aug1, return_projection=True)
                _, _, proj2 = model(x_aug2, return_projection=True)
                
                # Calculate Losses
                loss_mse = mse_criterion(recon_clean, x_clean)
                loss_cont = contrastive_criterion(proj1, proj2)
                
                # Joint Loss
                loss = loss_mse + (CONFIG['lambda_contrastive'] * loss_cont)
                
                cont_accum += loss_cont.item()
            else:
                # Standard Reconstruction Training
                recon, _ = model(x_clean)
                loss = mse_criterion(recon, x_clean)
                loss_mse = loss

            # --- Backward Pass ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item()
            mse_accum += loss_mse.item()
            
            # Update Log
            logs = {'Loss': loss.item(), 'MSE': loss_mse.item()}
            if CONFIG['use_contrastive']:
                logs['Cont'] = loss_cont.item()
            progress_bar.set_postfix(**logs)

        # 4. Validation Phase
        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                # Handle batch - TensorDataset returns tuple/list, extract tensor
                if isinstance(batch, (list, tuple)):
                    x_val = batch[0].to(CONFIG['device'])
                else:
                    x_val = batch.to(CONFIG['device'])
                
                recon_val, _ = model(x_val)
                loss_val = mse_criterion(recon_val, x_val)
                val_loss_accum += loss_val.item()

        avg_train_loss = train_loss_accum / len(train_loader)
        avg_val_loss = val_loss_accum / len(val_loader)
        
        print(f"   End Epoch {epoch+1}: Train Loss {avg_train_loss:.6f} | Val MSE {avg_val_loss:.6f}")

        # 5. Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(CONFIG['save_path']), exist_ok=True)
            torch.save(model.state_dict(), CONFIG['save_path'])
            print(f"   Model saved to {CONFIG['save_path']}")

if __name__ == "__main__":
    train_model()