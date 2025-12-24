import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block for 1D CNNs.
    
    Research Justification:
    Recent 2024 studies demonstrate that SE-Blocks enhance 
    RF fingerprint extraction by adaptively recalibrating channel weights, 
    effectively suppressing channel noise and highlighting hardware impairments.
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class RFAutoencoder(nn.Module):
    """
    Deep Convolutional Autoencoder for RF Fingerprinting (RadioML 2018.01A).
    
    Architecture:
    - Input: (Batch, 2, 128) -> I/Q samples
    - Encoder: 3x Conv1D layers + SE Attention + MaxPool
    - Latent: 64-dim vector (L2-normalized for hypersphere alignment)
    - Decoder: 3x ConvTranspose1D layers -> Reconstruction
    
    Includes a Projection Head for optional Contrastive Learning (SimCLR/InfoNCE).
    """
    def __init__(self, input_channels=2, seq_len=128, latent_dim=64):
        super(RFAutoencoder, self).__init__()
        
        # --- Encoder ---
        # Layer 1: Capture low-level temporal features
        self.enc_conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.enc_bn1 = nn.BatchNorm1d(32)
        self.enc_se1 = SEBlock(32)
        
        # Layer 2: Mid-level features
        self.enc_conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.enc_bn2 = nn.BatchNorm1d(64)
        self.enc_se2 = SEBlock(64)
        
        # Layer 3: High-level features
        self.enc_conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.enc_bn3 = nn.BatchNorm1d(128)
        self.enc_se3 = SEBlock(128)
        
        # Flattening dimension calculation: 128 -> 64 -> 32 -> 16 (due to 3 MaxPools)
        self.flatten_dim = 128 * (seq_len // 8) 
        
        # Latent Bottleneck
        self.fc_latent = nn.Linear(self.flatten_dim, latent_dim)
        self.dropout = nn.Dropout(0.2) # Regularization against overfitting

        # --- Projection Head (for Contrastive Learning - SOTA 2024) ---
        # Used only during training if contrastive loss is applied
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # --- Decoder ---
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        
        # Layer 3 (Inverse)
        self.dec_trans3 = nn.ConvTranspose1d(128, 64, kernel_size=3, padding=1)
        self.dec_bn3 = nn.BatchNorm1d(64)
        
        # Layer 2 (Inverse)
        self.dec_trans2 = nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1)
        self.dec_bn2 = nn.BatchNorm1d(32)
        
        # Layer 1 (Inverse)
        self.dec_trans1 = nn.ConvTranspose1d(32, input_channels, kernel_size=3, padding=1)

    def encoder(self, x):
        # Block 1
        x = self.enc_conv1(x)
        x = self.enc_bn1(x)
        x = F.relu(x)
        x = self.enc_se1(x) # Apply Attention
        x = F.max_pool1d(x, 2)
        
        # Block 2
        x = self.enc_conv2(x)
        x = self.enc_bn2(x)
        x = F.relu(x)
        x = self.enc_se2(x)
        x = F.max_pool1d(x, 2)
        
        # Block 3
        x = self.enc_conv3(x)
        x = self.enc_bn3(x)
        x = F.relu(x)
        x = self.enc_se3(x)
        x = F.max_pool1d(x, 2)
        
        # Flatten and Latent
        x = x.view(x.size(0), -1)
        x = self.fc_latent(x)
        
        # L2 Normalization: Critical for SVDD/OC-SVM downstream tasks 
        # This forces embeddings onto the unit hypersphere.
        x = F.normalize(x, p=2, dim=1)
        
        return x

    def decoder(self, z):
        x = self.fc_decode(z)
        x = x.view(x.size(0), 128, -1) # Reshape back to (Batch, 128, 16)
        
        # Block 3
        x = F.interpolate(x, scale_factor=2) # Upsample
        x = self.dec_trans3(x)
        x = self.dec_bn3(x)
        x = F.relu(x)
        
        # Block 2
        x = F.interpolate(x, scale_factor=2)
        x = self.dec_trans2(x)
        x = self.dec_bn2(x)
        x = F.relu(x)
        
        # Block 1
        x = F.interpolate(x, scale_factor=2)
        x = self.dec_trans1(x)
        
        # Output activation: Tanh usually fits normalized I/Q data [-1, 1]
        x = torch.tanh(x) 
        return x

    def forward(self, x, return_projection=False):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        
        if return_projection:
            # Return projection for Contrastive Loss calculation
            projection = self.projection_head(latent)
            return reconstruction, latent, projection
            
        return reconstruction, latent

# --- Example Usage for Testing ---
if __name__ == "__main__":
    # Simulate input batch: 32 samples, 2 channels (I/Q), 128 length
    dummy_input = torch.randn(32, 2, 128)
    
    # Initialize model
    model = RFAutoencoder()
    
    # Forward pass
    recon, z = model(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Reconstruction Shape: {recon.shape}")
    print(f"Latent Embedding Shape: {z.shape}")
    
    # Calculate Loss (MSE for Reconstruction)
    criterion = nn.MSELoss()
    loss = criterion(recon, dummy_input)
    print(f"Reconstruction Loss: {loss.item()}")