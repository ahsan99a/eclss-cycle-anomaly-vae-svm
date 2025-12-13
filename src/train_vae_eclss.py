"""
train_vae_eclss.py

Dense (MLP) Variational Autoencoder for the ECLSS synthetic dataset.

Pipeline:
  1) Load preprocessed data from data/eclss_preprocessed/
        - X_train_nom_flat.npy      (nominal training, flattened, scaled)
        - X_val_nom_flat.npy        (nominal validation, flattened, scaled)
        - X_test_all_flat.npy       (nominal + faulty, flattened, scaled)
        - y_test_all_binary.npy     (0 = nominal, 1 = anomaly)
  2) Define a fully-connected VAE:
        - Encoder: 3000 → 512 → 256 → 128 → (μ, logσ²) in ℝ^latent_dim
        - Decoder: latent_dim → 128 → 256 → 512 → 3000
  3) Train on NOMINAL data only (train set).
  4) Use validation loss for early stopping.
  5) Evaluate anomaly detection on test set using reconstruction error.
  6) Save:
        - Best VAE weights: models/vae_dense.pth
        - Per-sample reconstruction errors for test set: .npy
        - Latent vectors for test set (for later SVM): .npy
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

# ============================================================
# PATH HANDLING
# ============================================================

# If running as script, __file__ is defined; inside a notebook, fall back to cwd
from pathlib import Path

try:
    # project_root directory (where the .py file lives)
    REPO_ROOT = Path(__file__).resolve().parent
except NameError:
    # when running inside a notebook
    REPO_ROOT = Path.cwd()


DATA_ROOT = REPO_ROOT / "data"
PRE_DIR = DATA_ROOT / "eclss_preprocessed"
MODEL_DIR = REPO_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class VAEConfig:
    # Data dimensions
    input_dim: int = 3000          # 1000 timesteps × 3 sensors, flattened
    latent_dim: int = 32           # larger latent space

    # Hidden layer sizes (encoder; decoder mirrors this list)
    hidden_dims: Tuple[int, ...] = (1024, 512, 256)

    # Training hyperparameters
    batch_size: int = 8
    num_epochs: int = 200
    learning_rate: float = 1e-3
    beta: float = 0.3              # 

    # Early stopping
    early_stopping_patience: int = 30

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"



cfg = VAEConfig()


# ============================================================
# DATASET WRAPPER
# ============================================================

class NumpyDataset(Dataset):
    """
    Simple Dataset wrapper around a 2D numpy array (N, D).

    We already pre-flattened and normalized the data, so each row is
    a 3000-dimensional feature vector (1000 timesteps × 3 sensors).
    """

    def __init__(self, X: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.X[idx]


# ============================================================
# DENSE VAE DEFINITION
# ============================================================

class DenseVAE(nn.Module):
    """
    Fully-connected Variational Autoencoder.

    Encoder:
        x ∈ ℝ^D
          → hidden layers (Linear + BatchNorm + LeakyReLU + Dropout)
          → h ∈ ℝ^{hidden_dims[-1]}
          → μ, logσ² ∈ ℝ^{latent_dim}

    Decoder:
        z ∈ ℝ^{latent_dim}
          → hidden layers (mirror of encoder)
          → x̂ ∈ ℝ^D  (reconstructed, same dimensionality as input)

    We train with:
        L = reconstruction_loss + β * KL_divergence
    """

    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...], latent_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        # -------------------------
        # Encoder network
        # -------------------------
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.LeakyReLU(0.2))
            encoder_layers.append(nn.Dropout(0.1))
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Final linear layers to output μ and logσ²
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # -------------------------
        # Decoder network
        # -------------------------
        decoder_layers = []
        # Start by mapping latent z up to the last hidden size
        decoder_layers.append(nn.Linear(latent_dim, hidden_dims[-1]))
        decoder_layers.append(nn.BatchNorm1d(hidden_dims[-1]))
        decoder_layers.append(nn.LeakyReLU(0.2))

        # Then mirror the hidden_dims backwards
        reversed_hidden = list(hidden_dims[::-1])
        in_dim = reversed_hidden[0]
        for h_dim in reversed_hidden[1:]:
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.LeakyReLU(0.2))
            in_dim = h_dim

        # Final layer back to input_dim, no activation
        decoder_layers.append(nn.Linear(in_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    # ---------- VAE core methods ----------

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map input x → encoder hidden representation → (μ, logσ²).
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick:
            z = μ + σ ⊙ ε,    ε ~ N(0, I)

        Allows gradients to flow through μ and logσ².
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Map latent code z back to reconstruction x̂.
        """
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full VAE forward pass:
            x → (μ, logσ²) → z → x̂
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


# ============================================================
# LOSS FUNCTION
# ============================================================

def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute VAE loss: reconstruction + β * KL.

    - Reconstruction: mean-squared error (MSE) between x and x̂.
      Inputs are standardized (zero-mean, unit-variance), so MSE is natural.

    - KL divergence between q(z|x) = N(μ,σ²) and p(z) = N(0,I):
        KL = -0.5 * Σ (1 + logσ² - μ² - σ²)

    Returns:
        total_loss, recon_loss, kl_loss
    """
    # Mean squared error over features, averaged over batch
    recon_loss = F.mse_loss(x_hat, x, reduction="mean")

    # KL divergence (average over batch)
    kl_element = 1 + logvar - mu.pow(2) - logvar.exp()
    kl_loss = -0.5 * torch.mean(torch.sum(kl_element, dim=1))

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


# ============================================================
# TRAINING / EVAL LOOPS
# ============================================================

def train_one_epoch(
    model: DenseVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    beta: float,
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_batches = 0

    for x in loader:
        x = x.to(device)

        optimizer.zero_grad()
        x_hat, mu, logvar = model(x)
        loss, recon, kl = vae_loss(x, x_hat, mu, logvar, beta=beta)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        n_batches += 1

    return (
        total_loss / n_batches,
        total_recon / n_batches,
        total_kl / n_batches,
    )


def eval_one_epoch(
    model: DenseVAE,
    loader: DataLoader,
    device: str,
    beta: float,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_batches = 0

    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            loss, recon, kl = vae_loss(x, x_hat, mu, logvar, beta=beta)

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            n_batches += 1

    return (
        total_loss / n_batches,
        total_recon / n_batches,
        total_kl / n_batches,
    )


# ============================================================
# ANOMALY SCORING
# ============================================================

def compute_reconstruction_errors(
    model: DenseVAE,
    X: np.ndarray,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-sample reconstruction error and latent vectors.

    Args:
        model: trained VAE
        X:     numpy array (N, D)
    Returns:
        errors: np.array of shape (N,), MSE per sample
        Z:      np.array of shape (N, latent_dim), latent mean μ
    """
    model.eval()
    dataset = NumpyDataset(X)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    all_errors = []
    all_mu = []

    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            x_hat, mu, logvar = model(x)

            # MSE per sample (mean over features)
            mse = F.mse_loss(x_hat, x, reduction="none")
            mse_per_sample = mse.mean(dim=1).cpu().numpy()

            all_errors.append(mse_per_sample)
            all_mu.append(mu.cpu().numpy())

    errors = np.concatenate(all_errors, axis=0)
    Z = np.concatenate(all_mu, axis=0)
    return errors, Z


# ============================================================
# MAIN
# ============================================================

def main():
    print("============================================================")
    print(" DENSE VAE TRAINING FOR ECLSS ANOMALY DETECTION")
    print("============================================================")
    print(f"Using device: {cfg.device}")
    print(f"Preprocessed data dir: {PRE_DIR}\n")

    # --------------------------------------------------------
    # 1) LOAD PREPROCESSED DATA
    # --------------------------------------------------------
    X_train_nom = np.load(PRE_DIR / "X_train_nom_flat.npy")
    X_val_nom = np.load(PRE_DIR / "X_val_nom_flat.npy")
    X_test_all = np.load(PRE_DIR / "X_test_all_flat.npy")
    y_test_all_binary = np.load(PRE_DIR / "y_test_all_binary.npy")  # 0=nominal, 1=anomaly

    print("Shapes:")
    print(f"  X_train_nom: {X_train_nom.shape}")
    print(f"  X_val_nom:   {X_val_nom.shape}")
    print(f"  X_test_all:  {X_test_all.shape}")
    print(f"  y_test_all:  {y_test_all_binary.shape}\n")

    # Wrap in Datasets / DataLoaders
    train_dataset = NumpyDataset(X_train_nom)
    val_dataset = NumpyDataset(X_val_nom)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False
    )

    # --------------------------------------------------------
    # 2) INITIALIZE VAE
    # --------------------------------------------------------
    model = DenseVAE(
        input_dim=cfg.input_dim,
        hidden_dims=cfg.hidden_dims,
        latent_dim=cfg.latent_dim,
    ).to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    print(model)
    print("\nNumber of trainable parameters:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))
    print()

    # --------------------------------------------------------
    # 3) TRAIN WITH EARLY STOPPING (ON VALIDATION LOSS)
    # --------------------------------------------------------
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    best_model_path = MODEL_DIR / "vae_dense_eclss.pth"

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss, train_recon, train_kl = train_one_epoch(
            model, train_loader, optimizer, cfg.device, cfg.beta
        )
        val_loss, val_recon, val_kl = eval_one_epoch(
            model, val_loader, cfg.device, cfg.beta
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train: loss={train_loss:.5f}, recon={train_recon:.5f}, KL={train_kl:.5f} | "
            f"Val: loss={val_loss:.5f}, recon={val_recon:.5f}, KL={val_kl:.5f}"
        )

        # Early stopping check
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}. "
                      f"Best epoch was {best_epoch} with val_loss={best_val_loss:.5f}")
                break

    # Load best model weights
    model.load_state_dict(torch.load(best_model_path, map_location=cfg.device))
    print(f"\n✅ Loaded best model from: {best_model_path}")

    # --------------------------------------------------------
    # 4) ANOMALY DETECTION EVALUATION ON TEST SET
    # --------------------------------------------------------
    print("\nComputing reconstruction errors on test set...")
    test_errors, test_latent = compute_reconstruction_errors(
        model, X_test_all, cfg.device
    )

    # Simple threshold: use 99th percentile of TRAIN reconstruction errors
    train_errors, _ = compute_reconstruction_errors(model, X_train_nom, cfg.device)
    threshold = np.quantile(train_errors, 0.99)

    print(f"\nReconstruction error threshold (99th percentile of train): {threshold:.5f}")

    y_pred_binary = (test_errors > threshold).astype(int)

    # Metrics
    auc = roc_auc_score(y_test_all_binary, test_errors)
    cm = confusion_matrix(y_test_all_binary, y_pred_binary)
    report = classification_report(
        y_test_all_binary, y_pred_binary, target_names=["Nominal", "Anomaly"]
    )

    print("\n============================================================")
    print(" ANOMALY DETECTION PERFORMANCE (USING RECONSTRUCTION ERROR)")
    print("============================================================")
    print(f"AUC (recon error vs. binary label): {auc:.4f}")
    print("\nThreshold-based confusion matrix (99th percentile of train):")
    print(cm)
    print("\nClassification report:")
    print(report)

    # Also show approximate ROC operating point for reference
    fpr, tpr, roc_thresh = roc_curve(y_test_all_binary, test_errors)
    print(f"\nROC curve: first 5 points (FPR, TPR, thresh):")
    for i in range(min(5, len(fpr))):
        print(f"  {i}: FPR={fpr[i]:.3f}, TPR={tpr[i]:.3f}, thr={roc_thresh[i]:.5f}")

    # --------------------------------------------------------
    # 5) SAVE ERRORS & LATENT VECTORS (FOR SVM, PLOTS, ETC.)
    # --------------------------------------------------------
    np.save(PRE_DIR / "vae_test_recon_errors.npy", test_errors)
    np.save(PRE_DIR / "vae_test_latent_mu.npy", test_latent)

    print(f"\n✅ Saved test reconstruction errors to: "
          f"{(PRE_DIR / 'vae_test_recon_errors.npy').resolve()}")
    print(f"✅ Saved test latent μ vectors to: "
          f"{(PRE_DIR / 'vae_test_latent_mu.npy').resolve()}")

    print("\nDone.")
    print("  • Use 'vae_test_recon_errors.npy' to tune thresholds/plot ROC.")
    print("  • Use 'vae_test_latent_mu.npy' as features for SVM fault classifier.")
    print("  • Visualize latent space with t-SNE or PCA.\n")


if __name__ == "__main__":
    main()
