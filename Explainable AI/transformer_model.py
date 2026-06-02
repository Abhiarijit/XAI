"""
transformer_model.py — TransformerFD001 classifier with attention capture.

Architecture (pre-norm):
    (B, T, D)
      → Linear(D, d_model) + sinusoidal PE
      → N_LAYERS × TransformerEncoderLayer (norm_first=True, batch_first=True)
      → LayerNorm → MeanPool over T → Dropout → Linear(d_model, K)
      → logits (B, K)

When capture_attn=True the forward pass manually iterates through layers
so we can call MultiheadAttention with need_weights=True, average_attn_weights=False
and obtain per-head weights (B, H, T, T) for attention rollout.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d

from config import Config, cfg


# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal positional encoding  (fixed, not learned)
# ─────────────────────────────────────────────────────────────────────────────

class _SinPE(nn.Module):
    def __init__(self, T: int, d_model: int):
        super().__init__()
        pe       = torch.zeros(T, d_model)
        position = torch.arange(T).unsqueeze(1).float()
        div      = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe)          # (T, d_model)

    def forward(self) -> torch.Tensor:
        return self.pe                           # (T, d_model)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class TransformerFD001(nn.Module):
    """
    3-class Transformer classifier for CMAPSS FD001.
    Supports attention rollout and input-gradient computation.
    """

    def __init__(self, c: Config = cfg):
        super().__init__()
        T, D, K       = c.T, c.D, c.K
        d, h, ff, l   = c.D_MODEL, c.N_HEADS, c.D_FF, c.N_LAYERS
        drop          = c.DROPOUT

        self.T = T
        self.D = D

        self.inp_proj = nn.Linear(D, d)
        self.pos_enc  = _SinPE(T, d)
        self.drop     = nn.Dropout(drop)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=h, dim_feedforward=ff,
            dropout=drop, activation="relu",
            batch_first=True, norm_first=True,    # pre-norm
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=l)
        self.ln_out  = nn.LayerNorm(d)
        self.head    = nn.Linear(d, K)

        self._attn_weights: list[torch.Tensor] = []
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, capture_attn: bool = False) -> torch.Tensor:
        """
        x            : (B, T, D)
        capture_attn : when True, fills self._attn_weights (list of (B, H, T, T))
        returns      : logits (B, K)
        """
        h = self.inp_proj(x) + self.pos_enc()   # (B, T, d_model)
        h = self.drop(h)

        if capture_attn:
            self._attn_weights = []
            for layer in self.encoder.layers:
                # Pre-norm attention
                h_norm   = layer.norm1(h)
                attn_out, attn_w = layer.self_attn(
                    h_norm, h_norm, h_norm,
                    need_weights=True,
                    average_attn_weights=False,   # keeps all H heads: (B, H, T, T)
                )
                self._attn_weights.append(attn_w.detach())
                h = h + layer.dropout1(attn_out)
                # Pre-norm FFN
                h = h + layer.dropout2(
                    layer.linear2(
                        layer.dropout(layer.activation(layer.linear1(layer.norm2(h))))
                    )
                )
        else:
            h = self.encoder(h)

        h      = self.ln_out(h)
        h_pool = h.mean(dim=1)                   # (B, d_model)
        return self.head(self.drop(h_pool))      # (B, K)

    # ── Explanation helpers ───────────────────────────────────────────────────

    @torch.no_grad()
    def attention_rollout(self, x_np: np.ndarray) -> np.ndarray:
        """
        Compute Abnar & Zuidema (2020) attention rollout saliency.

        x_np    : (T, D) single normalised window
        returns : s ∈ [0, 1]^T  (temporal saliency)
        """
        self.eval()
        x_t = torch.tensor(x_np[None], dtype=torch.float32)
        self.forward(x_t, capture_attn=True)

        T       = x_np.shape[0]
        rollout = torch.eye(T, dtype=torch.float64)

        for attn in self._attn_weights:           # (1, H, T, T)
            A_mean = attn[0].mean(dim=0).double() # (T, T) — average over heads
            A_hat  = 0.5 * A_mean + 0.5 * torch.eye(T, dtype=torch.double)
            A_hat  = A_hat / (A_hat.sum(dim=-1, keepdim=True) + 1e-9)
            rollout = rollout @ A_hat

        s = rollout.sum(dim=0).numpy()            # column sums → (T,)
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        return s.astype(np.float32)

    def input_gradient(self, x_np: np.ndarray, target_class: int) -> np.ndarray:
        """
        Gradient of logit[target_class] w.r.t. input.

        x_np         : (T, D)
        target_class : int
        returns      : (T, D) gradient array
        """
        self.eval()
        x_t    = torch.tensor(x_np[None], dtype=torch.float32, requires_grad=True)
        logits = self.forward(x_t)
        logits[0, target_class].backward()
        return x_t.grad[0].detach().cpu().numpy()

    # ── Prediction helpers ────────────────────────────────────────────────────

    def predict_proba(self, x_np: np.ndarray) -> np.ndarray:
        """
        x_np : (T, D)  or  (N, T, D)
        returns : (K,)  or  (N, K)  probability arrays
        """
        self.eval()
        squeeze = x_np.ndim == 2
        if squeeze:
            x_np = x_np[None]
        with torch.no_grad():
            logits = self.forward(torch.tensor(x_np, dtype=torch.float32))
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs[0] if squeeze else probs

    def predict(self, x_np: np.ndarray) -> int:
        """x_np : (T, D) → predicted class int"""
        return int(self.predict_proba(x_np).argmax())


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model: TransformerFD001,
                    loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: TransformerFD001, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        total_loss += criterion(logits, y_batch).item() * len(y_batch)
        correct    += (logits.argmax(dim=1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total


def train_transformer(model: TransformerFD001,
                      train_loader, test_loader,
                      c: Config = cfg,
                      device: str = "cpu") -> dict:
    """
    Full training loop with CosineAnnealingLR and early stopping on val accuracy.

    Returns history dict with train/val loss and accuracy per epoch.
    """
    device = torch.device(device)
    model  = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=c.TRAIN_LR, weight_decay=c.TRAIN_WD
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=c.EPOCHS
    )
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc   = 0.0
    best_state = None

    for epoch in range(1, c.EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        tr_loss2, tr_acc = evaluate(model, train_loader, criterion, device)
        va_loss,  va_acc = evaluate(model, test_loader,  criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        if va_acc > best_acc:
            best_acc   = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{c.EPOCHS} | "
                  f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
                  f"val_acc={va_acc:.4f}  (best={best_acc:.4f})")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n[Train] Restored best model  val_acc={best_acc:.4f}")

    model.to("cpu")
    return history


def save_model(model: TransformerFD001, path) -> None:
    torch.save(model.state_dict(), path)
    print(f"[Model] Saved → {path}")


def load_model(path, c: Config = cfg) -> TransformerFD001:
    model = TransformerFD001(c)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    print(f"[Model] Loaded ← {path}")
    return model
