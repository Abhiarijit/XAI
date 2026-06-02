"""
transformer_model.py — Transformer classifier for NATOPS (T=51, D=24, K=6).

Architecture:
    Input  : (B, T, D)
    Linear projection → (B, T, d_model)
    Positional encoding (sinusoidal)
    N_LAYERS × AttentionLayer  [multi-head self-attention + FFN]
    Global average pool → (B, d_model)
    Linear → (B, K)
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from config import Config, cfg


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class AttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn    = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff      = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, w = self.attn(x, x, x, need_weights=True, average_attn_weights=True)
        self.attn_weights = w.detach()
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class TransformerNATOPS(nn.Module):
    def __init__(self, c: Config = cfg):
        super().__init__()
        self.input_proj = nn.Linear(c.D, c.D_MODEL)
        self.pos_enc    = PositionalEncoding(c.D_MODEL, max_len=c.T + 10, dropout=c.DROPOUT)
        self.layers     = nn.ModuleList([
            AttentionLayer(c.D_MODEL, c.N_HEADS, c.D_FF, c.DROPOUT)
            for _ in range(c.N_LAYERS)
        ])
        self.classifier = nn.Linear(c.D_MODEL, c.K)
        self.c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(self.input_proj(x))
        for layer in self.layers:
            x = layer(x)
        return self.classifier(x.mean(dim=1))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32)
            if t.dim() == 2:
                t = t.unsqueeze(0)
            return torch.softmax(self(t), dim=-1).cpu().numpy()

    def predict(self, X: np.ndarray) -> int:
        return int(self.predict_proba(X[None])[0].argmax())


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += criterion(logits, y).item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
            n          += len(y)
    return total_loss / n, correct / n


def train_transformer(model, train_dl, test_dl, c: Config = cfg, device: str = "cpu"):
    dev = torch.device(device)
    model.to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=c.TRAIN_LR, weight_decay=c.TRAIN_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c.EPOCHS)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc, best_state = 0.0, None

    for epoch in range(1, c.EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        for X, y in train_dl:
            X, y = X.to(dev), y.to(dev)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * len(y)
        scheduler.step()

        val_loss, val_acc = evaluate(model, test_dl, criterion, dev)
        ep_loss /= len(train_dl.dataset)

        history["train_loss"].append(ep_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{c.EPOCHS}  "
                  f"train_loss={ep_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"val_acc={val_acc:.4f}  best={best_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(torch.device("cpu"))
    print(f"\n  Training complete. Best val_acc = {best_acc:.4f}")
    return history


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  Model saved → {path}")


def load_model(path: Path, c: Config = cfg):
    model = TransformerNATOPS(c)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
