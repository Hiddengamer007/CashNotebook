#!/usr/bin/env python3
"""
Crypto Price Direction Predictor
================================
A production-grade ML pipeline for predicting short-horizon crypto price
direction using strictly causal features and calibrated probability outputs.

Architecture: Temporal CNN (causal dilated convolutions) with multi-head output
Output: 3-class probabilities (Down/Flat/Up) + confidence + volatility estimate
Training: Walk-forward with sequential batching, no shuffling, no future leakage

Requirements:
    pip install torch numpy pandas scikit-learn

Usage:
    python crypto_predictor.py --demo                    # synthetic data
    python crypto_predictor.py --data btc_5min.csv       # real data
"""

import argparse
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.calibration import calibration_curve
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore", category=FutureWarning)



# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

@dataclass
class Config:
    """All hyperparameters in one place. No magic numbers scattered in code."""

    num_workers: int = 4              # CPU cores for data loading
    pin_memory: bool = True           # Faster GPU transfer (if using GPU)
    
    # For CPU-only training
    torch_threads: int = 4            # PyTorch computation threads

    # Data
    bar_interval_minutes: int = 5
    max_missing_bars_fill: int = 12       # forward-fill up to N consecutive gaps

    # Labels — derived strictly from FUTURE prices
    horizon_bars: int = 12                # predict 3 bars ahead (15 min)
    label_threshold_bps: float = 35.0    # dead-zone: 15 bps > typical fees+slippage
    # Classes: 0=Down, 1=Flat, 2=Up

    # Features
    lookback_window: int = 48            # 48 bars = 4 hours of context
    ewm_span: int = 20                   # EWM span for causal normalization
    outlier_clip_std: float = 4.0        # winsorize at ±4σ

    # Model (kept small for CPU stability)
    input_features: int = 0              # set dynamically after feature build
    hidden_dim: int = 32
    num_layers: int = 1
    dropout: float = 0.5
    num_classes: int = 3

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 60
    patience: int = 8

    # Walk-forward windows
    #train_bars: int = 8640               # ~30 days of 5-min bars
    train_bars: int = 20920              # around 3 months to have enough data after feature engineering and label construction
    #val_bars: int = 2880                 # ~10 days
    val_bars: int = 8640                 # ~30 days for more robust validation
    step_bars: int = 8640                # ~30 day slide, no overlap between train/val

    # Loss function
    focal_gamma: float = 2.0             # focal loss focusing parameter
    direction_penalty_weight: float = 2.5 # extra cost for wrong-direction errors
    entropy_reg_weight: float = 0.1      # prevent overconfidence collapse
    flat_class_weight: float = 1.0       # downweight the dominant flat class
    aux_vol_loss_weight: float = 0.0     # auxiliary volatility prediction

    # Paths
    checkpoint_dir: str = "checkpoints"

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# §1  Data Loading & Validation
# ──────────────────────────────────────────────────────────────

def load_and_validate_data(path: str, cfg: Config) -> pd.DataFrame:
    """
    Load OHLCV CSV, align to a strict 5-min grid, and handle
    missing bars explicitly (no silent interpolation).
    """
    df = pd.read_csv(path)

    # ---- Standardize column names ----
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ("timestamp", "time", "date", "datetime"):
            col_map[col] = "timestamp"
        elif cl in ("open", "o"):
            col_map[col] = "open"
        elif cl in ("high", "h"):
            col_map[col] = "high"
        elif cl in ("low", "l"):
            col_map[col] = "low"
        elif cl in ("close", "c"):
            col_map[col] = "close"
        elif cl in ("volume", "vol", "v"):
            col_map[col] = "volume"
    df = df.rename(columns=col_map)

    required = ["timestamp", "open", "high", "low", "close", "volume"]
    assert all(c in df.columns for c in required), \
        f"Missing columns. Need: {required}, got: {list(df.columns)}"

    # ---- Parse timestamps ----
    if df["timestamp"].dtype == "object":
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    elif df["timestamp"].max() > 1e12:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    df = df.dropna(subset=["timestamp"])
    
    if df.empty:
        raise ValueError("No valid timestamps found in the CSV. Check your date format.")

    df = df.sort_values("timestamp").reset_index(drop=True)

    # ---- Build complete time grid (no gaps) ----
    full_index = pd.date_range(
        start=df["timestamp"].iloc[0],
        end=df["timestamp"].iloc[-1],
        freq=f"{cfg.bar_interval_minutes}min",
    )
    df = df.set_index("timestamp").reindex(full_index)

    # ---- Explicit missing-bar handling ----
    df["is_missing"] = df["close"].isna()
    missing_count = df["is_missing"].sum()
    total = len(df)
    print(f"Data: {total} bars, {missing_count} missing "
          f"({100 * missing_count / total:.1f}%)")

    # Only forward-fill SHORT gaps (≤ max_missing_bars_fill consecutive bars).
    # Longer gaps stay NaN and are excluded — no silent interpolation.
    consecutive_missing = df["is_missing"].groupby(
        (~df["is_missing"]).cumsum()
    ).cumsum()
    fillable = consecutive_missing <= cfg.max_missing_bars_fill

    for col in ["open", "high", "low", "close", "volume"]:
        filled = df[col].ffill()
        df[col] = df[col].where(~(df["is_missing"] & fillable), filled)

    df["has_gap"] = df["close"].isna()
    df = df.dropna(subset=["close"])

    df.index.name = "timestamp"
    df = df.reset_index()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    print(f"After cleaning: {len(df)} usable bars")
    return df

# ──────────────────────────────────────────────────────────────
# §2  Feature Engineering  (Strictly Causal)
# ──────────────────────────────────────────────────────────────
#
#   Rules enforced:
#     • Only log-returns, deltas, and ratios — NO absolute price levels
#     • All normalization uses EWM (causal) — never future data
#     • Time encoded as sin/cos cycles, not integer timestamps
#     • Volume as z-score (relative), never raw
#     • Outliers clipped at ±4σ
#     • Near-constant features dropped automatically
#

def build_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Return a DataFrame of strictly causal, normalized features."""

    feat = pd.DataFrame(index=df.index)

    close = df["close"].values.astype(np.float64)
    high  = df["high"].values.astype(np.float64)
    low   = df["low"].values.astype(np.float64)
    open_ = df["open"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)

    # ── Log returns at multiple look-back horizons ──────────────
    log_close = np.log(np.maximum(close, 1e-10))
    for lag in [1, 2, 3, 5, 8, 13]:
        ret = np.full_like(log_close, np.nan)
        ret[lag:] = log_close[lag:] - log_close[:-lag]
        feat[f"log_ret_{lag}"] = ret

    # ── Intra-bar microstructure ratios ─────────────────────────
    bar_range = np.maximum(high - low, 1e-10)
    feat["bar_range_ratio"]   = bar_range / np.maximum(close, 1e-10)
    feat["close_position"]    = (close - low) / bar_range
    feat["body_ratio"]        = (close - open_) / bar_range
    feat["upper_wick_ratio"]  = (high - np.maximum(close, open_)) / bar_range
    feat["lower_wick_ratio"]  = (np.minimum(close, open_) - low) / bar_range

    # ── Realised volatility (EWM, causal) ───────────────────────
    ret1 = pd.Series(feat["log_ret_1"].values)
    for span in [8, 21]:
        feat[f"volatility_ewm_{span}"] = (
            ret1.ewm(span=span, min_periods=span).std().values
        )
    feat["vol_ratio_8_21"] = (
        feat["volatility_ewm_8"].values
        / np.maximum(feat["volatility_ewm_21"].values, 1e-10)
    )

    # ── Volume features (relative only) ────────────────────────
    log_vol = np.log(np.maximum(volume, 1.0))
    log_vol_s = pd.Series(log_vol)
    vol_ewm_mean = log_vol_s.ewm(span=cfg.ewm_span, min_periods=10).mean()
    vol_ewm_std  = log_vol_s.ewm(span=cfg.ewm_span, min_periods=10).std()
    feat["volume_zscore"] = (
        (log_vol_s - vol_ewm_mean) / np.maximum(vol_ewm_std, 1e-10)
    ).values
    feat["volume_change"] = pd.Series(volume).pct_change().values
    feat["vol_price_corr"] = (
        ret1.rolling(13).corr(pd.Series(volume).pct_change()).values
    )

    # ── Normalised momentum (return / volatility) ──────────────
    for span in [5, 13, 21]:
        ewm_ret = ret1.ewm(span=span, min_periods=span).mean()
        ewm_vol = ret1.ewm(span=span, min_periods=span).std()
        feat[f"momentum_norm_{span}"] = (
            ewm_ret / np.maximum(ewm_vol, 1e-10)
        ).values

    # ── Mean-reversion z-score ──────────────────────────────────
    log_c = pd.Series(log_close)
    for span in [13, 34]:
        ewm_price = log_c.ewm(span=span, min_periods=span).mean()
        ewm_std   = log_c.ewm(span=span, min_periods=span).std()
        feat[f"mean_rev_{span}"] = (
            (log_c - ewm_price) / np.maximum(ewm_std, 1e-10)
        ).values

    # ── Return autocorrelation (regime detector) ────────────────
    feat["ret_autocorr_5"] = ret1.rolling(21).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 5 else 0, raw=False
    ).values

    # ── Cyclical time encoding (sin / cos) ──────────────────────
    ts = df["timestamp"]
    hour = ts.dt.hour + ts.dt.minute / 60.0
    feat["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    feat["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    dow = ts.dt.dayofweek.astype(float)
    feat["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    feat["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # ── Causal EWM normalisation of ALL non-cyclical features ───
    cyclical = {"hour_sin", "hour_cos", "dow_sin", "dow_cos"}
    for col in feat.columns:
        if col in cyclical:
            continue
        s = feat[col]
        mu  = s.ewm(span=cfg.ewm_span, min_periods=10).mean()
        sig = s.ewm(span=cfg.ewm_span, min_periods=10).std()
        feat[col] = ((s - mu) / np.maximum(sig, 1e-10)).clip(
            -cfg.outlier_clip_std, cfg.outlier_clip_std
        )

    # ── Drop near-constant features (low-vol regime artefacts) ─
    drop_cols = [
        col for col in feat.columns
        if col not in cyclical
        and feat[col].rolling(100, min_periods=50).std().median() < 0.05
    ]
    if drop_cols:
        print(f"Dropping {len(drop_cols)} low-variance features: {drop_cols}")
        feat.drop(columns=drop_cols, inplace=True)

    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"Final feature count: {len(feat.columns)}")
    return feat

# ──────────────────────────────────────────────────────────────
# §3  Label Construction  (Strictly from Future Prices)
# ──────────────────────────────────────────────────────────────
#
#   • 3-class: Down (0), Flat (1), Up (2)
#   • Threshold set ABOVE expected fees + slippage
#   • Auxiliary targets: |return|, future volatility
#   • No label depends on present or past data
#

def build_labels(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Derive labels from future prices only. Never from current bar."""

    labels = pd.DataFrame(index=df.index)
    close = df["close"].values.astype(np.float64)
    log_close = np.log(np.maximum(close, 1e-10))

    h = cfg.horizon_bars

    # ── Future log-return ───────────────────────────────────────
    future_ret = np.full_like(log_close, np.nan)
    future_ret[:-h] = log_close[h:] - log_close[:-h]
    labels["future_return"] = future_ret

    # ── 3-class direction label ─────────────────────────────────
    threshold = cfg.label_threshold_bps / 10_000.0
    direction = np.full(len(future_ret), np.nan)
    direction[future_ret >  threshold] = 2    # Up
    direction[future_ret < -threshold] = 0    # Down
    direction[(future_ret >= -threshold) &
              (future_ret <=  threshold)] = 1  # Flat (dead zone)
    labels["direction"] = direction

    # ── Auxiliary: magnitude & future vol ───────────────────────
    labels["abs_return"] = np.abs(future_ret)

    log_ret_1 = np.diff(log_close, prepend=log_close[0])
    future_vol = np.full_like(log_close, np.nan)
    for i in range(len(log_close) - h):
        future_vol[i] = np.std(log_ret_1[i + 1 : i + 1 + h])
    labels["future_volatility"] = future_vol

    # ── Print distribution ──────────────────────────────────────
    valid = ~np.isnan(direction)
    n = valid.sum()
    for cls, name in [(0, "Down"), (1, "Flat"), (2, "Up")]:
        c = (direction[valid] == cls).sum()
        print(f"  {name}: {c:>7,} ({100 * c / n:.1f}%)")

    return labels

# ──────────────────────────────────────────────────────────────
# §4  Dataset with Sequential Windows
# ──────────────────────────────────────────────────────────────
#
#   • Sliding windows in strict chronological order
#   • Labels lie OUTSIDE the input window (no overlap)
#   • No shuffling — SequentialBatchSampler enforces order
#   • Invalid windows (containing NaN) are excluded
#

class SequentialWindowDataset(Dataset):
    """
    Creates (window, label) pairs where each window of features
    is followed by a label derived from FUTURE bars only.
    """
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        aux_targets: np.ndarray,
        sample_weights: np.ndarray,
        window_size: int,
    ):
        self.features = features
        self.labels = labels
        self.aux_targets = aux_targets
        self.sample_weights = sample_weights
        self.window_size = window_size

        # Pre-compute valid indices (full window + valid label, no NaN)
        self.valid_indices = []
        for i in range(window_size - 1, len(features)):
            window = features[i - window_size + 1 : i + 1]
            if not np.isnan(labels[i]) and not np.any(np.isnan(window)):
                self.valid_indices.append(i)
        self.valid_indices = np.array(self.valid_indices)
        print(f"  Dataset: {len(self.valid_indices):,} valid windows "
              f"from {len(features):,} bars")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        window = self.features[i - self.window_size + 1 : i + 1]
        return (
            torch.FloatTensor(window),
            torch.LongTensor([int(self.labels[i])]),
            torch.FloatTensor(self.aux_targets[i]),
            torch.FloatTensor([self.sample_weights[i]]),
        )


class SequentialBatchSampler:
    """Yields batches in strict chronological order. Never shuffles."""
    def __init__(self, dataset_length: int, batch_size: int):
        self.n = dataset_length
        self.bs = batch_size

    def __iter__(self):
        for i in range(0, self.n, self.bs):
            yield list(range(i, min(i + self.bs, self.n)))

    def __len__(self):
        return (self.n + self.bs - 1) // self.bs

# ──────────────────────────────────────────────────────────────
# §5  Model Architecture
# ──────────────────────────────────────────────────────────────
#
#   Temporal CNN with causal dilated convolutions
#   + attention pooling + three output heads:
#
#     1. Direction head  →  3-class softmax (temperature-scaled)
#     2. Confidence head →  scalar in [0, 1]
#     3. Volatility head →  positive scalar (Softplus)
#
#   Capacity intentionally kept low for CPU training stability.
#

class TemporalBlock(nn.Module):
    """Single causal dilated convolution block with residual."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation          # causal padding
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=self.pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=self.pad, dilation=dilation)
        self.norm1 = nn.LayerNorm(out_ch)
        self.norm2 = nn.LayerNorm(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.skip  = (nn.Conv1d(in_ch, out_ch, 1)
                      if in_ch != out_ch else nn.Identity())

    def forward(self, x):                               # (B, C, T)
        res = self.skip(x)
        # First causal conv
        h = self.conv1(x)
        if self.pad > 0:
            h = h[:, :, :-self.pad]                     # trim future
        h = self.norm1(h.transpose(1, 2)).transpose(1, 2)
        h = F.gelu(h)
        h = self.drop(h)
        # Second causal conv
        h = self.conv2(h)
        if self.pad > 0:
            h = h[:, :, :-self.pad]
        h = self.norm2(h.transpose(1, 2)).transpose(1, 2)
        h = F.gelu(h)
        h = self.drop(h)
        return h + res


class CryptoPredictor(nn.Module):
    """
    Multi-head temporal CNN.

    Outputs dict with keys:
        logits      – raw class scores (temperature-scaled)
        probs       – calibrated softmax probabilities (sum to 1)
        confidence  – learned scalar uncertainty proxy ∈ [0, 1]
        vol_pred    – predicted future volatility (≥ 0)
        raw_logits  – un-scaled logits (for diagnostics)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # Input projection
        self.input_proj = nn.Linear(cfg.input_features, cfg.hidden_dim)

        # Temporal blocks with exponentially growing dilation
        self.blocks = nn.ModuleList([
            TemporalBlock(cfg.hidden_dim, cfg.hidden_dim,
                          kernel_size=3, dilation=2 ** i,
                          dropout=cfg.dropout)
            for i in range(cfg.num_layers)
        ])

        # Attention pooling over the time axis
        self.attn = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )

        # ── Output heads ───────────────────────────────────────
        self.direction_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, cfg.num_classes),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
        self.volatility_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim // 4, 1),
            nn.Softplus(),
        )

        # Learnable temperature for probability calibration
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):                               # (B, T, F)
        h = self.input_proj(x)                          # (B, T, H)
        h = h.transpose(1, 2)                           # (B, H, T)
        for block in self.blocks:
            h = block(h)
        h = h.transpose(1, 2)                           # (B, T, H)

        # Attention-weighted pooling
        w = F.softmax(self.attn(h), dim=1)              # (B, T, 1)
        ctx = (h * w).sum(dim=1)                        # (B, H)

        # Direction (temperature-scaled)
        raw = self.direction_head(ctx)
        temp = self.temperature.clamp(0.1, 5.0)
        scaled = raw / temp
        probs = F.softmax(scaled, dim=-1)

        return {
            "logits":     scaled,
            "probs":      probs,
            "confidence": self.confidence_head(ctx).squeeze(-1),
            "vol_pred":   self.volatility_head(ctx).squeeze(-1),
            "raw_logits": raw,
        }

# ──────────────────────────────────────────────────────────────
# §6  Loss Functions
# ──────────────────────────────────────────────────────────────
#
#   Primary:  Focal cross-entropy
#     • Penalise wrong DIRECTION more than weak confidence
#     • Weight each sample by |future return| (larger moves matter more)
#     • Downweight the flat class (most bars are flat)
#
#   Regularisation:
#     • Entropy term prevents overconfidence collapse
#     • Confidence head trained to predict correctness (calibration)
#
#   Auxiliary:
#     • Huber loss on predicted vs realised volatility
#

class DirectionalFocalLoss(nn.Module):
    """Multi-component loss with direction awareness."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.gamma   = cfg.focal_gamma
        self.dir_pen = cfg.direction_penalty_weight
        self.ent_w   = cfg.entropy_reg_weight
        self.vol_w   = cfg.aux_vol_loss_weight

        # Class weights: flat class downweighted
        cw = torch.FloatTensor([1.0, cfg.flat_class_weight, 1.0])
        self.register_buffer("class_weights", cw)

    def forward(self, outputs, targets, aux_targets, sample_weights):
        logits = outputs["logits"]
        probs  = outputs["probs"]
        conf   = outputs["confidence"]
        vpred  = outputs["vol_pred"]

        labels  = targets.squeeze(-1)
        abs_ret = aux_targets[:, 0]
        fut_vol = aux_targets[:, 1]
        sw      = sample_weights.squeeze(-1)

        # ── Focal cross-entropy ─────────────────────────────────
        ce = F.cross_entropy(logits, labels,
                             weight=self.class_weights, reduction="none")
        pt = probs[torch.arange(len(labels)), labels]
        focal = ((1 - pt) ** self.gamma) * ce

        # ── Direction penalty (wrong side is 2.5× worse) ───────
        pred_cls = probs.argmax(dim=-1)
        wrong_dir = ((labels == 0) & (pred_cls == 2)) | \
                    ((labels == 2) & (pred_cls == 0))
        dir_mult = 1.0 + wrong_dir.float() * self.dir_pen

        # ── Magnitude weighting ─────────────────────────────────
        #mag_w = 1.0 + abs_ret * 100.0
        mag_w = 1.0 + np.tanh(abs_ret * 10.0) #no extreme weights for outliers

        primary = (focal * mag_w * dir_mult * sw).mean()

        # ── Entropy regularisation ──────────────────────────────
        ent = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        ent_reg = -self.ent_w * ent.mean() / math.log(3)

        # ── Confidence calibration (BCE) ────────────────────────
        correct = (pred_cls == labels).float()
        conf_loss = F.binary_cross_entropy(conf, correct, reduction="mean")

        # ── Auxiliary volatility loss (Huber) ───────────────────
        valid = ~torch.isnan(fut_vol)
        vol_loss = (F.huber_loss(vpred[valid], fut_vol[valid], delta=1.0)
                    if valid.any() else torch.tensor(0.0))

        #total = primary + ent_reg + 0.1 * conf_loss + self.vol_w * vol_loss
        total = primary + 0.1 * conf_loss + self.vol_w * vol_loss # no entropy regularization for now, to see if it learns confidence at all

        metrics = {
            "primary":    primary.item(),
            "entropy":    ent.mean().item(),
            "conf_loss":  conf_loss.item(),
            "vol_loss":   vol_loss.item(),
            "mean_conf":  conf.mean().item(),
        }
        return total, metrics


class SimpleLoss(nn.Module):
    """
    Simplified loss with inverse-frequency class weighting.
    Automatically balances rare Up/Down classes against dominant Flat.
    """
    def __init__(self, cfg: Config, class_counts: np.ndarray = None):
        super().__init__()
        
        # Default: assume roughly 25% Down, 50% Flat, 25% Up
        if class_counts is None:
            class_counts = np.array([0.25, 0.50, 0.25])
        
        # Inverse frequency weighting
        # Flat (50%) gets weight ~1.0, Down/Up (25%) get weight ~2.0
        total = class_counts.sum()
        weights = total / (len(class_counts) * class_counts + 1e-10)
        
        # Normalize so mean weight = 1.0
        weights = weights / weights.mean()
        
        # Cap maximum weight to prevent instability
        weights = np.clip(weights, 0.5, 3.0)
        
        print(f"  Class weights: Down={weights[0]:.2f}, Flat={weights[1]:.2f}, Up={weights[2]:.2f}")
        
        self.register_buffer("class_weights", torch.FloatTensor(weights))

    def forward(self, outputs, targets, aux_targets, sample_weights):
        logits = outputs["logits"]
        conf = outputs["confidence"]
        labels = targets.squeeze(-1)

        # Weighted Cross Entropy
        ce = F.cross_entropy(logits, labels, weight=self.class_weights)

        # Confidence Calibration
        pred_cls = outputs["probs"].argmax(dim=-1)
        correct = (pred_cls == labels).float()
        conf_loss = F.binary_cross_entropy(conf, correct)

        total = ce + 0.1 * conf_loss

        metrics = {
            "primary": ce.item(),
            "conf_loss": conf_loss.item(),
            "mean_conf": conf.mean().item(),
            "entropy": 0.0,
            "vol_loss": 0.0,
        }
        return total, metrics
    
class DirectionalAuxiliaryLoss(nn.Module):
    """
    Multi-objective loss that combines:
    1. Direction prediction (weighted cross-entropy)
    2. Wrong-direction penalty (don't bet against the trend)
    3. Magnitude awareness (bigger moves = more important to get right)
    4. Volatility prediction (auxiliary task for regime detection)
    5. Confidence calibration (teach model when to say "I don't know")
    """
    def __init__(self, cfg: Config, class_counts: np.ndarray = None):
        super().__init__()
        
        # Class weights (inverse frequency)
        if class_counts is None:
            class_counts = np.array([0.25, 0.50, 0.25])
        
        total = class_counts.sum()
        weights = total / (len(class_counts) * class_counts + 1e-10)
        weights = weights / weights.mean()
        weights = np.clip(weights, 0.5, 3.0)
        
        print(f"  Class weights: Down={weights[0]:.2f}, Flat={weights[1]:.2f}, Up={weights[2]:.2f}")
        self.register_buffer("class_weights", torch.FloatTensor(weights))
        
        # Hyperparameters
        self.direction_penalty = 3.0    # Wrong direction is 3x worse than weak confidence
        self.magnitude_scale = 100.0    # Scale returns from [0.0001] to [0.01] range
        self.vol_weight = 0.15          # Volatility task weight
        self.conf_weight = 0.1          # Confidence calibration weight

    def forward(self, outputs, targets, aux_targets, sample_weights):
        logits = outputs["logits"]
        probs = outputs["probs"]
        conf = outputs["confidence"]
        vol_pred = outputs["vol_pred"]
        
        labels = targets.squeeze(-1)
        abs_ret = aux_targets[:, 0]     # Magnitude of the move
        fut_vol = aux_targets[:, 1]     # Realized future volatility
        sw = sample_weights.squeeze(-1)
        
        batch_size = labels.size(0)
        
        # ══════════════════════════════════════════════════════════
        # 1. BASE CLASSIFICATION LOSS (Weighted Cross-Entropy)
        # ══════════════════════════════════════════════════════════
        ce_loss = F.cross_entropy(
            logits, labels, 
            weight=self.class_weights, 
            reduction='none'
        )
        
        # ══════════════════════════════════════════════════════════
        # 2. DIRECTIONAL PENALTY (Punish wrong-side predictions)
        # ══════════════════════════════════════════════════════════
        # If true=Up but predicted=Down (or vice versa), this is catastrophic
        pred_class = probs.argmax(dim=-1)
        
        # Wrong direction = (true=Down AND pred=Up) OR (true=Up AND pred=Down)
        wrong_direction = (
            ((labels == 0) & (pred_class == 2)) |  # Predicted rally in a dump
            ((labels == 2) & (pred_class == 0))    # Predicted dump in a rally
        ).float()
        
        # Apply penalty multiplier
        direction_penalty = 1.0 + wrong_direction * self.direction_penalty
        
        # ══════════════════════════════════════════════════════════
        # 3. MAGNITUDE WEIGHTING (Big moves are more important)
        # ══════════════════════════════════════════════════════════
        # abs_ret is typically 0.0001 to 0.05 (1bp to 500bp)
        # Scale it so: 10bp move → weight=1.0, 100bp move → weight=10.0
        magnitude_weight = 1.0 + abs_ret * self.magnitude_scale
        
        # ══════════════════════════════════════════════════════════
        # 4. COMBINE INTO PRIMARY LOSS
        # ══════════════════════════════════════════════════════════
        primary_loss = (
            ce_loss * direction_penalty * magnitude_weight * sw
        ).mean()
        
        # ══════════════════════════════════════════════════════════
        # 5. AUXILIARY: VOLATILITY PREDICTION (Huber Loss)
        # ══════════════════════════════════════════════════════════
        # Teaching the model to predict future volatility helps it:
        # - Recognize regime changes (low→high vol = dangerous)
        # - Adjust confidence appropriately (high vol = lower confidence)
        valid_vol = ~torch.isnan(fut_vol)
        if valid_vol.sum() > 0:
            vol_loss = F.huber_loss(
                vol_pred[valid_vol], 
                fut_vol[valid_vol], 
                delta=1.0,
                reduction='mean'
            )
        else:
            vol_loss = torch.tensor(0.0, device=logits.device)
        
        # ══════════════════════════════════════════════════════════
        # 6. CONFIDENCE CALIBRATION (Binary Cross-Entropy)
        # ══════════════════════════════════════════════════════════
        # Train the confidence head to output:
        #   1.0 when prediction is correct
        #   0.0 when prediction is wrong
        # This creates a second "sanity check" network
        is_correct = (pred_class == labels).float()
        conf_loss = F.binary_cross_entropy(conf, is_correct, reduction='mean')
        
        # ══════════════════════════════════════════════════════════
        # 7. TOTAL LOSS (Weighted combination)
        # ══════════════════════════════════════════════════════════
        total_loss = (
            primary_loss +                          # Classification + direction + magnitude
            self.vol_weight * vol_loss +            # Auxiliary volatility task
            self.conf_weight * conf_loss            # Confidence calibration
        )
        
        # ══════════════════════════════════════════════════════════
        # 8. METRICS (for logging/debugging)
        # ══════════════════════════════════════════════════════════
        with torch.no_grad():
            # Entropy (measure of uncertainty)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            
            # What % of predictions were wrong-direction?
            wrong_dir_pct = wrong_direction.mean()
            
            # Average magnitude weight applied
            avg_mag_weight = magnitude_weight.mean()
        
        metrics = {
            "primary": primary_loss.item(),
            "vol_loss": vol_loss.item() if isinstance(vol_loss, torch.Tensor) else 0.0,
            "conf_loss": conf_loss.item(),
            "entropy": entropy.item(),
            "mean_conf": conf.mean().item(),
            "wrong_dir_pct": wrong_dir_pct.item(),
            "avg_mag_weight": avg_mag_weight.item(),
        }
        
        return total_loss, metrics

# ──────────────────────────────────────────────────────────────
# §7  Training Engine
# ──────────────────────────────────────────────────────────────
#
#   • Walk-forward: fresh model + optimizer per fold (no state leakage)
#   • Sequential batching (never shuffled)
#   • Monitors entropy, confidence, calibration every epoch
#   • Early-stops if confidence rises without accuracy improvement
#   • Saves intermediate checkpoints, not just best
#

class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device("cpu")

    # ── Sample Weights for Class Imbalance ──────────────────────
    def compute_sample_weights(
        self, labels: np.ndarray, returns: np.ndarray
    ) -> np.ndarray:
        """
        Inverse-frequency weighting + modest boost for rare large moves.
        No naive random duplication.
        """
        w = np.ones_like(returns, dtype=np.float64)
        valid = ~np.isnan(labels)
        if valid.sum() == 0:
            return w

        for cls in [0, 1, 2]:
            mask = (labels == cls) & valid
            cnt = mask.sum()
            if cnt > 0:
                w[mask] = 1.0 / (3.0 * (cnt / valid.sum()) + 1e-10)

        abs_ret = np.abs(returns)
        ok = valid & ~np.isnan(abs_ret)
        if ok.sum() > 0:
            p90 = np.percentile(abs_ret[ok], 90)
            w[abs_ret > p90] *= 1.5           # modest, not extreme

        w[valid] /= w[valid].mean()           # normalize to mean=1
        return w

    # ── Single Fold Training ────────────────────────────────────
    def train_fold(
        self,
        tr_feat, tr_lab, tr_aux, tr_sw,
        va_feat, va_lab, va_aux, va_sw,
        fold_id: int,
    ) -> Tuple[Optional[CryptoPredictor], Dict]:
        cfg = self.cfg

        train_ds = SequentialWindowDataset(
            tr_feat, tr_lab, tr_aux, tr_sw, cfg.lookback_window
        )
        val_ds = SequentialWindowDataset(
            va_feat, va_lab, va_aux, va_sw, cfg.lookback_window
        )
        if len(train_ds) < 100 or len(val_ds) < 50:
            print(f"  Fold {fold_id}: insufficient data — skipping")
            return None, {}

        # DataLoaders with sequential batching and parallel workers

        train_loader = DataLoader(
            train_ds,
            batch_sampler=SequentialBatchSampler(len(train_ds), cfg.batch_size),
            num_workers=cfg.num_workers,      # NEW: parallel data loading
            pin_memory=cfg.pin_memory,        # NEW: faster memory transfer
            persistent_workers=True,           # NEW: keep workers alive between epochs
        )
        val_loader = DataLoader(
            val_ds,
            batch_sampler=SequentialBatchSampler(len(val_ds), cfg.batch_size),
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )

        # Fresh model + optimizer (reset between folds)
        model = CryptoPredictor(cfg).to(self.device)
        optim = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=cfg.max_epochs, eta_min=1e-6
        )

        # Calculate class distribution
        valid_labels = tr_lab[~np.isnan(tr_lab)]
        class_counts = np.array([
            (valid_labels == 0).sum(),
            (valid_labels == 1).sum(),
            (valid_labels == 2).sum(),
        ]).astype(np.float64)
        
        print(f"  Class distribution: Down={class_counts[0]:.0f}, "
              f"Flat={class_counts[1]:.0f}, Up={class_counts[2]:.0f}")

        #criterion = DirectionalFocalLoss(cfg).to(self.device)
        #criterion = SimpleLoss(cfg).to(self.device) # simpler loss for stable training
        criterion = DirectionalAuxiliaryLoss(cfg, class_counts).to(self.device) #imporved Loss with auxiliary volatility prediction and confidence calibration.
        
        best_val, patience_ctr = float("inf"), 0
        hist = dict(train_loss=[], val_loss=[], val_entropy=[], val_acc=[])

        for epoch in range(cfg.max_epochs):
            # ---- train ----
            model.train()
            t_losses = []
            for feat, lab, aux, sw in train_loader:
                feat, lab, aux, sw = [t.to(self.device)
                                      for t in (feat, lab, aux, sw)]
                optim.zero_grad()
                out = model(feat)
                loss, _ = criterion(out, lab, aux, sw)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                t_losses.append(loss.item())

            # ---- validate ----
            model.eval()
            v_losses, v_ent, v_conf = [], [], []
            correct, total = 0, 0
            with torch.no_grad():
                for feat, lab, aux, sw in val_loader:
                    feat, lab, aux, sw = [t.to(self.device)
                                          for t in (feat, lab, aux, sw)]
                    out = model(feat)
                    loss, met = criterion(out, lab, aux, sw)
                    v_losses.append(loss.item())
                    pred = out["probs"].argmax(-1)
                    correct += (pred == lab.squeeze(-1)).sum().item()
                    total += len(lab)
                    ent = -(out["probs"] * torch.log(out["probs"] + 1e-10))
                    v_ent.extend(ent.sum(-1).cpu().tolist())
                    v_conf.extend(out["confidence"].cpu().tolist())
            sched.step()

            avg_t = np.mean(t_losses)
            avg_v = np.mean(v_losses)
            acc   = correct / max(total, 1)
            ent_m = np.mean(v_ent)
            cnf_m = np.mean(v_conf)

            for k, v in [("train_loss", avg_t), ("val_loss", avg_v),
                         ("val_entropy", ent_m), ("val_acc", acc)]:
                hist[k].append(v)

            # ---- checkpointing ----
            if avg_v < best_val:
                best_val = avg_v
                patience_ctr = 0
                torch.save({"epoch": epoch,
                             "state": model.state_dict(),
                             "val_loss": avg_v, "val_acc": acc},
                           f"{cfg.checkpoint_dir}/fold{fold_id}_best.pt")
            else:
                patience_ctr += 1

            if (epoch + 1) % 10 == 0:                   # intermediate saves
                torch.save({"epoch": epoch,
                             "state": model.state_dict()},
                           f"{cfg.checkpoint_dir}/fold{fold_id}_ep{epoch}.pt")

            if epoch % 5 == 0:
                print(f"  E{epoch:3d}  t={avg_t:.4f}  v={avg_v:.4f}  "
                      f"acc={acc:.3f}  H={ent_m:.3f}  conf={cnf_m:.3f}")

            if patience_ctr >= cfg.patience:
                print(f"  Early stop @ epoch {epoch}")
                break

            # Overconfidence alarm
            if epoch > 10 and ent_m < 0.3 and acc < 0.45:
                print(f"  ⚠  Low entropy ({ent_m:.3f}) with low "
                      f"accuracy ({acc:.3f}) — possible overfit")

        ckpt = torch.load(f"{cfg.checkpoint_dir}/fold{fold_id}_best.pt",
                          weights_only=False)
        model.load_state_dict(ckpt["state"])
        return model, hist

# ──────────────────────────────────────────────────────────────
# §8  Baselines & Evaluation
# ──────────────────────────────────────────────────────────────
#
#   Mandatory baselines the model MUST beat:
#     1. Random direction (uniform 3-class)
#     2. Always-flat predictor
#     3. Simple 1-bar momentum
#
#   Additional diagnostics:
#     • Calibration error per class
#     • Confidence separation (correct vs wrong)
#     • No-edge ratio (how often model says "I don't know")
#     • Probability jitter (smoothness over time)
#

class BaselineEvaluator:
    """Compare model against mandatory baselines."""

    @staticmethod
    def random_baseline(labels: np.ndarray) -> Dict:
        valid = ~np.isnan(labels)
        preds = np.random.choice([0, 1, 2], size=valid.sum(),
                                 p=[1/3, 1/3, 1/3])
        return {"name": "Random",
                "accuracy": (preds == labels[valid]).mean()}

    @staticmethod
    def always_flat_baseline(labels: np.ndarray) -> Dict:
        valid = ~np.isnan(labels)
        return {"name": "Always Flat",
                "accuracy": (labels[valid] == 1).mean()}

    @staticmethod
    def momentum_baseline(ret1: np.ndarray, labels: np.ndarray,
                           thr_bps: float) -> Dict:
        valid = ~np.isnan(labels) & ~np.isnan(ret1)
        thr = thr_bps / 10_000
        preds = np.ones_like(ret1)
        preds[ret1 >  thr] = 2
        preds[ret1 < -thr] = 0
        return {"name": "Momentum",
                "accuracy": (preds[valid] == labels[valid]).mean()}

    @staticmethod
    def evaluate_model(model, dataset, device, cfg) -> Dict:
        loader = DataLoader(
            dataset,
            batch_sampler=SequentialBatchSampler(len(dataset), cfg.batch_size),
        )
        model.eval()
        all_p, all_l, all_c, all_pred = [], [], [], []

        with torch.no_grad():
            for feat, lab, aux, sw in loader:
                out = model(feat.to(device))
                all_p.append(out["probs"].cpu().numpy())
                all_l.append(lab.squeeze(-1).cpu().numpy())
                all_c.append(out["confidence"].cpu().numpy())
                all_pred.append(out["probs"].argmax(-1).cpu().numpy())

        P = np.concatenate(all_p)
        L = np.concatenate(all_l)
        C = np.concatenate(all_c)
        Pr = np.concatenate(all_pred)

        acc = (Pr == L).mean()

        # Per-class accuracy
        names = ["Down", "Flat", "Up"]
        per_cls = {}
        for c, n in enumerate(names):
            m = L == c
            if m.sum():
                per_cls[n] = (Pr[m] == c).mean()

        # Calibration error
        cal = {}
        for c, n in enumerate(names):
            try:
                pt, pp = calibration_curve((L == c).astype(int),
                                           P[:, c], n_bins=5)
                cal[n] = np.abs(pt - pp).mean()
            except Exception:
                cal[n] = float("nan")

        # Confidence separation
        ok = Pr == L
        c_ok  = C[ok].mean()  if ok.sum()  else 0
        c_bad = C[~ok].mean() if (~ok).sum() else 0

        # No-edge ratio
        no_edge = ((P[:, 1] > 0.5) | (C < 0.3)).mean()

        # Probability jitter
        jitter = np.abs(np.diff(P, axis=0)).mean()

        return {
            "accuracy": acc,
            "per_class": per_cls,
            "calibration": cal,
            "conf_correct": c_ok,
            "conf_wrong": c_bad,
            "no_edge_ratio": no_edge,
            "jitter": jitter,
            "flat_pred_ratio": (Pr == 1).mean(),
        }

# ──────────────────────────────────────────────────────────────
# §9  Walk-Forward Pipeline
# ──────────────────────────────────────────────────────────────

def run_walk_forward(df: pd.DataFrame, cfg: Config):
    """
    Full walk-forward training + evaluation.
    Slides a (train, val) window forward in time.
    No overlap between train and val. No future leakage.
    """
    print("\n" + "=" * 60 + "\nBUILDING FEATURES\n" + "=" * 60)
    feat_df = build_features(df, cfg)

        # --- ADD THIS BLOCK ---
    # 1. Forward fill feature gaps (regime continuity)
    feat_df = feat_df.fillna(method="ffill")
    # 2. Fill remaining startup NaNs (start of file) with 0
    feat_df = feat_df.fillna(0.0)
    
    # Check if we still have NaNs (Should be 0)
    nans = feat_df.isna().sum().sum()
    if nans > 0:
        print(f"WARNING: Feature matrix still contains {nans} NaNs. Force filling with 0.")
        feat_df = feat_df.fillna(0.0)
    # ----------------------

    print("\n" + "=" * 60 + "\nBUILDING LABELS\n" + "=" * 60)
    lab_df = build_labels(df, cfg)

    cfg.input_features = len(feat_df.columns)

    features  = feat_df.values.astype(np.float32)
    dir_lab   = lab_df["direction"].values
    fut_ret   = lab_df["future_return"].values
    abs_ret   = lab_df["abs_return"].values
    fut_vol   = lab_df["future_volatility"].values
    aux       = np.column_stack([abs_ret, fut_vol]).astype(np.float32)

    trainer = Trainer(cfg)
    sw = trainer.compute_sample_weights(dir_lab, fut_ret).astype(np.float32)

    total = len(features)
    min_req = cfg.train_bars + cfg.val_bars + cfg.lookback_window
    if total < min_req:
        print(f"\nInsufficient data ({total} < {min_req}). Adjusting…")
        cfg.train_bars = int(total * 0.6)
        cfg.val_bars   = int(total * 0.2)
        cfg.step_bars  = int(total * 0.1)

    results = []
    fold, start = 0, 0

    print("\n" + "=" * 60 + "\nWALK-FORWARD TRAINING\n" + "=" * 60)

    while start + cfg.train_bars + cfg.val_bars <= total:
        t_end = start + cfg.train_bars
        v_end = t_end + cfg.val_bars

        print(f"\n── Fold {fold} ──  train [{start}:{t_end}]  "
              f"val [{t_end}:{v_end}]")

        model, hist = trainer.train_fold(
            features[start:t_end], dir_lab[start:t_end],
            aux[start:t_end],      sw[start:t_end],
            features[t_end:v_end], dir_lab[t_end:v_end],
            aux[t_end:v_end],      sw[t_end:v_end],
            fold,
        )

        if model is not None:
            val_ds = SequentialWindowDataset(
                features[t_end:v_end], dir_lab[t_end:v_end],
                aux[t_end:v_end], sw[t_end:v_end], cfg.lookback_window,
            )
            ev = BaselineEvaluator.evaluate_model(
                model, val_ds, trainer.device, cfg
            )
            bl_r = BaselineEvaluator.random_baseline(dir_lab[t_end:v_end])
            bl_f = BaselineEvaluator.always_flat_baseline(dir_lab[t_end:v_end])

            ret1 = (feat_df.iloc[t_end:v_end]["log_ret_1"].values
                    if "log_ret_1" in feat_df.columns
                    else np.zeros(v_end - t_end))
            bl_m = BaselineEvaluator.momentum_baseline(
                ret1, dir_lab[t_end:v_end], cfg.label_threshold_bps
            )

            print(f"  Model acc    {ev['accuracy']:.4f}")
            print(f"  Random       {bl_r['accuracy']:.4f}")
            print(f"  Always-flat  {bl_f['accuracy']:.4f}")
            print(f"  Momentum     {bl_m['accuracy']:.4f}")
            print(f"  No-edge      {ev['no_edge_ratio']:.4f}")
            print(f"  Conf ok/bad  {ev['conf_correct']:.3f} / "
                  f"{ev['conf_wrong']:.3f}")

            beat_r = ev["accuracy"] > bl_r["accuracy"]
            beat_f = ev["accuracy"] > bl_f["accuracy"]
            beat_m = ev["accuracy"] > bl_m["accuracy"]
            print(f"  Beats random={beat_r}  flat={beat_f}  "
                  f"momentum={beat_m}")

            results.append(dict(fold=fold, eval=ev, history=hist,
                                baselines=dict(r=bl_r, f=bl_f, m=bl_m)))

        start += cfg.step_bars
        fold += 1

    return results

# ──────────────────────────────────────────────────────────────
# §10  Inference & Decision Logic
# ──────────────────────────────────────────────────────────────

def predict(model: CryptoPredictor, features: np.ndarray,
            cfg: Config) -> Dict:
    """
    Run inference on the latest window of features.

    Returns calibrated probabilities, confidence, volatility
    estimate, and a trade/no-trade recommendation.

    The model frequently outputs "NO TRADE" — this is correct
    behaviour.  If it rarely says "no edge", it is overfit.
    """
    model.eval()

    if len(features) < cfg.lookback_window:
        return {"error": "Insufficient data for prediction"}

    window = features[-cfg.lookback_window:]
    if np.any(np.isnan(window)):
        return {"error": "NaN values in input window"}

    x = torch.FloatTensor(window).unsqueeze(0)

    with torch.no_grad():
        out = model(x)

    probs = out["probs"][0].numpy()
    conf  = out["confidence"][0].item()
    vol   = out["vol_pred"][0].item()

    # Entropy as secondary uncertainty measure
    H = -(probs * np.log(probs + 1e-10)).sum()
    H_norm = H / np.log(3)                      # 0 = certain, 1 = max

    pred = int(probs.argmax())
    names = ["DOWN", "FLAT", "UP"]

    # No-edge detection
    #no_edge = (
    #    probs[1] > 0.45
    #    or conf < 0.35
    #    or H_norm > 0.85
    #    or probs.max() < 0.45
    #)

    is_decisive = probs.max() > 0.55 #increased from 0.5
    is_confident = conf > 0.55 #increased from 0.5
    not_confused_by_flat = (probs[1] < 0.35) if pred != 1 else True

    no_edge = not (is_decisive and is_confident and not_confused_by_flat)

    return {
        "direction":      names[pred],
        "probabilities":  {"down": float(probs[0]),
                           "flat": float(probs[1]),
                           "up":   float(probs[2])},
        "confidence":     round(float(conf), 4),
        "entropy":        round(float(H_norm), 4),
        "pred_volatility": round(float(vol), 6),
        "has_edge":       not no_edge,
        "recommendation": "NO TRADE" if no_edge else names[pred],
    }

# ──────────────────────────────────────────────────────────────
# §11  Synthetic Data Generator & Main Entry Point
# ──────────────────────────────────────────────────────────────

def generate_synthetic_data(n_bars: int = 20_000) -> pd.DataFrame:
    """
    Produce realistic synthetic 5-min OHLCV with regime changes,
    volume clustering, and a few missing bars — for testing only.
    """
    np.random.seed(42)
    ts = pd.date_range("2024-01-01", periods=n_bars, freq="5min")

    price, prices = 40_000.0, []
    regime_len = 500

    for i in range(n_bars):
        regime = (i // regime_len) % 4
        if regime == 0:                       # trending up
            drift, vol = 2e-5, 1e-3
        elif regime == 1:                     # mean-reverting
            drift = -5e-6 * (price - 40_000) / 40_000
            vol = 8e-4
        elif regime == 2:                     # trending down
            drift, vol = -2e-5, 1.2e-3
        else:                                 # high volatility
            drift, vol = 0, 2e-3

        price *= np.exp(drift + vol * np.random.randn())
        prices.append(price)

    prices = np.array(prices)
    noise = np.abs(np.random.randn(n_bars) * 1e-3)
    base_vol = 100 + 50 * np.sin(np.arange(n_bars) * 2 * np.pi / 288)
    volume = base_vol * np.exp(np.random.randn(n_bars) * 0.5)

    for i in range(n_bars):
        if (i // regime_len) % 4 == 3:
            volume[i] *= 2

    df = pd.DataFrame({
        "timestamp": ts,
        "open":   prices * (1 + np.random.randn(n_bars) * 3e-4),
        "high":   prices * (1 + noise),
        "low":    prices * (1 - noise),
        "close":  prices,
        "volume": volume,
    })

    # Simulate ~0.5% missing bars
    drop = np.random.choice(n_bars, size=int(n_bars * 0.005), replace=False)
    df = df.drop(index=drop).reset_index(drop=True)
    return df


def main():
    torch.set_num_threads(6)  # Use 4 CPU cores for computation

    ap = argparse.ArgumentParser(description="Crypto Direction Predictor")
    ap.add_argument("--data",      type=str,   default=None)
    ap.add_argument("--demo",      action="store_true")
    ap.add_argument("--horizon",   type=int,   default=3)
    ap.add_argument("--threshold", type=float, default=15.0)
    ap.add_argument("--window",    type=int,   default=48)
    ap.add_argument("--epochs",    type=int,   default=60)
    args = ap.parse_args()

    cfg = Config()
    cfg.horizon_bars       = args.horizon
    cfg.label_threshold_bps = args.threshold
    cfg.lookback_window    = args.window
    cfg.max_epochs         = args.epochs

    if args.data:
        print(f"Loading {args.data}")
        df = load_and_validate_data(args.data, cfg)
    else:
        print("Generating synthetic demo data…")
        df = generate_synthetic_data(20_000)
        df.to_csv("synthetic_ohlcv.csv", index=False)
        df = load_and_validate_data("synthetic_ohlcv.csv", cfg)

    results = run_walk_forward(df, cfg)

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60 + "\nOVERALL SUMMARY\n" + "=" * 60)
    if results:
        accs = [r["eval"]["accuracy"] for r in results]
        ne   = [r["eval"]["no_edge_ratio"] for r in results]
        print(f"Folds:       {len(results)}")
        print(f"Accuracy:    {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"No-edge:     {np.mean(ne):.4f}")

        with open("training_summary.json", "w") as f:
            json.dump({
                "folds": len(results),
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy":  float(np.std(accs)),
                "mean_no_edge":  float(np.mean(ne)),
            }, f, indent=2)
        print("Saved training_summary.json")
    else:
        print("No successful folds.")


if __name__ == "__main__":
    main()
