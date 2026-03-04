"""
Per-Dataset Training Script for Comparison
==========================================
Trains one independent model per dataset using the SAME ACL2026Model
architecture as main.py — but trained on a single dataset only.

This mirrors what TIM-Net and EmoFusioNet do, giving a fair
per-dataset upper-bound to compare against our joint multilingual model.

Usage (run from project root):
    python comparison/train_per_dataset.py --dataset IEMOCAP
    python comparison/train_per_dataset.py --dataset EmoDB
    python comparison/train_per_dataset.py --dataset EMOVO
    python comparison/train_per_dataset.py --dataset SUBESCO
    python comparison/train_per_dataset.py --dataset RAVDESS

    # Or train all datasets sequentially:
    python comparison/train_per_dataset.py --dataset ALL
"""

import sys
import os

# Allow imports from project root (models/, etc.)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
import warnings
import numpy as np
import random
import pickle
import argparse
import json
from collections import Counter
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from models.novel_components import (
    AffectSpaceBidirectionalAttention,
    AdaptiveModalityGating,
    CrossModalProjectionHead
)

# ============================================================
# DATASET REGISTRY — Update paths to match your server
# ============================================================

BASE = "/dist_home/suryansh/sharukesh/speech/features_common_6"

DATASETS = {
    "IEMOCAP": {
        "lang": "English",
        "train": f"{BASE}/IEMOCAP_Common6_train.pkl",
        "val":   f"{BASE}/IEMOCAP_Common6_val.pkl",
        "test":  f"{BASE}/IEMOCAP_Common6_test.pkl",
    },
    "EmoDB": {
        "lang": "German",
        "train": f"{BASE}/EmoDB_Common6_train.pkl",
        "val":   f"{BASE}/EmoDB_Common6_val.pkl",
        "test":  f"{BASE}/EmoDB_Common6_test.pkl",
    },
    "EMOVO": {
        "lang": "Italian",
        "train": f"{BASE}/EMOVO_Common6_train.pkl",
        "val":   f"{BASE}/EMOVO_Common6_val.pkl",
        "test":  f"{BASE}/EMOVO_Common6_test.pkl",
    },
    "SUBESCO": {
        "lang": "Bangla",
        "train": f"{BASE}/SUBESCO_Common6_train.pkl",
        "val":   f"{BASE}/SUBESCO_Common6_val.pkl",
        "test":  f"{BASE}/SUBESCO_Common6_test.pkl",
    },
    "RAVDESS": {
        "lang": "English (actor)",
        "train": f"{BASE}/RAVDESS_Common6_train.pkl",
        "val":   f"{BASE}/RAVDESS_Common6_val.pkl",
        "test":  f"{BASE}/RAVDESS_Common6_test.pkl",
    },
}

EMOTION_LABELS = ["Anger", "Sadness", "Happiness", "Neutral", "Fear", "Disgust"]
ALL_LABEL_IDS  = [0, 1, 2, 3, 4, 5]

VAD_CONFIGS = {
    0: [-0.430,  0.800,  0.500],   # anger
    1: [-0.800, -0.700, -0.650],   # sadness
    2: [ 0.960,  0.650,  0.590],   # happiness
    3: [ 0.000,  0.000,  0.000],   # neutral
    4: [-0.640,  0.600, -0.650],   # fear
    5: [-0.600,  0.350, -0.400],   # disgust
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "per_dataset_models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "per_dataset_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    text_dim:   int   = 768
    audio_dim:  int   = 1024
    hidden_dim: int   = 384
    num_heads:  int   = 8
    num_layers: int   = 2
    dropout:    float = 0.4
    num_classes: int  = 6

    batch_size:    int   = 32
    lr:            float = 2e-5
    weight_decay:  float = 0.05
    epochs:        int   = 100
    patience:      int   = 15
    warmup_ratio:  float = 0.1

    num_runs: int   = 5
    seed:     int   = 42

    cls_weight:   float = 1.0
    vad_weight:   float = 0.3
    supcon_weight:float = 0.2
    vad_lambda:   float = 0.1
    supcon_temp:  float = 0.07
    micl_dim:     int   = 128
    gamma:        float = 2.0


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# DATASET
# ============================================================

class EmotionDataset(Dataset):
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        print(f"  Loading: {path}")
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        print(f"  Samples: {len(self.data)}")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        def to_ft(x):
            return torch.tensor(x, dtype=torch.float32) if not torch.is_tensor(x) else x.clone().detach().float()
        text  = to_ft(item['text_embed'])
        audio = to_ft(item['audio_embed'])
        label = torch.tensor(int(item['label']), dtype=torch.long) if not torch.is_tensor(item['label']) \
                else item['label'].clone().detach().long()
        return text, audio, label


def get_class_weights(dataset: Dataset, device) -> torch.Tensor:
    labels = [int(dataset[i][2].item()) for i in range(len(dataset))]
    counts = Counter(labels)
    total  = len(labels)
    weights = [total / (6 * counts.get(i, 1)) for i in range(6)]
    return torch.tensor(weights, dtype=torch.float32).to(device)

# ============================================================
# MODEL  (identical to main.py)
# ============================================================

class AttentivePooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.W = nn.Linear(input_dim, input_dim)
        self.w_context = nn.Linear(input_dim, 1, bias=False)

    def forward(self, features, mask=None):
        scores = self.w_context(torch.tanh(self.W(features)))
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        weights = F.softmax(scores, dim=1)
        return torch.sum(features * weights, dim=1)


class ACL2026Model(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        h = cfg.hidden_dim
        self.text_proj  = nn.Sequential(nn.Linear(cfg.text_dim,  h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(cfg.dropout))
        self.audio_proj = nn.Sequential(nn.Linear(cfg.audio_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(cfg.dropout))
        self.text_embed  = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.audio_embed = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.text_sa  = nn.TransformerEncoderLayer(d_model=h, nhead=cfg.num_heads, dim_feedforward=h*4, dropout=cfg.dropout, activation='gelu', batch_first=True)
        self.audio_sa = nn.TransformerEncoderLayer(d_model=h, nhead=cfg.num_heads, dim_feedforward=h*4, dropout=cfg.dropout, activation='gelu', batch_first=True)
        self.vga_layers = nn.ModuleList([AffectSpaceBidirectionalAttention(h, cfg.num_heads, cfg.dropout, cfg.vad_lambda) for _ in range(cfg.num_layers)])
        self.text_pool  = AttentivePooling(h)
        self.audio_pool = AttentivePooling(h)
        self.eaaf = AdaptiveModalityGating(h, cfg.dropout)
        self.micl_projector = CrossModalProjectionHead(h, cfg.micl_dim, h)
        self.vad_head   = nn.Sequential(nn.Linear(h, h//2), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(h//2, 3), nn.Tanh())
        self.classifier = nn.Sequential(nn.Linear(h+3, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(h, cfg.num_classes))
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, text, audio):
        if text.dim()  == 2: text  = text.unsqueeze(1)
        if audio.dim() == 2: audio = audio.unsqueeze(1)
        th = self.text_sa(self.text_proj(text)   + self.text_embed)
        ah = self.audio_sa(self.audio_proj(audio) + self.audio_embed)
        for vga in self.vga_layers: ah, th = vga(ah, th)
        tp, ap = self.text_pool(th), self.audio_pool(ah)
        fused, _ = self.eaaf(tp, ap)
        tp2, ap2 = self.micl_projector(tp, ap)
        vad = self.vad_head(fused)
        logits = self.classifier(torch.cat([fused, vad], dim=-1))
        return {
            'logits': logits,
            'probs':  F.softmax(logits, dim=-1),
            'vad':    vad,
            'text_proj':  tp2,
            'audio_proj': ap2,
        }

# ============================================================
# LOSSES
# ============================================================

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        bs = features.shape[0]
        if bs <= 1: return torch.tensor(0.0, device=device)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        sim = torch.div(torch.matmul(features, features.T), self.temperature)
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()
        lm = torch.scatter(torch.ones_like(mask), 1, torch.arange(bs).view(-1, 1).to(device), 0)
        mask = mask * lm
        exp = torch.exp(sim) * lm
        log_prob = sim - torch.log(exp.sum(1, keepdim=True) + 1e-8)
        return -((mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)).mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()


class CombinedLoss(nn.Module):
    def __init__(self, cfg: Config, class_weights, device):
        super().__init__()
        self.focal    = FocalLoss(alpha=class_weights, gamma=cfg.gamma)
        self.mse      = nn.MSELoss()
        self.supcon   = SupConLoss(cfg.supcon_temp)
        self.cls_w    = cfg.cls_weight
        self.vad_w    = cfg.vad_weight
        self.supcon_w = cfg.supcon_weight
        self.device   = device

    def forward(self, outputs, labels):
        cls_loss = self.focal(outputs['logits'], labels)
        vad_tgt  = torch.tensor([VAD_CONFIGS.get(l.item(), [0,0,0]) for l in labels],
                                  dtype=torch.float32, device=self.device)
        vad_loss = self.mse(outputs['vad'], vad_tgt)
        tn = F.normalize(outputs['text_proj'],  p=2, dim=1)
        an = F.normalize(outputs['audio_proj'], p=2, dim=1)
        sc = (self.supcon(tn, labels) + self.supcon(an, labels)) / 2
        return self.cls_w * cls_loss + self.vad_w * vad_loss + self.supcon_w * sc

# ============================================================
# TRAIN / EVALUATE
# ============================================================

def compute_metrics(y_true, y_pred) -> Dict:
    # Macro metrics computed ONLY over classes present in y_true
    # This prevents Disgust (0 support on IEMOCAP) from dragging down macro averages
    present_labels = sorted(set(y_true.tolist()))
    excluded = [EMOTION_LABELS[i] for i in ALL_LABEL_IDS if i not in present_labels]
    if excluded:
        print(f"  [Metrics] Excluding zero-support classes: {excluded}")

    wa = accuracy_score(y_true, y_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ua = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro',    labels=present_labels, zero_division=0)
    wf1      = f1_score(y_true, y_pred, average='weighted', labels=present_labels, zero_division=0)
    per_cls  = f1_score(y_true, y_pred, average=None,       labels=ALL_LABEL_IDS,  zero_division=0)
    return {
        'WA': wa, 'UA': ua, 'Macro_F1': macro_f1, 'WF1': wf1,
        'per_class_f1': {EMOTION_LABELS[i]: float(per_cls[i]) for i in range(6)},
        'excluded_classes': excluded
    }


@torch.no_grad()
def evaluate(model, loader, device) -> Dict:
    model.eval()
    preds, labels = [], []
    for t, a, l in loader:
        t, a = t.to(device), a.to(device)
        out = model(t, a)
        preds.extend(out['probs'].argmax(1).cpu().numpy())
        labels.extend(l.numpy())
    return compute_metrics(np.array(labels), np.array(preds))


def train_one_run(cfg: Config, train_ds, val_ds, run_idx: int, device, dataset_name: str):
    set_seed(cfg.seed + run_idx)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)

    model = ACL2026Model(cfg).to(device)
    cw    = get_class_weights(train_ds, device)
    criterion = CombinedLoss(cfg, cw, device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr * 10,
        total_steps=len(train_loader) * cfg.epochs,
        pct_start=cfg.warmup_ratio
    )

    best_ua, patience_ctr, best_state = 0.0, 0, None

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for t, a, l in train_loader:
            t, a, l = t.to(device), a.to(device), l.to(device)
            optimizer.zero_grad()
            out = model(t, a)
            loss = criterion(out, l)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        vm = evaluate(model, val_loader, device)
        print(f"  [{dataset_name}] Run {run_idx+1} | Ep {epoch+1:03d} | Loss {total_loss/len(train_loader):.4f} | Val UA {vm['UA']*100:.2f}%")

        if vm['UA'] > best_ua:
            best_ua = vm['UA']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= cfg.patience:
            print(f"  Early stopping at epoch {epoch+1}.")
            break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Save this run's best checkpoint
    ckpt_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_run{run_idx}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Saved checkpoint: {ckpt_path}")

    return evaluate(model, val_loader, device), ckpt_path


def train_dataset(dataset_name: str, cfg: Config):
    info = DATASETS[dataset_name]
    print(f"\n{'='*60}")
    print(f"TRAINING: {dataset_name} ({info['lang']})")
    print(f"{'='*60}")

    train_ds = EmotionDataset(info['train'])
    val_ds   = EmotionDataset(info['val'])
    test_ds  = EmotionDataset(info['test'])
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    all_val_metrics = []
    ckpt_paths      = []

    for i in range(cfg.num_runs):
        vm, ckpt = train_one_run(cfg, train_ds, val_ds, i, device, dataset_name)
        all_val_metrics.append(vm)
        ckpt_paths.append(ckpt)

    # Best checkpoint by val UA
    best_run = int(np.argmax([m['UA'] for m in all_val_metrics]))
    best_ckpt = ckpt_paths[best_run]

    # Test evaluation
    print(f"\n  Loading best checkpoint for test: {best_ckpt} (run {best_run})")
    best_model = ACL2026Model(cfg).to(device)
    best_model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
    test_m = evaluate(best_model, test_loader, device)

    # Validation mean ± std
    uas = [m['UA']  * 100 for m in all_val_metrics]
    was = [m['WA']  * 100 for m in all_val_metrics]

    result = {
        "dataset": dataset_name,
        "language": info['lang'],
        "num_runs": cfg.num_runs,
        "val": {
            "UA_mean": float(np.mean(uas)), "UA_std": float(np.std(uas)),
            "WA_mean": float(np.mean(was)), "WA_std": float(np.std(was)),
        },
        "test": {k: float(v) if not isinstance(v, dict) else v for k, v in test_m.items()},
        "best_run": best_run,
        "checkpoints": ckpt_paths,
    }

    # Save per-dataset result JSON
    out_path = os.path.join(RESULTS_DIR, f"{dataset_name}_results.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n  ── {dataset_name} Test Results ──")
    print(f"  WA        : {test_m['WA']*100:.2f}%")
    print(f"  UA        : {test_m['UA']*100:.2f}%")
    print(f"  Macro-F1  : {test_m['Macro_F1']*100:.2f}%")
    print(f"  Val UA (mean±std): {np.mean(uas):.2f}% ± {np.std(uas):.2f}%")
    print(f"  Results saved to: {out_path}")

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Per-Dataset SER Training for Comparison")
    parser.add_argument("--dataset", type=str, default="ALL",
                        choices=list(DATASETS.keys()) + ["ALL"],
                        help="Dataset to train on, or ALL to train all sequentially")
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of runs for mean±std reporting")
    parser.add_argument("--base_path", type=str, default=None,
                        help="Override the BASE path to feature pickle files")
    args = parser.parse_args()

    # Allow overriding base path from CLI (useful if running locally)
    if args.base_path:
        for ds in DATASETS.values():
            for split in ['train', 'val', 'test']:
                fname = os.path.basename(ds[split])
                ds[split] = os.path.join(args.base_path, fname)

    cfg = Config(epochs=args.epochs, num_runs=args.num_runs)

    if args.dataset == "ALL":
        datasets_to_train = list(DATASETS.keys())
    else:
        datasets_to_train = [args.dataset]

    all_results = {}
    for ds_name in datasets_to_train:
        all_results[ds_name] = train_dataset(ds_name, cfg)

    # Final summary table
    print(f"\n{'='*70}")
    print("PER-DATASET COMPARISON RESULTS (same architecture, per-dataset training)")
    print(f"{'='*70}")
    print(f"{'Dataset':<12} {'Lang':<16} {'Test WA':>9} {'Test UA':>9} {'Macro-F1':>10} {'Val UA (mean±std)':>22}")
    print("-" * 80)
    for ds_name, res in all_results.items():
        t = res['test']
        v = res['val']
        print(f"{ds_name:<12} {res['language']:<16} {t['WA']*100:>8.2f}% {t['UA']*100:>8.2f}% {t['Macro_F1']*100:>9.2f}%  {v['UA_mean']:>6.2f}% ± {v['UA_std']:.2f}%")

    # Combined JSON
    summary_path = os.path.join(RESULTS_DIR, "all_per_dataset_results.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to: {summary_path}")


if __name__ == "__main__":
    main()
