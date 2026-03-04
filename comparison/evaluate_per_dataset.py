"""
Per-Dataset Evaluation & Comparison Report Generator
=====================================================
Loads the saved per-dataset checkpoints and generates:
  1. Per-dataset test metrics (WA, UA, Macro-F1, per-class F1)
  2. Confusion matrices (saved as PNG)
  3. A paper-ready comparison table vs. your joint multilingual model results
  4. Full markdown report saved to comparison/per_dataset_results/

Usage (run from project root):
    python comparison/evaluate_per_dataset.py

Optional flags:
    --joint_results  Path to results_multilingual/all_metrics.json  (from evaluate_metrics.py)
                     to include side-by-side comparison in the report.
    --base_path      Override base path for feature pickle files.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
import json
import argparse
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score

from models.novel_components import (
    AffectSpaceBidirectionalAttention,
    AdaptiveModalityGating,
    CrossModalProjectionHead
)

# ============================================================
# PATHS — mirror train_per_dataset.py
# ============================================================

BASE = "/dist_home/suryansh/sharukesh/speech/features_common_6"

DATASETS = {
    "IEMOCAP": {"lang": "English",        "test": f"{BASE}/IEMOCAP_Common6_test.pkl"},
    "EmoDB":   {"lang": "German",         "test": f"{BASE}/EmoDB_Common6_test.pkl"},
    "EMOVO":   {"lang": "Italian",        "test": f"{BASE}/EMOVO_Common6_test.pkl"},
    "SUBESCO": {"lang": "Bangla",         "test": f"{BASE}/SUBESCO_Common6_test.pkl"},
    "RAVDESS": {"lang": "English (actor)","test": f"{BASE}/RAVDESS_Common6_test.pkl"},
}

EMOTION_LABELS = ["Anger", "Sadness", "Happiness", "Neutral", "Fear", "Disgust"]
ALL_LABEL_IDS  = [0, 1, 2, 3, 4, 5]

MODEL_DIR   = os.path.join(os.path.dirname(__file__), "per_dataset_models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "per_dataset_results")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "confusion_matrices")
os.makedirs(PLOTS_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# MODEL (identical to train_per_dataset.py)
# ============================================================

class AttentivePooling(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W = nn.Linear(d, d)
        self.w_context = nn.Linear(d, 1, bias=False)   # must match train_per_dataset.py
    def forward(self, x, mask=None):
        s = self.w_context(torch.tanh(self.W(x)))
        if mask is not None: s = s.masked_fill(~mask.unsqueeze(-1), -1e9)
        return (x * F.softmax(s, dim=1)).sum(1)


class ACL2026Model(nn.Module):
    def __init__(self, text_dim=768, audio_dim=1024, hidden_dim=384, num_heads=8,
                 num_layers=2, num_classes=6, dropout=0.4, vad_lambda=0.1, micl_dim=128):
        super().__init__()
        h = hidden_dim
        # NOTE: attribute names MUST match train_per_dataset.py exactly for weights to load correctly
        self.text_proj  = nn.Sequential(nn.Linear(text_dim,  h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout))
        self.audio_proj = nn.Sequential(nn.Linear(audio_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout))
        self.text_embed  = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.audio_embed = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.text_sa  = nn.TransformerEncoderLayer(d_model=h, nhead=num_heads, dim_feedforward=h*4, dropout=dropout, activation='gelu', batch_first=True)
        self.audio_sa = nn.TransformerEncoderLayer(d_model=h, nhead=num_heads, dim_feedforward=h*4, dropout=dropout, activation='gelu', batch_first=True)
        self.vga_layers     = nn.ModuleList([AffectSpaceBidirectionalAttention(h, num_heads, dropout, vad_lambda) for _ in range(num_layers)])
        self.text_pool      = AttentivePooling(h)
        self.audio_pool     = AttentivePooling(h)
        self.eaaf           = AdaptiveModalityGating(h, dropout)
        self.micl_projector = CrossModalProjectionHead(h, micl_dim, h)
        self.vad_head       = nn.Sequential(nn.Linear(h, h//2), nn.GELU(), nn.Dropout(dropout), nn.Linear(h//2, 3), nn.Tanh())
        self.classifier     = nn.Sequential(nn.Linear(h+3, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, num_classes))

    def forward(self, t, a):
        if t.dim()==2: t = t.unsqueeze(1)
        if a.dim()==2: a = a.unsqueeze(1)
        th = self.text_sa(self.text_proj(t)   + self.text_embed)
        ah = self.audio_sa(self.audio_proj(a) + self.audio_embed)
        for vga in self.vga_layers: ah, th = vga(ah, th)
        tp, ap = self.text_pool(th), self.audio_pool(ah)
        fused, _ = self.eaaf(tp, ap)
        vad = self.vad_head(fused)
        return self.classifier(torch.cat([fused, vad], dim=-1))


def load_model(ckpt_path: str) -> ACL2026Model:
    model = ACL2026Model().to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE), strict=True)
    return model

# ============================================================
# DATASET
# ============================================================

class SimpleDataset(Dataset):
    def __init__(self, path):
        with open(path, 'rb') as f: self.data = pickle.load(f)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        d = self.data[idx]
        t = torch.tensor(d['text_embed']).float()  if not torch.is_tensor(d['text_embed'])  else d['text_embed'].float()
        a = torch.tensor(d['audio_embed']).float() if not torch.is_tensor(d['audio_embed']) else d['audio_embed'].float()
        l = torch.tensor(int(d['label'])).long()
        return t, a, l

# ============================================================
# EVALUATION HELPERS
# ============================================================

@torch.no_grad()
def get_preds(model, loader):
    model.eval()
    preds, targets = [], []
    for t, a, l in loader:
        logits = model(t.to(DEVICE), a.to(DEVICE))
        preds.extend(logits.argmax(1).cpu().numpy())
        targets.extend(l.numpy())
    return np.array(preds), np.array(targets)


def compute_metrics(y_true, y_pred):
    wa  = accuracy_score(y_true, y_pred)
    ua  = balanced_accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average='macro',    labels=ALL_LABEL_IDS, zero_division=0)
    wf1 = f1_score(y_true, y_pred, average='weighted', labels=ALL_LABEL_IDS, zero_division=0)
    pc  = f1_score(y_true, y_pred, average=None,       labels=ALL_LABEL_IDS, zero_division=0)
    return {'WA': wa, 'UA': ua, 'Macro_F1': mf1, 'WF1': wf1,
            'per_class_f1': {EMOTION_LABELS[i]: float(pc[i]) for i in range(6)}}


def save_confusion_matrix(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=ALL_LABEL_IDS, normalize='true')
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion matrix: {out_path}")

# ============================================================
# FIND BEST CHECKPOINT PER DATASET
# ============================================================

def find_best_checkpoint(dataset_name: str):
    """
    Tries: (1) per_dataset_results/<dataset>_results.json (from training),
           (2) fallback: pick checkpoint with highest run index as proxy.
    """
    json_path = os.path.join(RESULTS_DIR, f"{dataset_name}_results.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            info = json.load(f)
        best_run = info.get('best_run', 0)
        ckpt = os.path.join(MODEL_DIR, f"{dataset_name}_run{best_run}.pth")
        if os.path.exists(ckpt):
            return ckpt

    # Fallback: find any matching checkpoint
    ckpts = sorted(glob.glob(os.path.join(MODEL_DIR, f"{dataset_name}_run*.pth")))
    if ckpts:
        print(f"  [WARN] No results JSON found; using: {ckpts[-1]}")
        return ckpts[-1]

    return None

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint_results", type=str, default=None,
                        help="Path to results_multilingual/all_metrics.json from evaluate_metrics.py")
    parser.add_argument("--base_path", type=str, default=None)
    args = parser.parse_args()

    if args.base_path:
        for ds in DATASETS.values():
            ds['test'] = os.path.join(args.base_path, os.path.basename(ds['test']))

    # Load joint model results if provided
    joint_results = {}
    if args.joint_results and os.path.exists(args.joint_results):
        with open(args.joint_results) as f:
            jdata = json.load(f)
        # Try to extract from first checkpoint's per_dataset metrics
        if 'per_dataset' in jdata and jdata['per_dataset']:
            run0 = jdata['per_dataset'][0]
            name_map = {
                "English (IEMOCAP)": "IEMOCAP",
                "German (EmoDB)":    "EmoDB",
                "Italian (EMOVO)":   "EMOVO",
                "Bangla (SUBESCO)":  "SUBESCO",
                "English (RAVDESS)": "RAVDESS",
            }
            for long_name, short in name_map.items():
                if long_name in run0:
                    joint_results[short] = run0[long_name]
        print(f"Loaded joint model results for: {list(joint_results.keys())}")

    results = {}
    report_lines = [
        "# Per-Dataset Model Evaluation Report",
        "## Same architecture (ACL2026Model), trained per-dataset\n",
        "Comparison: Per-Dataset training vs. Joint Multilingual training\n",
        "---",
    ]

    for ds_name, info in DATASETS.items():
        print(f"\n── {ds_name} ({info['lang']}) ──")
        ckpt = find_best_checkpoint(ds_name)
        if ckpt is None:
            print(f"  No checkpoint found for {ds_name}. Skipping.")
            print(f"  Run: python comparison/train_per_dataset.py --dataset {ds_name}")
            continue

        if not os.path.exists(info['test']):
            print(f"  Test file not found: {info['test']}. Skipping.")
            continue

        print(f"  Checkpoint: {ckpt}")
        model = load_model(ckpt)
        ds    = SimpleDataset(info['test'])
        dl    = DataLoader(ds, batch_size=32, shuffle=False)
        preds, targets = get_preds(model, dl)
        m = compute_metrics(targets, preds)
        results[ds_name] = m

        # Confusion matrix
        cm_path = os.path.join(PLOTS_DIR, f"CM_{ds_name}_PerDataset.png")
        save_confusion_matrix(targets, preds, f"{ds_name} — Per-Dataset Model", cm_path)

        # Report block
        report_lines.append(f"\n### {ds_name} ({info['lang']})")
        report_lines.append(f"| Metric    | Per-Dataset Model | Joint Multilingual |")
        report_lines.append(f"|-----------|-------------------|-------------------|")

        def jv(key): 
            if ds_name in joint_results:
                v = joint_results[ds_name].get(key, None)
                return f"{v*100:.2f}%" if v is not None else "—"
            return "—"

        report_lines.append(f"| WA        | {m['WA']*100:.2f}%  | {jv('WA')} |")
        report_lines.append(f"| UA        | {m['UA']*100:.2f}%  | {jv('UA')} |")
        report_lines.append(f"| Macro-F1  | {m['Macro_F1']*100:.2f}%  | {jv('Macro_F1')} |")
        report_lines.append(f"| WF1       | {m['WF1']*100:.2f}%  | {jv('WF1')} |")
        report_lines.append("\n**Per-class F1:**")
        for emo, f1 in m['per_class_f1'].items():
            report_lines.append(f"- {emo}: {f1*100:.2f}%")

        print(f"  WA={m['WA']*100:.2f}% | UA={m['UA']*100:.2f}% | Macro-F1={m['Macro_F1']*100:.2f}%")

    if not results:
        print("\nNo results generated. Please run train_per_dataset.py first.")
        return

    # Paper-style summary table
    report_lines.append("\n---\n## Paper Comparison Table\n")
    report_lines.append("| Dataset | Lang | Per-DS WA | Per-DS UA | Per-DS MF1 | Joint WA | Joint UA | Joint MF1 |")
    report_lines.append("|---------|------|-----------|-----------|------------|----------|----------|-----------|")
    for ds_name, m in results.items():
        lang = DATASETS[ds_name]['lang']
        jr = joint_results.get(ds_name, {})
        jwa  = f"{jr['WA']*100:.2f}%"  if 'WA'       in jr else "—"
        jua  = f"{jr['UA']*100:.2f}%"  if 'UA'       in jr else "—"
        jmf1 = f"{jr['Macro_F1']*100:.2f}%" if 'Macro_F1' in jr else "—"
        report_lines.append(
            f"| {ds_name} | {lang} | {m['WA']*100:.2f}% | {m['UA']*100:.2f}% | {m['Macro_F1']*100:.2f}% | {jwa} | {jua} | {jmf1} |"
        )

    # Print terminal summary
    print(f"\n{'='*75}")
    print(f"{'Dataset':<10} {'Lang':<16} {'WA':>8} {'UA':>8} {'Macro-F1':>10}")
    print("-" * 55)
    for ds_name, m in results.items():
        print(f"{ds_name:<10} {DATASETS[ds_name]['lang']:<16} {m['WA']*100:>7.2f}% {m['UA']*100:>7.2f}% {m['Macro_F1']*100:>9.2f}%")

    # Save
    report_path = os.path.join(RESULTS_DIR, "per_dataset_report.md")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"\nMarkdown report: {report_path}")

    result_path = os.path.join(RESULTS_DIR, "per_dataset_test_metrics.json")
    with open(result_path, 'w') as f:
        json.dump({ds: {k: v for k, v in m.items()} for ds, m in results.items()}, f, indent=2)
    print(f"JSON metrics:    {result_path}")
    print(f"Confusion plots: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
