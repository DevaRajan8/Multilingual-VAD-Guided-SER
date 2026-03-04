"""
Comprehensive Evaluation & Saving for Multilingual SER
Saves plots and metrics to 'results_multilingual' folder.
Fix: Explicitly handles missing classes in classification_report.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "/dist_home/suryansh/sharukesh/speech/pths/best_model_run_4.pth"  # Point to your best saved model
OUTPUT_DIR = "results_multilingual"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Labels (Common-6)
EMOTION_LABELS = ["Anger", "Sadness", "Happiness", "Neutral", "Fear", "Disgust"]
ALL_LABEL_IDS = [0, 1, 2, 3, 4, 5] # Explicit IDs for safety

BASE = "/dist_home/suryansh/sharukesh/speech/features_common_6"

# Test Files — one entry per dataset for per-language breakdown
TEST_FILES = {
    "English (IEMOCAP)" : f"{BASE}/IEMOCAP_Common6_test.pkl",
    "German (EmoDB)"    : f"{BASE}/EmoDB_Common6_test.pkl",
    "Italian (EMOVO)"   : f"{BASE}/EMOVO_Common6_test.pkl",
    "Bangla (SUBESCO)"  : f"{BASE}/SUBESCO_Common6_test.pkl",
    "English (RAVDESS)" : f"{BASE}/RAVDESS_Common6_test.pkl",
}

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. MODEL DEFINITION
# ==========================================
class AttentivePooling(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.W = nn.Linear(input_dim, input_dim)
        self.w_context = nn.Linear(input_dim, 1, bias=False)
    def forward(self, features, mask=None):
        attn_scores = self.w_context(torch.tanh(self.W(features)))
        if mask is not None: attn_scores = attn_scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        attn_weights = F.softmax(attn_scores, dim=1)
        return torch.sum(features * attn_weights, dim=1)

from models.novel_components import AffectSpaceBidirectionalAttention, AdaptiveModalityGating, CrossModalProjectionHead

class ACL2026Model(nn.Module):
    def __init__(self, text_dim=768, audio_dim=1024, hidden_dim=384, num_heads=8, num_layers=2, num_classes=6, dropout=0.3, vad_lambda=0.1, micl_dim=128):
        super().__init__()
        self.text_proj = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.audio_proj = nn.Sequential(nn.Linear(audio_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.text_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.audio_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.text_self_attn = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, dropout=dropout, activation='gelu', batch_first=True)
        self.audio_self_attn = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, dropout=dropout, activation='gelu', batch_first=True)
        self.vga_layers = nn.ModuleList([AffectSpaceBidirectionalAttention(hidden_dim, num_heads, dropout, vad_lambda) for _ in range(num_layers)])
        self.text_pool = AttentivePooling(hidden_dim)
        self.audio_pool = AttentivePooling(hidden_dim)
        self.eaaf = AdaptiveModalityGating(hidden_dim, dropout)
        self.micl_projector = CrossModalProjectionHead(hidden_dim, micl_dim, hidden_dim)
        self.vad_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim//2, 3), nn.Tanh())
        self.classifier = nn.Sequential(nn.Linear(hidden_dim+3, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, text_feat, audio_feat):
        if text_feat.dim() == 2: text_feat = text_feat.unsqueeze(1)
        if audio_feat.dim() == 2: audio_feat = audio_feat.unsqueeze(1)
        text_h = self.text_self_attn(self.text_proj(text_feat) + self.text_embed)
        audio_h = self.audio_self_attn(self.audio_proj(audio_feat) + self.audio_embed)
        for vga in self.vga_layers: audio_h, text_h = vga(audio_h, text_h)
        text_p, audio_p = self.text_pool(text_h), self.audio_pool(audio_h)
        fused, _ = self.eaaf(text_p, audio_p)
        vad = self.vad_head(fused)
        return self.classifier(torch.cat([fused, vad], dim=-1))

# ==========================================
# 3. UTILITIES
# ==========================================
class SimpleDataset(Dataset):
    def __init__(self, path):
        with open(path, 'rb') as f: self.data = pickle.load(f)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        t = torch.tensor(item['text_embed']).float() if not torch.is_tensor(item['text_embed']) else item['text_embed'].float()
        a = torch.tensor(item['audio_embed']).float() if not torch.is_tensor(item['audio_embed']) else item['audio_embed'].float()
        l = torch.tensor(int(item['label'])).long()
        return t, a, l

def get_predictions(model, dataloader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for t, a, l in dataloader:
            t, a = t.to(DEVICE), a.to(DEVICE)
            logits = model(t, a)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            targets.extend(l.numpy())
    return np.array(preds), np.array(targets)

def compute_all_metrics(y_true, y_pred):
    """Return a dict with WA, UA, Macro-F1, WF1, and per-class F1.
    Macro metrics exclude zero-support classes (e.g. Disgust on IEMOCAP test split)
    so the reported numbers are not artificially deflated.
    """
    import warnings
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score
    )
    # Only macro-average over classes that actually appear in the ground truth
    present_labels = sorted(set(y_true.tolist()))
    excluded = [EMOTION_LABELS[i] for i in ALL_LABEL_IDS if i not in present_labels]
    if excluded:
        print(f"  [Metrics] Excluding zero-support classes from macro metrics: {excluded}")

    wa = accuracy_score(y_true, y_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ua = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro',    labels=present_labels, zero_division=0)
    wf1      = f1_score(y_true, y_pred, average='weighted', labels=present_labels, zero_division=0)
    per_cls  = f1_score(y_true, y_pred, average=None,       labels=ALL_LABEL_IDS,  zero_division=0)
    return {
        'WA': wa, 'UA': ua, 'Macro_F1': macro_f1, 'WF1': wf1,
        'per_class_f1': {EMOTION_LABELS[i]: per_cls[i] for i in range(len(per_cls))},
        'excluded_classes': excluded
    }

def save_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=ALL_LABEL_IDS, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {save_path}")

def load_model(ckpt_path):
    model = ACL2026Model().to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE), strict=False)
    return model

# ==========================================
# 4. MAIN EVALUATION
# ==========================================
def evaluate_single_model(model, report_lines, global_preds, global_targets):
    """Run per-dataset evaluation for one model checkpoint. Returns per-dataset metrics dict."""
    all_metrics = {}
    for lang_name, path in TEST_FILES.items():
        if not os.path.exists(path):
            print(f"Skipping {lang_name} — file not found: {path}")
            continue

        ds = SimpleDataset(path)
        dl = DataLoader(ds, batch_size=32, shuffle=False)
        p, t = get_predictions(model, dl)
        m = compute_all_metrics(t, p)
        all_metrics[lang_name] = m

        global_preds.extend(p)
        global_targets.extend(t)

        # Confusion Matrix
        safe_name = lang_name.replace(" ", "_").replace("(", "").replace(")", "")
        save_confusion_matrix(t, p, title=f"{lang_name} Confusion Matrix",
                              filename=f"CM_{safe_name}.png")

        # Per-dataset block in report
        report_lines.append(f"--- {lang_name} ---")
        report_lines.append(f"  WA        : {m['WA']*100:.2f}%")
        report_lines.append(f"  UA        : {m['UA']*100:.2f}%")
        report_lines.append(f"  Macro-F1  : {m['Macro_F1']*100:.2f}%")
        report_lines.append(f"  WF1       : {m['WF1']*100:.2f}%")
        report_lines.append("  Per-class F1:")
        for emo, f1 in m['per_class_f1'].items():
            report_lines.append(f"    {emo:<12}: {f1*100:.2f}%")
        report_lines.append("")

    return all_metrics

def print_summary_table(all_run_metrics):
    """
    all_run_metrics: list of dicts  {lang_name: {WA, UA, Macro_F1, WF1}}
    Prints a paper-style table with mean ± std across runs.
    """
    import sys

    datasets  = list(all_run_metrics[0].keys())
    metrics   = ['WA', 'UA', 'Macro_F1', 'WF1']
    col_w     = 20

    header = f"{'Dataset':<25}" + "".join(f"{m:>{col_w}}" for m in metrics)
    print("\n" + "="*len(header))
    print("SUMMARY TABLE (mean ± std across checkpoints)")
    print("="*len(header))
    print(header)
    print("-"*len(header))

    table_rows = {}
    for ds in datasets:
        row = {}
        for metric in metrics:
            vals = [r[ds][metric] * 100 for r in all_run_metrics if ds in r]
            mean = np.mean(vals)
            std  = np.std(vals)
            row[metric] = (mean, std)
        table_rows[ds] = row
        line = f"{ds:<25}" + "".join(f"{mean:>14.2f}±{std:<5.2f}" for mean, std in row.values())
        print(line)

    print("="*len(header) + "\n")
    return table_rows

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Find all available checkpoints ──────────────────────────────────────────
    # Accept either a single MODEL_PATH or auto-discover all best_model_run_*.pth
    if os.path.exists(MODEL_PATH):
        ckpt_paths = [MODEL_PATH]
    else:
        # Auto-discover from current directory
        import glob as _glob
        ckpt_paths = sorted(_glob.glob("best_model_run_*.pth"))
        if not ckpt_paths:
            raise FileNotFoundError(
                f"No checkpoint found at {MODEL_PATH} and no best_model_run_*.pth files found."
            )

    print(f"Found {len(ckpt_paths)} checkpoint(s): {ckpt_paths}")

    all_run_metrics  = []   # list of per-dataset metrics dicts, one per checkpoint
    all_global_metrics = [] # list of global metrics dicts, one per checkpoint

    report_lines = [
        "MULTILINGUAL SER EVALUATION REPORT",
        "====================================\n",
        f"Checkpoints evaluated: {', '.join(ckpt_paths)}",
        ""
    ]

    for ckpt_idx, ckpt_path in enumerate(ckpt_paths):
        print(f"\n── Checkpoint {ckpt_idx+1}/{len(ckpt_paths)}: {ckpt_path} ──")
        report_lines.append(f"==== Checkpoint: {ckpt_path} ====")

        model = load_model(ckpt_path)
        global_preds, global_targets = [], []

        ds_metrics = evaluate_single_model(model, report_lines, global_preds, global_targets)
        all_run_metrics.append(ds_metrics)

        # Global pooled metrics
        gp = np.array(global_preds)
        gt = np.array(global_targets)
        gm = compute_all_metrics(gt, gp)
        all_global_metrics.append(gm)

        save_confusion_matrix(gt, gp,
                              title=f"Global Confusion Matrix — {os.path.basename(ckpt_path)}",
                              filename=f"CM_Global_ckpt{ckpt_idx}.png")

        report_lines.append("--- GLOBAL (all datasets pooled) ---")
        report_lines.append(f"  WA        : {gm['WA']*100:.2f}%")
        report_lines.append(f"  UA        : {gm['UA']*100:.2f}%")
        report_lines.append(f"  Macro-F1  : {gm['Macro_F1']*100:.2f}%")
        report_lines.append(f"  WF1       : {gm['WF1']*100:.2f}%")
        report_lines.append("")

    # ── Cross-checkpoint summary table ──────────────────────────────────────────
    if len(all_run_metrics) > 1:
        table_rows = print_summary_table(all_run_metrics)
        # Global summary
        for metric in ['WA', 'UA', 'Macro_F1', 'WF1']:
            vals = [m[metric]*100 for m in all_global_metrics]
            print(f"Global {metric}: {np.mean(vals):.2f}% ± {np.std(vals):.2f}%")
    else:
        # Single checkpoint — just print the values
        for lang, m in all_run_metrics[0].items():
            print(f"{lang}: WA={m['WA']*100:.2f}% UA={m['UA']*100:.2f}% Macro-F1={m['Macro_F1']*100:.2f}%")

    # ── Save text report ─────────────────────────────────────────────────────────
    report_path = os.path.join(OUTPUT_DIR, "performance_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport saved to: {os.path.abspath(report_path)}")

    # ── Save structured JSON (all per-dataset + global metrics, all checkpoints) ─
    json_path = os.path.join(OUTPUT_DIR, "all_metrics.json")
    # Convert per_class_f1 keys to strings for JSON serialization
    def serialize(m):
        return {k: (v if not isinstance(v, dict) else {ek: float(ev) for ek, ev in v.items()})
                for k, v in m.items()}

    json_data = {
        "checkpoints": ckpt_paths,
        "per_dataset": [{ds: serialize(m) for ds, m in run.items()} for run in all_run_metrics],
        "global":      [serialize(gm) for gm in all_global_metrics],
    }
    with open(json_path, "w") as f:
        import json
        json.dump(json_data, f, indent=2)
    print(f"Structured JSON saved to: {os.path.abspath(json_path)}")
    print(f"\nEvaluation Complete! All outputs in: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()