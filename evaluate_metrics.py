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
MODEL_PATH = "./bestmodels/best_model_run_1.pth"  # Point to your model
OUTPUT_DIR = "results_multilingual"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Labels (Common-6)
EMOTION_LABELS = ["Anger", "Sadness", "Happiness", "Neutral", "Fear", "Disgust"]
ALL_LABEL_IDS = [0, 1, 2, 3, 4, 5] # Explicit IDs for safety

# Test Files
TEST_FILES = {
    "English (IEMOCAP)": "features_common_6/IEMOCAP_Common6_test.pkl",
    "German (EmoDB)":    "features_common_6/EmoDB_Common6_test.pkl",
    "Italian (EMOVO)":   "features_common_6/EMOVO_Common6_test.pkl"
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

from models.novel_components import VADGuidedBidirectionalAttention, EmotionAwareAdaptiveFusion, MICLProjector

class ACL2026Model(nn.Module):
    def __init__(self, text_dim=768, audio_dim=1024, hidden_dim=384, num_heads=8, num_layers=2, num_classes=6, dropout=0.3, vad_lambda=0.1, micl_dim=128):
        super().__init__()
        self.text_proj = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.audio_proj = nn.Sequential(nn.Linear(audio_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.text_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.audio_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.text_self_attn = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, dropout=dropout, activation='gelu', batch_first=True)
        self.audio_self_attn = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, dropout=dropout, activation='gelu', batch_first=True)
        self.vga_layers = nn.ModuleList([VADGuidedBidirectionalAttention(hidden_dim, num_heads, dropout, vad_lambda) for _ in range(num_layers)])
        self.text_pool = AttentivePooling(hidden_dim)
        self.audio_pool = AttentivePooling(hidden_dim)
        self.eaaf = EmotionAwareAdaptiveFusion(hidden_dim, dropout)
        self.micl_projector = MICLProjector(hidden_dim, micl_dim, hidden_dim)
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

def save_confusion_matrix(y_true, y_pred, title, filename):
    # Ensure all labels are represented in the matrix, even if missing from data
    cm = confusion_matrix(y_true, y_pred, labels=ALL_LABEL_IDS, normalize='true')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")

# ==========================================
# 4. MAIN EVALUATION
# ==========================================
def main():
    print(f"Loading model: {MODEL_PATH}...")
    model = ACL2026Model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    
    global_preds, global_targets = [], []
    report_lines = []
    
    report_lines.append(f"MULTILINGUAL SER EVALUATION REPORT")
    report_lines.append(f"==================================\n")

    # --- 1. Per-Language Evaluation ---
    for lang_name, path in TEST_FILES.items():
        if not os.path.exists(path): 
            print(f"Skipping {lang_name} (File not found)")
            continue
            
        print(f"Evaluating {lang_name}...")
        ds = SimpleDataset(path)
        dl = DataLoader(ds, batch_size=32, shuffle=False)
        p, t = get_predictions(model, dl)
        
        # Save per-language CM
        safe_name = lang_name.split(" ")[0] 
        save_confusion_matrix(t, p, 
                            title=f"{lang_name} Confusion Matrix", 
                            filename=f"CM_{safe_name}.png")
        
        # Calculate Metrics - FIX: Pass labels=ALL_LABEL_IDS
        acc = accuracy_score(t, p)
        lang_report = classification_report(
            t, p, 
            labels=ALL_LABEL_IDS, 
            target_names=EMOTION_LABELS, 
            digits=4,
            zero_division=0 # Handle cases where a class is totally missing
        )
        
        report_lines.append(f"--- {lang_name} ---")
        report_lines.append(f"Accuracy: {acc*100:.2f}%")
        report_lines.append(lang_report)
        report_lines.append("\n")
        
        global_preds.extend(p)
        global_targets.extend(t)

    # --- 2. Global Evaluation ---
    print("Generating Global Metrics...")
    global_preds = np.array(global_preds)
    global_targets = np.array(global_targets)
    
    # Save Global CM
    save_confusion_matrix(global_targets, global_preds, 
                        title="Global (All Languages) Confusion Matrix", 
                        filename="CM_Global.png")
    
    global_acc = accuracy_score(global_targets, global_preds)
    global_report = classification_report(
        global_targets, global_preds, 
        labels=ALL_LABEL_IDS,
        target_names=EMOTION_LABELS, 
        digits=4,
        zero_division=0
    )
    
    report_lines.append(f"--- GLOBAL PERFORMANCE ---")
    report_lines.append(f"Overall Accuracy: {global_acc*100:.2f}%")
    report_lines.append(global_report)

    # --- 3. Save Text Report ---
    report_path = os.path.join(OUTPUT_DIR, "performance_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    
    print(f"\nEvaluation Complete!")
    print(f"All results saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()