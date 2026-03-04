"""
Multilingual Multimodal SER Training
File: main.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score, f1_score
)
import warnings
import numpy as np
import random
from collections import Counter
import pickle
import argparse
import os
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json

# Assuming models.novel_components is in the path
from models.novel_components import (
    AffectSpaceBidirectionalAttention,
    AdaptiveModalityGating,
    CrossModalProjectionHead
)

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    """Configuration for ACL 2026 Enhanced experiments."""
    # Feature dimensions
    text_dim: int = 768      # Multilingual BERT
    audio_dim: int = 1024    # emotion2vec
    hidden_dim: int = 384
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.4

    num_classes: int = 6
    emotion_config: str = "common_6"

    # Training
    batch_size: int = 32     # Increased slightly for combined data
    lr: float = 2e-5
    weight_decay: float = 0.05
    epochs: int = 100
    patience: int = 15
    warmup_ratio: float = 0.1

    # Multi-run evaluation
    num_runs: int = 5
    seed: int = 42

    # Loss weights
    cls_weight: float = 1.0
    vad_weight: float = 0.3
    supcon_weight: float = 0.2

    # Novel component settings
    vad_lambda: float = 0.1
    supcon_temp: float = 0.07
    micl_dim: int = 128
    gamma: float = 2.0


EMOTION_LABELS = {
    # 0: Anger, 1: Sadness, 2: Happiness, 3: Neutral, 4: Fear, 5: Disgust
    "common_6": ["anger", "sadness", "happiness", "neutral", "fear", "disgust"],
}


VAD_CONFIGS = {
    "common_6": {
        0: [-0.430, 0.800, 0.500],   # anger
        1: [-0.800, -0.700, -0.650], # sadness
        2: [0.960, 0.650, 0.590],    # happiness
        3: [0.000, 0.000, 0.000],    # neutral
        4: [-0.640, 0.600, -0.650],  # fear
        5: [-0.600, 0.350, -0.400],  # disgust
    },
}

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# DATASET (UPDATED FOR MULTIPLE SOURCES)
# ============================================================

class MultimodalEmotionDataset(Dataset):
    """Dataset that can load and combine multiple pickle files."""
    def __init__(self, data_paths: Union[str, List[str]]):
        super().__init__()
        
        if isinstance(data_paths, str):
            data_paths = [data_paths]
            
        self.data = []
        for path in data_paths:
            if not os.path.exists(path):
                print(f"Warning: File {path} not found. Skipping.")
                continue
            
            print(f"Loading data from: {path}")
            with open(path, 'rb') as f:
                loaded_data = pickle.load(f)
                self.data.extend(loaded_data)
        
        print(f"Total samples loaded: {len(self.data)}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        
        def to_float_tensor(x):
            if not torch.is_tensor(x):
                return torch.tensor(x, dtype=torch.float32)
            return x.clone().detach().float()

        text = to_float_tensor(item['text_embed'])
        audio = to_float_tensor(item['audio_embed'])
        
        label = item['label']
        if not torch.is_tensor(label):
            label = torch.tensor(int(label), dtype=torch.long)
        else:
            label = label.clone().detach().long()

        return text, audio, label

def get_class_weights(dataset: Dataset, device: torch.device) -> torch.Tensor:
    """Calculate class weights for imbalanced 6-class data."""
    labels = []
    # Iterate safely regardless of dataset size
    for i in range(len(dataset)):
        labels.append(int(dataset[i][2].item()))
        
    counts = Counter(labels)
    total = len(labels)
    num_classes = 6 
    
    weights = []
    for i in range(num_classes):
        count = counts.get(i, 0)
        if count > 0:
            weights.append(total / (num_classes * count))
        else:
            weights.append(1.0)
            
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ============================================================
# LAYERS (Unchanged from your specification)
# ============================================================

class AttentivePooling(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.W = nn.Linear(input_dim, input_dim)
        self.w_context = nn.Linear(input_dim, 1, bias=False)
        
    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_scores = self.w_context(torch.tanh(self.W(features)))
        if mask is not None:
             attn_scores = attn_scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled = torch.sum(features * attn_weights, dim=1)
        return pooled

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=device)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        logits_mask = torch.scatter(
            torch.ones_like(mask), 
            1, 
            torch.arange(batch_size).view(-1, 1).to(device), 
            0
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class ACL2026Model(nn.Module):
    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 1024,
        hidden_dim: int = 384,
        num_heads: int = 8,
        num_layers: int = 2,
        num_classes: int = 6,
        dropout: float = 0.3,
        vad_lambda: float = 0.1,
        micl_dim: int = 128
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Projections
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.text_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.audio_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        self.text_self_attn = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.audio_self_attn = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        
        # Cross-Modal Fusion (ASCA)
        self.vga_layers = nn.ModuleList([
            AffectSpaceBidirectionalAttention(hidden_dim, num_heads, dropout, vad_lambda)
            for _ in range(num_layers)
        ])
        
        # Pooling
        self.text_pool = AttentivePooling(hidden_dim)
        self.audio_pool = AttentivePooling(hidden_dim)
        
        # Fusion (AMG)
        self.eaaf = AdaptiveModalityGating(hidden_dim, dropout)
        
        # Cross-Modal Projector
        self.micl_projector = CrossModalProjectionHead(hidden_dim, micl_dim, hidden_dim)

        # VAD Head
        self.vad_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
            nn.Tanh()
        )
        
        # Residual Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim), 
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, text_feat: torch.Tensor, audio_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        if text_feat.dim() == 2: text_feat = text_feat.unsqueeze(1)
        if audio_feat.dim() == 2: audio_feat = audio_feat.unsqueeze(1)
        
        text_h = self.text_proj(text_feat) + self.text_embed
        audio_h = self.audio_proj(audio_feat) + self.audio_embed
        
        text_h = self.text_self_attn(text_h)
        audio_h = self.audio_self_attn(audio_h)
        
        for vga_layer in self.vga_layers:
            audio_h, text_h = vga_layer(audio_h, text_h)
            
        text_pooled = self.text_pool(text_h)
        audio_pooled = self.audio_pool(audio_h)
        
        fused, fusion_aux = self.eaaf(text_pooled, audio_pooled)
        text_proj, audio_proj = self.micl_projector(text_pooled, audio_pooled)
        
        vad_pred = self.vad_head(fused)
        fused_with_vad = torch.cat([fused, vad_pred], dim=-1)
        
        logits = self.classifier(fused_with_vad)
        probs = F.softmax(logits, dim=-1)
        
        return {
            'logits': logits,
            'probs': probs,
            'vad': vad_pred,
            'text_proj': text_proj,
            'audio_proj': audio_proj,
            'text_gate': fusion_aux['text_gate'],
            'audio_gate': fusion_aux['audio_gate']
        }

# ============================================================
# TRAIN & EVAL
# ============================================================

class ACL2026Loss(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        emotion_config: str = "common_6",
        class_weights: Optional[torch.Tensor] = None,
        cls_weight: float = 1.0,
        vad_weight: float = 0.3,
        supcon_weight: float = 0.2,
        gamma: float = 2.0,
        supcon_temp: float = 0.07
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.vad_weight = vad_weight
        self.supcon_weight = supcon_weight
        
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=gamma)
        self.mse_loss = nn.MSELoss()
        self.supcon_loss = SupConLoss(temperature=supcon_temp)
        
        self.vad_targets_map = VAD_CONFIGS.get(emotion_config, {})

    def forward(self, outputs: Dict, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        device = labels.device
        
        # 1. Focal Loss
        cls_loss = self.focal_loss(outputs['logits'], labels)
        
        # 2. VAD Regression
        vad_targets = torch.tensor([
            self.vad_targets_map.get(l.item(), [0, 0, 0]) for l in labels
        ], dtype=torch.float32, device=device)
        vad_loss = self.mse_loss(outputs['vad'], vad_targets)
        
        # 3. SupCon Loss
        text_norm = F.normalize(outputs['text_proj'], p=2, dim=1)
        audio_norm = F.normalize(outputs['audio_proj'], p=2, dim=1)
        supcon_t = self.supcon_loss(text_norm, labels)
        supcon_a = self.supcon_loss(audio_norm, labels)
        supcon_loss_val = (supcon_t + supcon_a) / 2
        
        total = (
            self.cls_weight * cls_loss +
            self.vad_weight * vad_loss +
            self.supcon_weight * supcon_loss_val
        )
        
        return {
            'total': total,
            'cls': cls_loss,
            'vad': vad_loss,
            'supcon': supcon_loss_val
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, emotion_labels: List[str]) -> Dict:
    # Only compute macro metrics over classes that actually appear in y_true
    # This prevents Disgust (0 samples in IEMOCAP test) from dragging down Macro-F1 and UA
    present_labels = sorted(set(y_true.tolist()))
    excluded = [emotion_labels[i] for i in range(len(emotion_labels)) if i not in present_labels]
    if excluded:
        print(f"  [Metrics] Excluding zero-support classes from macro metrics: {excluded}")

    wa = accuracy_score(y_true, y_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ua = balanced_accuracy_score(y_true, y_pred)   # already excludes zero-support classes
    wf1      = f1_score(y_true, y_pred, average='weighted', labels=present_labels, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro',    labels=present_labels, zero_division=0)

    return {
        'WA': wa, 'UA': ua, 'WF1': wf1, 'Macro_F1': macro_f1,
        'excluded_classes': excluded
    }

def evaluate(model, dataloader, device, config):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for text_feat, audio_feat, labels in dataloader:
            text_feat, audio_feat = text_feat.to(device), audio_feat.to(device)
            outputs = model(text_feat, audio_feat)
            preds = outputs['probs'].argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), EMOTION_LABELS[config.emotion_config])
    return metrics, {}

def train_single_run(config: Config, train_dataset: Dataset, val_dataset: Dataset, run_idx: int, device: torch.device):
    set_seed(config.seed + run_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    model = ACL2026Model(
        text_dim=config.text_dim,
        audio_dim=config.audio_dim,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        dropout=config.dropout,
        vad_lambda=config.vad_lambda,
        micl_dim=config.micl_dim
    ).to(device)
    
    class_weights = get_class_weights(train_dataset, device)
    criterion = ACL2026Loss(
        num_classes=config.num_classes,
        emotion_config=config.emotion_config,
        class_weights=class_weights,
        cls_weight=config.cls_weight,
        vad_weight=config.vad_weight,
        supcon_weight=config.supcon_weight,
        gamma=config.gamma,
        supcon_temp=config.supcon_temp
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.lr * 10,
        total_steps=len(train_loader) * config.epochs,
        pct_start=config.warmup_ratio
    )
    
    best_ua = 0.0
    patience_counter = 0
    best_model_state = None
    
    print(f"Starting training run {run_idx+1}...")
    
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        for text_feat, audio_feat, labels in train_loader:
            text_feat, audio_feat, labels = text_feat.to(device), audio_feat.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(text_feat, audio_feat)
            
            losses = criterion(outputs, labels)
            losses['total'].backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += losses['total'].item()
            
        # Validation
        val_res, _ = evaluate(model, val_loader, device, config)
        
        print(f"Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.4f} | Val UA: {val_res['UA']:.4f}")
        
        if val_res['UA'] > best_ua:
            best_ua = val_res['UA']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= config.patience:
            print("Early stopping triggered.")
            break
            
    if best_model_state:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        
    final_res, _ = evaluate(model, val_loader, device, config)
    return final_res, model

def main():
    parser = argparse.ArgumentParser(description="ACL 2026 Multilingual SER")
    
    # UPDATED: 'nargs=+' allows passing multiple files for training
    parser.add_argument("--train", nargs='+', default=[
        "/dist_home/suryansh/sharukesh/speech/features_common_6/IEMOCAP_Common6_train.pkl",
        "/dist_home/suryansh/sharukesh/speech/features_common_6/EmoDB_Common6_train.pkl",
        "/dist_home/suryansh/sharukesh/speech/features_common_6/EMOVO_Common6_train.pkl",
        "/dist_home/suryansh/sharukesh/speech/features_common_6/SUBESCO_Common6_train.pkl",
        "/dist_home/suryansh/sharukesh/speech/features_common_6/RAVDESS_Common6_train.pkl",
    ])
    parser.add_argument("--val", nargs='+', default=[
        "/dist_home/suryansh/sharukesh/speech/features_common_6/IEMOCAP_Common6_val.pkl",
        "/dist_home/suryansh/sharukesh/speech/features_common_6/EmoDB_Common6_val.pkl",
        "/dist_home/suryansh/sharukesh/speech/features_common_6/EMOVO_Common6_val.pkl",
        "/dist_home/suryansh/sharukesh/speech/features_common_6/SUBESCO_Common6_val.pkl",
        "/dist_home/suryansh/sharukesh/speech/features_common_6/RAVDESS_Common6_val.pkl",
    ])
    parser.add_argument("--test", nargs='+', default=[
        "/dist_home/suryansh/sharukesh/speech/features_common_6/IEMOCAP_Common6_test.pkl",
        "/dist_home/suryansh/sharukesh/speech/features_common_6/EmoDB_Common6_test.pkl",
        "/dist_home/suryansh/sharukesh/speech/features_common_6/EMOVO_Common6_test.pkl",
        "/dist_home/suryansh/sharukesh/speech/features_common_6/SUBESCO_Common6_test.pkl",
        "/dist_home/suryansh/sharukesh/speech/features_common_6/RAVDESS_Common6_test.pkl",
    ])
    
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--num_runs",     type=int,   default=5)
    parser.add_argument("--output",       type=str,   default="Multilingual_results.json")

    # Ablation flags — override loss weights without editing code
    parser.add_argument("--vad_weight",    type=float, default=0.3,
                        help="Weight for VAD regression loss (0.0 = ablate VAD guidance)")
    parser.add_argument("--supcon_weight", type=float, default=0.2,
                        help="Weight for supervised contrastive loss (0.0 = ablate SupCon)")
    parser.add_argument("--cls_weight",   type=float, default=1.0,
                        help="Weight for focal classification loss")
    
    args = parser.parse_args()
    
    config = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_runs=args.num_runs,
        emotion_config="common_6",
        vad_weight=args.vad_weight,
        supcon_weight=args.supcon_weight,
        cls_weight=args.cls_weight,
    )
    print(f"Loss weights → cls={config.cls_weight}  vad={config.vad_weight}  supcon={config.supcon_weight}")
    
    # Load and concatenate datasets
    print("Initializing Training Datasets...")
    train_ds = MultimodalEmotionDataset(args.train)
    print("Initializing Validation Datasets...")
    val_ds = MultimodalEmotionDataset(args.val)
    print("Initializing Test Datasets...")
    test_ds = MultimodalEmotionDataset(args.test)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_results = []
    best_run_idx = 0       # tracks which run achieved the highest val UA
    best_run_ua  = -1.0

    for i in range(config.num_runs):
        res, model = train_single_run(config, train_ds, val_ds, i, device)
        all_results.append(res)

        save_path = f"best_model_run_{i}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

        # Track which run yielded the best validation UA
        if res['UA'] > best_run_ua:
            best_run_ua  = res['UA']
            best_run_idx = i

    # ── Validation Stats ──────────────────────────────────────────────────────
    uas = [r['UA'] for r in all_results]
    print(f"\nValidation Results ({config.num_runs} runs):")
    print(f"Mean UA    : {np.mean(uas)*100:.2f}% ± {np.std(uas)*100:.2f}%")
    print(f"Best Run   : run {best_run_idx}  (Val UA = {best_run_ua*100:.2f}%)")

    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)

    # ── Test Set Evaluation (auto-select best checkpoint by val UA) ───────────
    best_ckpt = f"best_model_run_{best_run_idx}.pth"
    print(f"\n── Test Set Evaluation ──")
    print(f"Loading best checkpoint: {best_ckpt}  (run {best_run_idx}, Val UA={best_run_ua*100:.2f}%)")

    best_model = ACL2026Model(
        text_dim=config.text_dim,
        audio_dim=config.audio_dim,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        dropout=config.dropout,
        vad_lambda=config.vad_lambda,
        micl_dim=config.micl_dim
    ).to(device)
    best_model.load_state_dict(torch.load(best_ckpt, map_location=device))

    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
    test_metrics, _ = evaluate(best_model, test_loader, device, config)

    print(f"Test WA       : {test_metrics['WA']*100:.2f}%")
    print(f"Test UA       : {test_metrics['UA']*100:.2f}%")
    print(f"Test WF1      : {test_metrics['WF1']*100:.2f}%")
    print(f"Test Macro-F1 : {test_metrics['Macro_F1']*100:.2f}%")

    # Save test results alongside the val results JSON
    test_output_path = args.output.replace(".json", "_test.json")
    with open(test_output_path, 'w') as f:
        json.dump({**test_metrics, "best_run": best_run_idx, "best_run_val_ua": best_run_ua}, f, indent=2)
    print(f"Test results saved to {test_output_path}")
        
if __name__ == "__main__":
    main()