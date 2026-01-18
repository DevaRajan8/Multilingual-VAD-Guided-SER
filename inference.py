"""
Manual Inference for ACL 2026 Multilingual SER
Usage: python inference.py --audio path/to/file.wav --text "What you said in the audio"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import soundfile as sf
import numpy as np
import argparse
from transformers import BertTokenizer, BertModel
from funasr import AutoModel
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION (Must match training)
# ==========================================
MODEL_PATH = "./bestmodels/best_model_run_1.pth" # Path to your saved model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Common-6 Mapping
EMOTION_LABELS = {
    0: "Anger",
    1: "Sadness",
    2: "Happiness",
    3: "Neutral",
    4: "Fear",
    5: "Disgust"
}

# ==========================================
# 2. MODEL ARCHITECTURE (Copy from main.py)
# ==========================================
# We need the exact class definitions to load the weights

class AttentivePooling(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.W = nn.Linear(input_dim, input_dim)
        self.w_context = nn.Linear(input_dim, 1, bias=False)

    def forward(self, features: torch.Tensor, mask=None) -> torch.Tensor:
        attn_scores = self.w_context(torch.tanh(self.W(features)))
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled = torch.sum(features * attn_weights, dim=1)
        return pooled

# (Simplified imports for the components if you have them in models/novel_components.py)
# If you have the 'models' folder, you can just import them. 
# Assuming you have the folder structure from previous steps:
from models.novel_components import (
    VADGuidedBidirectionalAttention,
    EmotionAwareAdaptiveFusion,
    MICLProjector
)

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
        
        self.vga_layers = nn.ModuleList([
            VADGuidedBidirectionalAttention(hidden_dim, num_heads, dropout, vad_lambda)
            for _ in range(num_layers)
        ])
        
        self.text_pool = AttentivePooling(hidden_dim)
        self.audio_pool = AttentivePooling(hidden_dim)
        self.eaaf = EmotionAwareAdaptiveFusion(hidden_dim, dropout)
        self.micl_projector = MICLProjector(hidden_dim, micl_dim, hidden_dim)

        self.vad_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
            nn.Tanh()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim), 
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, text_feat: torch.Tensor, audio_feat: torch.Tensor):
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
        fused, _ = self.eaaf(text_pooled, audio_pooled)
        vad_pred = self.vad_head(fused)
        fused_with_vad = torch.cat([fused, vad_pred], dim=-1)
        logits = self.classifier(fused_with_vad)
        probs = F.softmax(logits, dim=-1)
        return probs, vad_pred

# ==========================================
# 3. FEATURE EXTRACTORS
# ==========================================
print("Loading Feature Extractors...")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
bert_model = BertModel.from_pretrained('bert-base-multilingual-cased').to(DEVICE)
emotion2vec = AutoModel(model="iic/emotion2vec_plus_large", model_revision="v2.0.5")
print("Extractors Loaded.")

def extract_live_features(audio_path, text_input):
    """
    Takes raw audio path and text string, returns tensors for the model.
    """
    # 1. Audio Processing
    # Resample to 16k
    y, sr = librosa.load(audio_path, sr=16000)
    # Save to temporary buffer for emotion2vec to read
    temp_path = "temp_inference.wav"
    sf.write(temp_path, y, 16000)
    
    # emotion2vec extraction
    res = emotion2vec.generate(temp_path, output_dir=None, granularity="utterance")
    audio_emb = torch.tensor(res[0]['feats']).float().to(DEVICE) # [768]? Check dim
    
    # emotion2vec large is 1024 dim. 
    # Ensure dimension matches training (1024).
    
    # 2. Text Processing
    inputs = bert_tokenizer(text_input, return_tensors="pt", truncation=True, max_length=100, padding="max_length")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        text_out = bert_model(**inputs)
    text_emb = text_out.last_hidden_state[:, 0, :][0] # CLS token [768]

    return text_emb, audio_emb

# ==========================================
# 4. MAIN INFERENCE LOOP
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True, help="Path to wav file")
    parser.add_argument("--text", type=str, required=True, help="Transcribed text")
    args = parser.parse_args()

    # Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = ACL2026Model().to(DEVICE)
    
    # Load Weights
    # strict=False is sometimes needed if 'vad_targets_map' buffers were saved
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()

    # Extract
    print("Extracting features...")
    try:
        text_feat, audio_feat = extract_live_features(args.audio, args.text)
        
        # Add Batch Dim [1, D]
        text_feat = text_feat.unsqueeze(0)
        audio_feat = audio_feat.unsqueeze(0)

        # Predict
        print("Running inference...")
        with torch.no_grad():
            probs, vad_vals = model(text_feat, audio_feat)
            
        # Result
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
        emotion = EMOTION_LABELS[pred_idx]
        
        print("\n" + "="*30)
        print(f"PREDICTION: {emotion.upper()}")
        print(f"Confidence: {confidence:.2%}")
        print(f"VAD Est.  : Val={vad_vals[0][0]:.2f}, Aro={vad_vals[0][1]:.2f}, Dom={vad_vals[0][2]:.2f}")
        print("="*30)
        
        print("\nClass Probabilities:")
        for idx, score in enumerate(probs[0]):
            print(f"  {EMOTION_LABELS[idx]:<10}: {score.item():.4f}")

    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    main()