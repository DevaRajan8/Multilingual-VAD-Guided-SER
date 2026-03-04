"""
Novel Components for Multilingual VAD-Guided Speech Emotion Recognition
=======================================================================

This module implements three novel contributions:

1. Adaptive Modality Gating (AMG)
   - Dynamically weights modalities based on emotion-discriminative confidence
   - Learns to emphasize the more reliable modality per sample

2. Affect Space Cross-Attention (ASCA)
   - Uses Valence-Arousal-Dominance space to guide attention weights
   - Incorporates psychological emotion theory into attention mechanism

3. Cross-Modal Alignment Loss (CMAL)
   - Learns emotion representations invariant across modalities
   - Enables robust cross-modal emotion understanding

Mathematical Formulations:
--------------------------

AMG (Adaptive Modality Gating):
    α_t = σ(W_g · [h_t; h_a; h_t ⊙ h_a] + b_g)
    h_fused = α_t · h_t + (1 - α_t) · h_a + β · (h_t ⊙ h_a)

ASCA (Affect Space Cross-Attention):
    A_guided = softmax(QK^T / √d_k + λ · M_VAD)
    M_VAD(i,j) = -||v_i - v_j||_2

CMAL (Cross-Modal Alignment Loss):
    L_CMAL = -log(exp(sim(z_t^i, z_a^i)/τ) / Σ_j exp(sim(z_t^i, z_a^j)/τ))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


# ============================================================
# 1. ADAPTIVE MODALITY GATING (AMG)
# ============================================================

class AdaptiveModalityGating(nn.Module):
    """
    Adaptive Modality Gating (AMG)

    Instead of fixed-weight fusion, AMG dynamically learns to weight
    modalities based on their emotion-discriminative confidence for each sample.

    Mathematical Formulation:
    -------------------------
    Given text features h_t ∈ R^d and audio features h_a ∈ R^d:

    1. Compute modality confidence gates:
       g = [h_t; h_a; h_t ⊙ h_a]  ∈ R^{3d}
       α_t = σ(W_g · g + b_g)     ∈ R^d  (text gate)
       α_a = σ(W_a · g + b_a)     ∈ R^d  (audio gate)

    2. Compute interaction term:
       h_inter = W_i · (h_t ⊙ h_a) + b_i  ∈ R^d

    3. Adaptive fusion:
       h_fused = α_t ⊙ h_t + α_a ⊙ h_a + β · h_inter

    where:
    - σ is the sigmoid function
    - ⊙ denotes element-wise multiplication (Hadamard product)
    - β is a learnable scalar controlling interaction strength

    Key Innovation:
    ---------------
    Unlike standard fusion that uses fixed weights, AMG learns sample-specific
    modality importance. This is crucial for SER because:
    - Some emotions are better expressed vocally (e.g., anger with high arousal)
    - Some emotions have clearer lexical markers (e.g., sadness with specific words)
    - The optimal modality weight varies per utterance
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Gate networks for modality confidence
        # Input: [h_t; h_a; h_t ⊙ h_a] -> 3 * hidden_dim
        self.text_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid()
        )

        self.audio_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid()
        )

        # Interaction projection
        self.interaction_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Learnable interaction strength
        self.beta = nn.Parameter(torch.tensor(0.5))

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            text_feat: [B, D] - Text features
            audio_feat: [B, D] - Audio features

        Returns:
            fused: [B, D] - Adaptively fused features
            aux_info: Dict with gate values for analysis
        """
        # Compute element-wise interaction
        interaction = text_feat * audio_feat  # [B, D]

        # Concatenate for gate input
        gate_input = torch.cat([text_feat, audio_feat, interaction], dim=-1)  # [B, 3D]

        # Compute modality-specific gates
        alpha_t = self.text_gate(gate_input)   # [B, D]
        alpha_a = self.audio_gate(gate_input)  # [B, D]

        # Project interaction
        h_inter = self.interaction_proj(interaction)  # [B, D]

        # Adaptive fusion with learned interaction strength
        fused = alpha_t * text_feat + alpha_a * audio_feat + self.beta * h_inter

        # Final projection
        output = self.output_proj(fused)

        # Return auxiliary info for analysis/visualization
        aux_info = {
            'text_gate': alpha_t.mean(dim=-1),   # [B] - average text confidence
            'audio_gate': alpha_a.mean(dim=-1),  # [B] - average audio confidence
            'beta': self.beta.item()
        }

        return output, aux_info


# ============================================================
# 2. AFFECT SPACE CROSS-ATTENTION (ASCA)
# ============================================================

class AffectSpaceCrossAttention(nn.Module):
    """
    Affect Space Cross-Attention (ASCA)

    Incorporates Valence-Arousal-Dominance (VAD) psychological theory
    into the cross-attention mechanism. VAD provides a continuous
    emotional representation that guides attention to focus on
    emotionally relevant cross-modal interactions.

    Mathematical Formulation:
    -------------------------
    Standard cross-attention:
        A = softmax(QK^T / √d_k)

    Affect-Space-Guided attention:
        1. Project features to VAD space:
           v_q = W_v · Q,  v_k = W_v · K  ∈ R^{N×3}

        2. Compute affect affinity matrix:
           M_VAD(i,j) = -||v_q[i] - v_k[j]||_2

        3. Guide attention with affect affinity:
           A_guided = softmax(QK^T / √d_k + λ · M_VAD)

    Key Innovation:
    ---------------
    By incorporating VAD structure, the attention mechanism learns to:
    - Focus on cross-modal features with similar emotional content
    - Reduce attention to emotionally irrelevant interactions
    - Leverage psychological emotion theory for better generalization

    The VAD space provides:
    - Valence: Positive vs negative emotional quality
    - Arousal: Level of activation/energy
    - Dominance: Level of control/power
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        vad_lambda: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.vad_lambda = nn.Parameter(torch.tensor(vad_lambda))

        # Standard attention projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # VAD projection (project features to 3D VAD space)
        self.vad_proj = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 3),  # 3D VAD space
            nn.Tanh()  # VAD values in [-1, 1]
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def compute_affect_affinity(
        self,
        query_vad: torch.Tensor,
        key_vad: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute affect-space-based affinity matrix.

        Args:
            query_vad: [B, Nq, 3] - VAD values for queries
            key_vad: [B, Nk, 3] - VAD values for keys

        Returns:
            affinity: [B, Nq, Nk] - affect affinity scores
        """
        # Compute pairwise L2 distance in VAD space
        # [B, Nq, 1, 3] - [B, 1, Nk, 3] -> [B, Nq, Nk, 3]
        diff = query_vad.unsqueeze(2) - key_vad.unsqueeze(1)
        distance = torch.norm(diff, p=2, dim=-1)  # [B, Nq, Nk]

        # Convert distance to affinity (negative distance)
        # Closer in VAD space = higher affinity
        affinity = -distance

        return affinity

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: [B, Nq, D] - Query features
            key: [B, Nk, D] - Key features
            value: [B, Nk, D] - Value features
            return_attention: Whether to return attention weights

        Returns:
            output: [B, Nq, D] - Attended features
            attn_weights: [B, H, Nq, Nk] - Attention weights (optional)
        """
        B, Nq, D = query.shape
        _, Nk, _ = key.shape

        # Project to Q, K, V
        Q = self.q_proj(query).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: [B, H, N, D_h]

        # Standard attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, Nq, Nk]

        # Compute affect affinity
        query_vad = self.vad_proj(query)  # [B, Nq, 3]
        key_vad = self.vad_proj(key)      # [B, Nk, 3]
        vad_affinity = self.compute_affect_affinity(query_vad, key_vad)  # [B, Nq, Nk]

        # Add VAD guidance to attention scores
        # Expand vad_affinity for all heads: [B, 1, Nq, Nk]
        vad_guidance = vad_affinity.unsqueeze(1)
        attn_scores_guided = attn_scores + self.vad_lambda * vad_guidance

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores_guided, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, V)  # [B, H, Nq, D_h]
        output = output.transpose(1, 2).contiguous().view(B, Nq, D)
        output = self.out_proj(output)

        # Residual connection and norm
        output = self.norm(query + self.dropout(output))

        if return_attention:
            return output, attn_weights
        return output, None


class AffectSpaceBidirectionalAttention(nn.Module):
    """
    Bidirectional Affect Space Cross-Attention for multimodal fusion.

    Applies affect-space-guided attention in both directions:
    - Audio → Text: Audio queries attend to text keys
    - Text → Audio: Text queries attend to audio keys
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        vad_lambda: float = 0.1
    ):
        super().__init__()

        # Audio attends to Text
        self.audio_to_text = AffectSpaceCrossAttention(
            dim, num_heads, dropout, vad_lambda
        )

        # Text attends to Audio
        self.text_to_audio = AffectSpaceCrossAttention(
            dim, num_heads, dropout, vad_lambda
        )

        # Feed-forward networks
        self.ffn_audio = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.ffn_text = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

        self.norm_audio = nn.LayerNorm(dim)
        self.norm_text = nn.LayerNorm(dim)

    def forward(
        self,
        audio_feat: torch.Tensor,
        text_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_feat: [B, Na, D] - Audio features
            text_feat: [B, Nt, D] - Text features

        Returns:
            audio_out: [B, Na, D] - Enhanced audio features
            text_out: [B, Nt, D] - Enhanced text features
        """
        # Cross-attention
        audio_cross, _ = self.audio_to_text(audio_feat, text_feat, text_feat)
        text_cross, _ = self.text_to_audio(text_feat, audio_feat, audio_feat)

        # Feed-forward with residual
        audio_out = self.norm_audio(audio_cross + self.ffn_audio(audio_cross))
        text_out = self.norm_text(text_cross + self.ffn_text(text_cross))

        return audio_out, text_out


# ============================================================
# 3. CROSS-MODAL ALIGNMENT LOSS (CMAL)
# ============================================================

class CrossModalAlignmentLoss(nn.Module):
    """
    Cross-Modal Alignment Loss (CMAL)

    Learns emotion representations that are invariant across modalities
    by pulling together text and audio representations of the same
    utterance while pushing apart different utterances.

    Mathematical Formulation:
    -------------------------
    Given text embeddings z_t ∈ R^{B×D} and audio embeddings z_a ∈ R^{B×D}:

    1. Cross-modal contrastive loss (text-to-audio):
       L_t2a = -1/B Σ_i log( exp(sim(z_t^i, z_a^i)/τ) / Σ_j exp(sim(z_t^i, z_a^j)/τ) )

    2. Cross-modal contrastive loss (audio-to-text):
       L_a2t = -1/B Σ_i log( exp(sim(z_a^i, z_t^i)/τ) / Σ_j exp(sim(z_a^i, z_t^j)/τ) )

    3. Final CMAL loss:
       L_CMAL = (L_t2a + L_a2t) / 2

    where:
    - sim(u, v) = u^T v / (||u|| ||v||)  (cosine similarity)
    - τ is temperature parameter
    - B is batch size

    Key Innovation:
    ---------------
    CMAL enforces that:
    1. Same utterance's text and audio should be close in embedding space
    2. Different utterances should be far apart regardless of modality
    3. The learned space captures emotion semantics shared across modalities

    This leads to:
    - Better cross-modal alignment
    - More robust representations under modality noise/missing
    - Improved few-shot transfer across domains
    """

    def __init__(self, temperature: float = 0.07, projection_dim: int = 128):
        super().__init__()
        self.temperature = temperature
        self.projection_dim = projection_dim

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute MICL loss.

        Args:
            text_features: [B, D] - Text embeddings (L2 normalized)
            audio_features: [B, D] - Audio embeddings (L2 normalized)
            labels: [B] - Optional emotion labels for supervised variant

        Returns:
            loss: Scalar MICL loss
            aux_info: Dict with component losses
        """
        device = text_features.device
        batch_size = text_features.shape[0]

        if batch_size <= 1:
            return torch.tensor(0.0, device=device), {}

        # L2 normalize features
        text_norm = F.normalize(text_features, p=2, dim=1)
        audio_norm = F.normalize(audio_features, p=2, dim=1)

        # Compute similarity matrix
        # sim_t2a[i,j] = similarity between text_i and audio_j
        sim_t2a = torch.matmul(text_norm, audio_norm.T) / self.temperature  # [B, B]
        sim_a2t = sim_t2a.T  # [B, B]

        # Labels: diagonal entries are positive pairs
        positive_labels = torch.arange(batch_size, device=device)

        # Cross-entropy loss (each row should match its diagonal)
        loss_t2a = F.cross_entropy(sim_t2a, positive_labels)
        loss_a2t = F.cross_entropy(sim_a2t, positive_labels)

        # Symmetric MICL loss
        loss = (loss_t2a + loss_a2t) / 2

        # If labels provided, add label-aware term
        if labels is not None:
            # Same-emotion pairs should also be similar
            label_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            # Exclude self (diagonal already handled)
            label_mask.fill_diagonal_(0)

            if label_mask.sum() > 0:
                # Additional loss: same-emotion samples should be close
                same_emotion_sim = sim_t2a * label_mask
                same_emotion_loss = -same_emotion_sim.sum() / (label_mask.sum() + 1e-8)
                loss = loss + 0.1 * same_emotion_loss

        aux_info = {
            'loss_t2a': loss_t2a.item(),
            'loss_a2t': loss_a2t.item(),
            'avg_positive_sim': torch.diagonal(sim_t2a).mean().item() * self.temperature,
            'avg_negative_sim': (sim_t2a.sum() - torch.diagonal(sim_t2a).sum()).item() / (batch_size * (batch_size - 1)) * self.temperature
        }

        return loss, aux_info



class CrossModalProjectionHead(nn.Module):
    """
    Projection head for CMAL that maps modality features to shared space.

    Uses separate projectors for each modality that map to a common
    low-dimensional space optimized for contrastive learning.
    """

    def __init__(self, input_dim: int, projection_dim: int = 128, hidden_dim: int = 256):
        super().__init__()

        # Text projector
        self.text_projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

        # Audio projector
        self.audio_projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(
        self,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project features to contrastive space.

        Args:
            text_feat: [B, D] - Text features
            audio_feat: [B, D] - Audio features

        Returns:
            text_proj: [B, projection_dim] - Projected text
            audio_proj: [B, projection_dim] - Projected audio
        """
        text_proj = self.text_projector(text_feat)
        audio_proj = self.audio_projector(audio_feat)

        return text_proj, audio_proj


# ============================================================
# 4. INTEGRATED NOVEL MODEL
# ============================================================

class MultilingualAffectFusionModel(nn.Module):
    """
    Multilingual Affect Fusion Model for Speech Emotion Recognition

    Integrates all three novel contributions:
    1. Affect Space Cross-Attention (ASCA) for cross-modal fusion
    2. Adaptive Modality Gating (AMG) for final fusion
    3. Cross-Modal Alignment Loss (CMAL) as auxiliary loss

    Architecture:
    -------------
    Input: Text features (BERT) + Audio features (emotion2vec)
           ↓
    Projection: Map to common dimension
           ↓
    Self-Attention: Unimodal refinement
           ↓
    Affect Space Cross-Attention: Cross-modal fusion with VAD guidance
           ↓
    Adaptive Modality Gating: Dynamic modality weighting
           ↓
    Cross-Modal Projection Head: Cross-modal contrastive learning
           ↓
    Classifier: Emotion prediction
    """

    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 1024,
        hidden_dim: int = 384,
        num_heads: int = 8,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.3,
        vad_lambda: float = 0.1,
        micl_dim: int = 128
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Input projections
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

        # Learnable modality embeddings
        self.text_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.audio_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Self-attention encoders
        self.text_self_attn = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.audio_self_attn = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        # Novel Component 1: Affect Space Cross-Attention layers
        self.vga_layers = nn.ModuleList([
            AffectSpaceBidirectionalAttention(hidden_dim, num_heads, dropout, vad_lambda)
            for _ in range(num_layers)
        ])

        # Novel Component 2: Adaptive Modality Gating
        self.eaaf = AdaptiveModalityGating(hidden_dim, dropout)

        # Novel Component 3: Cross-Modal Projection Head
        self.micl_projector = CrossModalProjectionHead(hidden_dim, micl_dim, hidden_dim)
        self.micl_loss = CrossModalAlignmentLoss(temperature=0.07)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        # VAD regression head (auxiliary)
        self.vad_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
            nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with all novel components.

        Args:
            text_feat: [B, D_text] or [B, N, D_text]
            audio_feat: [B, D_audio] or [B, N, D_audio]
            labels: [B] - Optional labels for MICL

        Returns:
            Dict containing logits, VAD predictions, MICL loss, etc.
        """
        # Add sequence dimension if needed
        if text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(1)
        if audio_feat.dim() == 2:
            audio_feat = audio_feat.unsqueeze(1)

        # Project to common dimension
        text_h = self.text_proj(text_feat)   # [B, Nt, D]
        audio_h = self.audio_proj(audio_feat)  # [B, Na, D]

        # Add modality embeddings
        text_h = text_h + self.text_embed
        audio_h = audio_h + self.audio_embed

        # Self-attention
        text_h = self.text_self_attn(text_h)
        audio_h = self.audio_self_attn(audio_h)

        # Affect Space Cross-Attention (Novel Component 1)
        for vga_layer in self.vga_layers:
            audio_h, text_h = vga_layer(audio_h, text_h)

        # Pool to get sequence-level features
        text_pooled = text_h.mean(dim=1)   # [B, D]
        audio_pooled = audio_h.mean(dim=1)  # [B, D]

        # Adaptive Modality Gating (Novel Component 2)
        fused, fusion_aux = self.eaaf(text_pooled, audio_pooled)

        # Cross-Modal Alignment Loss projection (Novel Component 3)
        text_proj, audio_proj = self.micl_projector(text_pooled, audio_pooled)
        micl_loss, micl_aux = self.micl_loss(text_proj, audio_proj, labels)

        # Classification
        logits = self.classifier(fused)
        probs = F.softmax(logits, dim=-1)

        # VAD prediction
        vad = self.vad_head(fused)

        return {
            'logits': logits,
            'probs': probs,
            'vad': vad,
            'features': fused,
            'text_features': text_pooled,
            'audio_features': audio_pooled,
            'micl_loss': micl_loss,
            'text_gate': fusion_aux['text_gate'],
            'audio_gate': fusion_aux['audio_gate'],
            'micl_aux': micl_aux
        }


# ============================================================
# 5. MULTI-TASK LOSS WITH NOVEL COMPONENTS
# ============================================================

class MultiObjectiveEmotionLoss(nn.Module):
    """
    Multi-task loss integrating all novel components.

    Total Loss:
        L = λ_cls * L_cls + λ_vad * L_vad + λ_cmal * L_cmal

    where:
    - L_cls: Cross-entropy classification loss
    - L_vad: VAD regression loss (MSE)
    - L_cmal: Cross-Modal Alignment Loss
    """

    def __init__(
        self,
        num_classes: int = 4,
        emotion_config: str = "iemocap_4",
        class_weights: Optional[torch.Tensor] = None,
        cls_weight: float = 1.0,
        vad_weight: float = 0.3,
        micl_weight: float = 0.2
    ):
        super().__init__()

        self.cls_weight = cls_weight
        self.vad_weight = vad_weight
        self.micl_weight = micl_weight

        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        self.mse_loss = nn.MSELoss()

        # VAD values per emotion
        self.vad_dict = {
            "iemocap_4": {
                0: [-0.666, 0.730, 0.314],   # anger
                1: [0.960, 0.648, 0.588],    # happiness
                2: [-0.062, -0.632, -0.286], # neutral
                3: [-0.896, -0.424, -0.672], # sadness
            }
        }.get(emotion_config, {})

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.

        Args:
            outputs: Model outputs dictionary
            labels: Ground truth labels

        Returns:
            Dictionary with individual and total losses
        """
        device = labels.device

        # Classification loss
        cls_loss = self.ce_loss(outputs['logits'], labels)

        # VAD regression loss
        vad_targets = torch.tensor([
            self.vad_dict.get(l.item(), [0, 0, 0]) for l in labels
        ], dtype=torch.float32, device=device)
        vad_loss = self.mse_loss(outputs['vad'], vad_targets)

        # MICL loss (already computed in forward)
        micl_loss = outputs['micl_loss']

        # Total loss
        total = (
            self.cls_weight * cls_loss +
            self.vad_weight * vad_loss +
            self.micl_weight * micl_loss
        )

        return {
            'total': total,
            'cls': cls_loss,
            'vad': vad_loss,
            'micl': micl_loss
        }
        