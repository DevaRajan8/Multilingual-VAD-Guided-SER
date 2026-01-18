"""
Novel Components for Multimodal Speech Emotion Recognition

This package contains the novel contributions for ACL 2026:
1. Emotion-Aware Adaptive Fusion (EAAF)
2. VAD-Guided Cross-Attention (VGA)
3. Modality-Invariant Contrastive Learning (MICL)
"""

from .novel_components import (
    EmotionAwareAdaptiveFusion,
    VADGuidedCrossAttention,
    VADGuidedBidirectionalAttention,
    ModalityInvariantContrastiveLoss,
    MICLProjector,
    NovelMultimodalSER,
    NovelMultiTaskLoss
)

__all__ = [
    'EmotionAwareAdaptiveFusion',
    'VADGuidedCrossAttention',
    'VADGuidedBidirectionalAttention',
    'ModalityInvariantContrastiveLoss',
    'MICLProjector',
    'NovelMultimodalSER',
    'NovelMultiTaskLoss'
]
