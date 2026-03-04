"""
Novel Components for Multilingual VAD-Guided Speech Emotion Recognition

This package contains the novel contributions:
1. Adaptive Modality Gating (AMG)
2. Affect Space Cross-Attention (ASCA)
3. Cross-Modal Alignment Loss (CMAL)
"""

from .novel_components import (
    AdaptiveModalityGating,
    AffectSpaceCrossAttention,
    AffectSpaceBidirectionalAttention,
    CrossModalAlignmentLoss,
    CrossModalProjectionHead,
    MultilingualAffectFusionModel,
    MultiObjectiveEmotionLoss
)

__all__ = [
    'AdaptiveModalityGating',
    'AffectSpaceCrossAttention',
    'AffectSpaceBidirectionalAttention',
    'CrossModalAlignmentLoss',
    'CrossModalProjectionHead',
    'MultilingualAffectFusionModel',
    'MultiObjectiveEmotionLoss'
]
