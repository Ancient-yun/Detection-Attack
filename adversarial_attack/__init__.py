"""Adversarial attack package for mmdetection object detection models.

Provides decision-based black-box L0 adversarial attacks
(SparseEvo, PointWise) adapted for object detection.
"""

from .metrics import compute_l0, compute_iou, compute_attack_success_rate
from .model_adapter import MMDetModelAdapter
from .sparse_evo import SpaEvoAtt
from .pointwise import PointWiseAtt
from .attack_pipeline import DetectionAttackPipeline

__all__ = [
    'compute_l0', 'compute_iou', 'compute_attack_success_rate',
    'MMDetModelAdapter', 'SpaEvoAtt', 'PointWiseAtt',
    'DetectionAttackPipeline',
]
