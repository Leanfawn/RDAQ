"""
RDAQ: DETR meets Refinement-Driven Adaptive Querying for Dense Aerial Imagery
Copyright (c) 2025 The RDAQ Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM (https://github.com/Intellindust-AI-Lab/DEIM)
Copyright(c) 2024 Shihua Huang. All Rights Reserved.
"""


from .rdaq import RDAQ

from .matcher import HungarianMatcher
from .hybrid_encoder import HybridEncoder

from .postprocessor import PostProcessor
from .rdaq_criterion import RDAQCriterion