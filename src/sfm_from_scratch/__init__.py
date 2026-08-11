"""Incremental Structure from Motion from scratch.

Educational but real: SIFT features, essential-matrix initialization, PnP
growth, and sparse bundle adjustment — the same skeleton COLMAP implements
industrially — in a few hundred readable lines.
"""

from sfm_from_scratch.core.dataset import Dataset, View
from sfm_from_scratch.core.reconstruction import Reconstruction

__all__ = ["Dataset", "Reconstruction", "View"]
