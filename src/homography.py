"""Estimation robuste des homographies avec RANSAC."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np


@dataclass
class HomographyResult:
    matrix: np.ndarray | None
    inlier_mask: np.ndarray | None
    inliers: int
    outliers: int
    elapsed_ms: float
    error: str | None = None

    @property
    def inlier_ratio(self) -> float:
        total = self.inliers + self.outliers
        return 0.0 if total == 0 else 100.0 * self.inliers / total


def estimate_homography(keypoints_a, keypoints_b, matches, ransac_threshold: float = 4.0) -> HomographyResult:
    """Calcule H telle que ``point_b ~ H @ point_a``."""

    start = perf_counter()
    if len(matches) < 4:
        return HomographyResult(
            None, None, 0, len(matches), (perf_counter() - start) * 1000.0,
            f"Homographie impossible : {len(matches)} bon(s) match(es), au moins 4 requis.",
        )
    src = np.float32([keypoints_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([keypoints_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, float(ransac_threshold))
    elapsed = (perf_counter() - start) * 1000.0
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        return HomographyResult(None, None, 0, len(matches), elapsed, "RANSAC n'a pas trouvé d'homographie exploitable.")

    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1e-10:
        return HomographyResult(None, None, 0, len(matches), elapsed, "L'homographie calculée est singulière.")
    flat_mask = mask.ravel().astype(bool)
    inliers = int(np.count_nonzero(flat_mask))
    if inliers < 4:
        return HomographyResult(None, flat_mask, inliers, len(matches) - inliers, elapsed, "Moins de 4 inliers après RANSAC.")
    return HomographyResult(matrix, flat_mask, inliers, len(matches) - inliers, elapsed)
