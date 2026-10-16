"""Détection des points d'intérêt et calcul de leurs descripteurs."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

import cv2
import numpy as np

from .descriptors import create_brief

ALGORITHMS = ("SIFT", "FAST + BRIEF", "ORB")


@dataclass
class FeatureResult:
    algorithm: str
    keypoints: list
    descriptors: np.ndarray | None
    detected_count: int
    elapsed_ms: float
    norm_type: int

    @property
    def descriptor_count(self) -> int:
        return 0 if self.descriptors is None else int(len(self.descriptors))

    @property
    def descriptor_size(self) -> int:
        if self.descriptors is None or self.descriptors.ndim != 2:
            return 0
        return int(self.descriptors.shape[1])


def _gray(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("L'image est vide.")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("Format d'image non pris en charge.")


def detect_features(
    image: np.ndarray,
    algorithm: str,
    params: Mapping[str, Any] | None = None,
) -> FeatureResult:
    """Détecte les points et calcule les descripteurs d'une image BGR.

    ``detected_count`` reste distinct du nombre de descripteurs pour FAST :
    BRIEF peut rejeter les points trop proches des bords.
    """

    if algorithm not in ALGORITHMS:
        raise ValueError(f"Algorithme inconnu : {algorithm}")
    p = dict(params or {})
    gray = _gray(image)
    start = perf_counter()

    if algorithm == "SIFT":
        detector = cv2.SIFT_create(
            nfeatures=int(p.get("nfeatures", 0)),
            contrastThreshold=float(p.get("contrast_threshold", 0.04)),
            edgeThreshold=float(p.get("edge_threshold", 10)),
            sigma=float(p.get("sigma", 1.6)),
        )
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        detected_count = len(keypoints)
        norm_type = cv2.NORM_L2

    elif algorithm == "FAST + BRIEF":
        detector = cv2.FastFeatureDetector_create(
            threshold=int(p.get("fast_threshold", 20)),
            nonmaxSuppression=bool(p.get("nonmax_suppression", True)),
        )
        detected = detector.detect(gray, None)
        detected_count = len(detected)
        max_features = int(p.get("max_features", 3000))
        if max_features > 0 and len(detected) > max_features:
            detected = sorted(detected, key=lambda kp: kp.response, reverse=True)[:max_features]
        descriptor = create_brief(
            bytes_count=int(p.get("brief_bytes", 32)),
            use_orientation=bool(p.get("brief_orientation", False)),
        )
        keypoints, descriptors = descriptor.compute(gray, detected)
        keypoints = keypoints or []
        norm_type = cv2.NORM_HAMMING

    else:  # ORB
        wta_k = int(p.get("wta_k", 2))
        detector = cv2.ORB_create(
            nfeatures=int(p.get("nfeatures", 2000)),
            scaleFactor=float(p.get("scale_factor", 1.2)),
            nlevels=int(p.get("nlevels", 8)),
            WTA_K=wta_k,
            fastThreshold=int(p.get("fast_threshold", 20)),
        )
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        detected_count = len(keypoints)
        norm_type = cv2.NORM_HAMMING2 if wta_k in (3, 4) else cv2.NORM_HAMMING

    elapsed_ms = (perf_counter() - start) * 1000.0
    return FeatureResult(
        algorithm=algorithm,
        keypoints=list(keypoints or []),
        descriptors=descriptors,
        detected_count=int(detected_count),
        elapsed_ms=elapsed_ms,
        norm_type=norm_type,
    )
