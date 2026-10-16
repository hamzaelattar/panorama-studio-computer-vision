"""Orchestration d'un pipeline complet et collecte des métriques."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from .detectors import FeatureResult, detect_features
from .homography import HomographyResult, estimate_homography
from .matching import MatchResult, match_descriptors
from .stitching import StitchResult, compose_to_reference, stitch_in_common_frame


@dataclass
class PairResult:
    index_a: int
    index_b: int
    matching: MatchResult
    homography: HomographyResult


@dataclass
class PipelineResult:
    algorithm: str
    features: list[FeatureResult] = field(default_factory=list)
    pairs: list[PairResult] = field(default_factory=list)
    stitch: StitchResult | None = None
    total_ms: float = 0.0
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None and self.stitch is not None and self.stitch.panorama is not None

    def metrics(self) -> dict[str, float | int | str]:
        inliers = sum(pair.homography.inliers for pair in self.pairs)
        outliers = sum(pair.homography.outliers for pair in self.pairs)
        total = inliers + outliers
        return {
            "Keypoints détectés": sum(item.detected_count for item in self.features),
            "Descripteurs calculés": sum(item.descriptor_count for item in self.features),
            "Bons matches": sum(len(pair.matching.good_matches) for pair in self.pairs),
            "Inliers RANSAC": inliers,
            "Taux d'inliers (%)": 0.0 if total == 0 else 100.0 * inliers / total,
            "Temps de détection (ms)": sum(item.elapsed_ms for item in self.features),
            "Temps de matching (ms)": sum(item.matching.elapsed_ms for item in self.pairs),
            "Temps total (ms)": self.total_ms,
            "Panorama généré": "Oui" if self.success else "Non",
        }


def run_pipeline(
    images: list[np.ndarray],
    algorithm: str,
    detector_params: Mapping[str, Any] | None = None,
    ratio: float = 0.75,
    ransac_threshold: float = 4.0,
    blending: str = "Feather",
) -> PipelineResult:
    """Exécute les étapes et transforme toute erreur attendue en diagnostic lisible."""

    start = perf_counter()
    result = PipelineResult(algorithm=algorithm)
    if len(images) < 2:
        result.error = "Chargez au moins deux images."
        return result
    try:
        for index, image in enumerate(images):
            feature = detect_features(image, algorithm, detector_params)
            result.features.append(feature)
            if feature.descriptor_count == 0:
                result.error = f"Aucun descripteur exploitable dans l'image {index + 1}."
                return result

        matrices = []
        for index in range(len(images) - 1):
            left, right = result.features[index], result.features[index + 1]
            matching = match_descriptors(left.descriptors, right.descriptors, left.norm_type, ratio)
            homography = estimate_homography(
                left.keypoints,
                right.keypoints,
                matching.good_matches,
                ransac_threshold,
            )
            result.pairs.append(PairResult(index, index + 1, matching, homography))
            if matching.error:
                result.error = f"Paire {index + 1}→{index + 2} : {matching.error}"
                return result
            if homography.error or homography.matrix is None:
                result.error = f"Paire {index + 1}→{index + 2} : {homography.error}"
                return result
            matrices.append(homography.matrix)

        transforms = compose_to_reference(matrices, len(images))
        result.stitch = stitch_in_common_frame(images, transforms, blending=blending)
        if result.stitch.error:
            result.error = result.stitch.error
    except Exception as exc:  # L'interface doit rester disponible pour les autres méthodes.
        result.error = f"{type(exc).__name__} : {exc}"
    finally:
        result.total_ms = (perf_counter() - start) * 1000.0
    return result
