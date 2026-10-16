"""Fonctions de visualisation OpenCV, sans dépendance à Streamlit."""

from __future__ import annotations

import cv2
import numpy as np


def draw_keypoints(image: np.ndarray, keypoints) -> np.ndarray:
    return cv2.drawKeypoints(
        image,
        keypoints,
        None,
        color=(60, 230, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


def draw_matches(image_a, keypoints_a, image_b, keypoints_b, matches, limit: int = 120) -> np.ndarray:
    selected = sorted(matches, key=lambda match: match.distance)[:limit]
    return cv2.drawMatches(
        image_a,
        keypoints_a,
        image_b,
        keypoints_b,
        selected,
        None,
        matchColor=(70, 220, 120),
        singlePointColor=(100, 100, 100),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def draw_inlier_matches(image_a, keypoints_a, image_b, keypoints_b, matches, inlier_mask) -> np.ndarray:
    if inlier_mask is None:
        return draw_matches(image_a, keypoints_a, image_b, keypoints_b, matches)
    return cv2.drawMatches(
        image_a,
        keypoints_a,
        image_b,
        keypoints_b,
        matches,
        None,
        matchColor=(40, 220, 90),
        singlePointColor=None,
        matchesMask=np.asarray(inlier_mask, dtype=np.uint8).tolist(),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
