import cv2

from src.detectors import detect_features
from src.homography import estimate_homography
from src.matching import match_descriptors


def test_matching_and_homography_on_known_overlap(overlapping_images):
    left = detect_features(overlapping_images[0], "ORB", {"nfeatures": 2500})
    right = detect_features(overlapping_images[1], "ORB", {"nfeatures": 2500})
    matches = match_descriptors(left.descriptors, right.descriptors, left.norm_type, 0.78)
    assert matches.error is None
    assert matches.query_count > 50
    assert len(matches.good_matches) >= 12

    homography = estimate_homography(left.keypoints, right.keypoints, matches.good_matches, 3.0)
    assert homography.error is None
    assert homography.inliers >= 8
    # Le second crop commence 350 pixels plus à droite dans la scène globale.
    assert abs(homography.matrix[0, 2] + 350) < 8
    assert abs(homography.matrix[1, 2]) < 8


def test_missing_descriptors_returns_clear_error():
    result = match_descriptors(None, None, cv2.NORM_HAMMING)
    assert result.error is not None
    assert result.good_matches == []
