import cv2
import numpy as np
import pytest

from src.detectors import detect_features


@pytest.mark.parametrize(
    ("algorithm", "expected_size", "expected_dtype", "expected_norm"),
    [
        ("SIFT", 128, np.float32, cv2.NORM_L2),
        ("FAST + BRIEF", 32, np.uint8, cv2.NORM_HAMMING),
        ("ORB", 32, np.uint8, cv2.NORM_HAMMING),
    ],
)
def test_three_feature_methods(overlapping_images, algorithm, expected_size, expected_dtype, expected_norm):
    result = detect_features(overlapping_images[0], algorithm)
    assert result.detected_count > 20
    assert result.descriptor_count > 20
    assert result.descriptors.shape[1] == expected_size
    assert result.descriptors.dtype == expected_dtype
    assert result.norm_type == expected_norm


def test_orb_uses_hamming2_when_wta_changes(overlapping_images):
    result = detect_features(overlapping_images[0], "ORB", {"wta_k": 4})
    assert result.norm_type == cv2.NORM_HAMMING2


def test_blank_image_does_not_crash():
    result = detect_features(np.zeros((120, 180, 3), np.uint8), "ORB")
    assert result.descriptors is None
    assert result.descriptor_count == 0
