import numpy as np

from src.comparison import run_pipeline
from src.stitching import compose_to_reference, stitch_in_common_frame


def test_direct_stitch_has_expected_crop(overlapping_images):
    translation_left_to_right = np.array(
        [[1.0, 0.0, -350.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    transforms = compose_to_reference([translation_left_to_right], 2)
    result = stitch_in_common_frame(list(overlapping_images), transforms, blending="Feather")
    assert result.error is None
    assert result.panorama is not None
    assert 995 <= result.panorama.shape[1] <= 1001
    assert result.panorama.shape[0] == 360


def test_black_pixels_are_not_used_as_empty_mask():
    left = np.zeros((80, 120, 3), np.uint8)
    right = np.full((80, 120, 3), 200, np.uint8)
    transform = np.array([[1.0, 0.0, -60.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    result = stitch_in_common_frame([left, right], compose_to_reference([transform], 2))
    assert result.error is None
    # Le noir participe bien à la moyenne dans le chevauchement.
    overlap_pixel = result.panorama[40, 90]
    assert np.all(overlap_pixel < 195)


def test_full_orb_pipeline(overlapping_images):
    result = run_pipeline(
        list(overlapping_images), "ORB", {"nfeatures": 2500}, ratio=0.78, ransac_threshold=3.0
    )
    assert result.success, result.error
    assert result.stitch.panorama.shape[1] > 900
