from __future__ import annotations

import cv2
import numpy as np
import pytest


def make_scene() -> np.ndarray:
    """Construit une scène reproductible riche en coins et en textures."""

    rng = np.random.default_rng(42)
    scene = rng.integers(10, 60, size=(360, 1000, 3), dtype=np.uint8)
    for index in range(55):
        x = int(rng.integers(15, 970))
        y = int(rng.integers(20, 330))
        color = tuple(int(value) for value in rng.integers(90, 255, size=3))
        if index % 2:
            cv2.circle(scene, (x, y), int(rng.integers(5, 22)), color, -1)
        else:
            cv2.rectangle(scene, (x, y), (min(999, x + 25), min(359, y + 20)), color, -1)
    for x in range(35, 950, 105):
        cv2.putText(scene, str(x), (x, 180 + (x % 3) * 45), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
    return scene


@pytest.fixture
def overlapping_images():
    scene = make_scene()
    return scene[:, :650].copy(), scene[:, 350:].copy()
