"""Composition géométrique et fusion des images dans un repère commun."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class StitchResult:
    panorama: np.ndarray | None
    transforms: list[np.ndarray]
    canvas_size: tuple[int, int] | None
    crop_box: tuple[int, int, int, int] | None
    error: str | None = None


def compose_to_reference(adjacent_homographies: list[np.ndarray], image_count: int) -> list[np.ndarray]:
    """Compose les H(i→i+1) vers l'image centrale, sans ré-échantillonner."""

    if image_count < 2 or len(adjacent_homographies) != image_count - 1:
        raise ValueError("Il faut exactement N-1 homographies adjacentes pour N images.")
    reference = image_count // 2
    transforms: list[np.ndarray | None] = [None] * image_count
    transforms[reference] = np.eye(3, dtype=np.float64)

    for i in range(reference - 1, -1, -1):
        transforms[i] = transforms[i + 1] @ adjacent_homographies[i]
    for i in range(reference + 1, image_count):
        try:
            inverse = np.linalg.inv(adjacent_homographies[i - 1])
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"Homographie {i} non inversible.") from exc
        transforms[i] = transforms[i - 1] @ inverse

    return [np.asarray(t, dtype=np.float64) for t in transforms]


def _image_corners(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    return np.float32([[[0, 0], [width, 0], [width, height], [0, height]]])


def _feather_map(height: int, width: int) -> np.ndarray:
    """Poids croissants vers le centre, y compris pour une image toute noire."""

    yy, xx = np.indices((height, width), dtype=np.float32)
    distance = np.minimum.reduce((xx + 1, width - xx, yy + 1, height - yy))
    maximum = float(distance.max())
    return distance / maximum if maximum > 0 else np.ones((height, width), np.float32)


def stitch_in_common_frame(
    images: list[np.ndarray],
    transforms: list[np.ndarray],
    blending: str = "Feather",
    max_megapixels: float = 100.0,
) -> StitchResult:
    """Projette une seule fois chaque image, fusionne par masque puis recadre."""

    if len(images) < 2 or len(images) != len(transforms):
        return StitchResult(None, transforms, None, None, "Il faut au moins deux images et une transformation par image.")
    if any(image is None or image.size == 0 for image in images):
        return StitchResult(None, transforms, None, None, "Une image d'entrée est vide.")

    try:
        transformed = [cv2.perspectiveTransform(_image_corners(img), matrix)[0] for img, matrix in zip(images, transforms)]
    except cv2.error as exc:
        return StitchResult(None, transforms, None, None, f"Transformation des coins impossible : {exc}")
    all_corners = np.vstack(transformed)
    if not np.isfinite(all_corners).all():
        return StitchResult(None, transforms, None, None, "Les limites projetées ne sont pas finies.")

    min_x, min_y = np.floor(all_corners.min(axis=0)).astype(np.int64)
    max_x, max_y = np.ceil(all_corners.max(axis=0)).astype(np.int64)
    width, height = int(max_x - min_x), int(max_y - min_y)
    if width <= 0 or height <= 0:
        return StitchResult(None, transforms, None, None, "Dimensions de panorama invalides.")
    if width * height > max_megapixels * 1_000_000 or width > 32766 or height > 32766:
        return StitchResult(
            None, transforms, (width, height), None,
            f"Canevas aberrant ({width}×{height}) : vérifiez les correspondances ou réduisez les images.",
        )

    translation = np.array([[1.0, 0.0, -min_x], [0.0, 1.0, -min_y], [0.0, 0.0, 1.0]])
    canvas_sum = np.zeros((height, width, 3), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    translated_transforms = []

    for image, transform in zip(images, transforms):
        matrix = translation @ transform
        translated_transforms.append(matrix)
        h, w = image.shape[:2]
        source_mask = np.full((h, w), 255, dtype=np.uint8)
        valid_mask = cv2.warpPerspective(source_mask, matrix, (width, height), flags=cv2.INTER_NEAREST) > 0
        warped = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR)
        if blending.lower().startswith("feather"):
            source_weight = _feather_map(h, w)
            weights = cv2.warpPerspective(source_weight, matrix, (width, height), flags=cv2.INTER_LINEAR)
            weights = np.where(valid_mask, np.maximum(weights, 1e-3), 0.0)
        else:
            weights = valid_mask.astype(np.float32)
        canvas_sum += warped.astype(np.float32) * weights[..., None]
        weight_sum += weights

    union_mask = weight_sum > 0
    if not np.any(union_mask):
        return StitchResult(None, translated_transforms, (width, height), None, "Aucun pixel valide après projection.")
    panorama = np.zeros_like(canvas_sum, dtype=np.uint8)
    panorama[union_mask] = np.clip(
        canvas_sum[union_mask] / weight_sum[union_mask, None], 0, 255
    ).astype(np.uint8)

    x, y, crop_w, crop_h = cv2.boundingRect(union_mask.astype(np.uint8))
    panorama = panorama[y : y + crop_h, x : x + crop_w]
    return StitchResult(panorama, translated_transforms, (width, height), (x, y, crop_w, crop_h))
