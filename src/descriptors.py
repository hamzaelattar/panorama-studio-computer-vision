"""Création des descripteurs utilisés par les détecteurs."""

from __future__ import annotations

import cv2


class BriefUnavailableError(RuntimeError):
    """Signale une installation OpenCV sans le module contrib BRIEF."""


def create_brief(bytes_count: int = 32, use_orientation: bool = False):
    """Crée BRIEF après une vérification explicite de ``xfeatures2d``.

    Le message d'erreur indique volontairement la correction à appliquer : il est
    fréquent que ``opencv-python`` masque ``opencv-contrib-python`` sous Windows.
    """

    xfeatures = getattr(cv2, "xfeatures2d", None)
    factory = getattr(xfeatures, "BriefDescriptorExtractor_create", None)
    if factory is None:
        raise BriefUnavailableError(
            "BRIEF est indisponible. Désinstallez toutes les distributions OpenCV "
            "puis installez uniquement opencv-contrib-python (voir README)."
        )
    return factory(bytes=int(bytes_count), use_orientation=bool(use_orientation))


def descriptor_kind(algorithm: str) -> str:
    """Retourne une description lisible du type de descripteur."""

    return "flottant (128 valeurs)" if algorithm == "SIFT" else "binaire"
