"""Encodeur Video VAE pour la conversion image-vers-vidéo de LTX-2.

L'encodeur compresse des images/vidéos en représentations latentes.
Utilisé pour le conditionnement I2V (image-to-video) en encodant l'image
d'entrée dans l'espace latent, qui peut ensuite servir à conditionner la
génération vidéo.
"""

import mlx.core as mx

from .video_vae import VideoEncoder


def encode_image(
    image: mx.array,
    encoder: VideoEncoder,
) -> mx.array:
    """Encode une image unique vers l'espace latent.

    Args:
        image : tenseur image de forme (H, W, 3) dans [0, 1] ou (B, H, W, 3)
        encoder : encodeur VAE chargé

    Returns:
        Tenseur latent de forme (1, 128, 1, H//32, W//32)
    """
    # Ajout de la dimension batch si nécessaire
    if image.ndim == 3:
        image = mx.expand_dims(image, axis=0)  # (1, H, W, 3)

    # Conversion (B, H, W, C) -> (B, C, H, W)
    image = mx.transpose(image, (0, 3, 1, 2))  # (B, 3, H, W)

    # Normalisation vers [-1, 1]
    if image.max() > 1.0:
        image = image / 255.0
    image = image * 2.0 - 1.0

    # Ajout de la dimension temporelle : (B, C, H, W) -> (B, C, 1, H, W)
    image = mx.expand_dims(image, axis=2)  # (B, 3, 1, H, W)

    # Encodage
    latent = encoder(image)

    return latent
