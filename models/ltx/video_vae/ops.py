"""Opérations pour le Video VAE."""


import mlx.core as mx
import mlx.nn as nn


def patchify(x: mx.array, patch_size_hw: int = 4, patch_size_t: int = 1) -> mx.array:
    """Convertit une vidéo en patchs.

    Déplace les pixels spatiaux des dimensions H, W vers la dimension canaux.

    Args:
        x : tenseur d'entrée de forme (B, C, F, H, W)
        patch_size_hw : taille de patch spatiale
        patch_size_t : taille de patch temporelle

    Returns:
        Tenseur patché de forme (B, C * patch_size_hw^2, F, H/patch_size_hw, W/patch_size_hw)
    """
    b, c, f, h, w = x.shape

    # Vérification que les dimensions sont divisibles
    assert h % patch_size_hw == 0 and w % patch_size_hw == 0
    assert f % patch_size_t == 0

    # Nouvelles dimensions
    new_h = h // patch_size_hw
    new_w = w // patch_size_hw
    new_f = f // patch_size_t
    new_c = c * patch_size_hw * patch_size_hw * patch_size_t

    # Reshape : (B, C, F, H, W) -> (B, C, F/pt, pt, H/ph, ph, W/pw, pw)
    x = mx.reshape(
        x, (b, c, new_f, patch_size_t, new_h, patch_size_hw, new_w, patch_size_hw)
    )

    # Permutation : (B, C, F', pt, H', ph, W', pw) -> (B, C, pt, pw, ph, F', H', W')
    # einops PyTorch utilise (c, p, r, q) = (c, temporel, largeur, hauteur), il faut donc pw avant ph
    x = mx.transpose(x, (0, 1, 3, 7, 5, 2, 4, 6))

    # Reshape : (B, C, pt, pw, ph, F', H', W') -> (B, C*pt*pw*ph, F', H', W')
    x = mx.reshape(x, (b, new_c, new_f, new_h, new_w))

    return x


def unpatchify(x: mx.array, patch_size_hw: int = 4, patch_size_t: int = 1) -> mx.array:
    """Reconvertit des patchs en vidéo.

    Inverse de patchify — ramène les pixels de la dimension canaux vers le spatial.
    Correspond à l'einops PyTorch : "b (c p r q) f h w -> b c (f p) (h q) (w r)"
    où p=patch_size_t, r=patch_size_hw (largeur), q=patch_size_hw (hauteur)

    Args:
        x : tenseur patché de forme (B, C * patch_size_hw^2, F, H, W)
        patch_size_hw : taille de patch spatiale
        patch_size_t : taille de patch temporelle

    Returns:
        Tenseur vidéo de forme (B, C, F * patch_size_t, H * patch_size_hw, W * patch_size_hw)
    """
    b, c_packed, f, h, w = x.shape

    # Calcul du nombre original de canaux
    c = c_packed // (patch_size_hw * patch_size_hw * patch_size_t)

    # Reshape : (B, C*pt*pr*pq, F, H, W) -> (B, C, pt, pr, pq, F, H, W)
    # avec pt=temporel, pr=patch_largeur (r), pq=patch_hauteur (q)
    # L'agencement des canaux côté PyTorch est (c, p, r, q) = (c, temporel, largeur, hauteur)
    x = mx.reshape(x, (b, c, patch_size_t, patch_size_hw, patch_size_hw, f, h, w))

    # Permutation pour entrelacer les patchs avec les dimensions spatiales :
    # (B, C, pt, pr, pq, F, H, W) -> (B, C, F, pt, H, pq, W, pr)

    x = mx.transpose(x, (0, 1, 5, 2, 6, 4, 7, 3))

    # Reshape : (B, C, F, pt, H, pq, W, pr) -> (B, C, F*pt, H*pq, W*pr)
    x = mx.reshape(x, (b, c, f * patch_size_t, h * patch_size_hw, w * patch_size_hw))

    return x


class PerChannelStatistics(nn.Module):

    def __init__(self, latent_channels: int = 128):

        super().__init__()
        self.latent_channels = latent_channels

        # Moyenne et écart-type par canal apprenables
        self.mean = mx.zeros((latent_channels,))
        self.std = mx.ones((latent_channels,))

    def normalize(self, x: mx.array) -> mx.array:
        """Normalise les latents à l'aide des statistiques par canal.

        Args:
            x : tenseur d'entrée de forme (B, C, ...)

        Returns:
            Tenseur normalisé
        """
        # Mise en forme de mean et std pour la diffusion : (C,) -> (1, C, 1, 1, 1)
        dtype = x.dtype
        # Cast en float32 pour la précision
        mean = self.mean.astype(mx.float32).reshape(1, -1, 1, 1, 1)
        std = self.std.astype(mx.float32).reshape(1, -1, 1, 1, 1)

        return ((x - mean) / std).astype(dtype)

    def un_normalize(self, x: mx.array) -> mx.array:
        """Dénormalise les latents à l'aide des statistiques par canal.

        Args:
            x : tenseur normalisé de forme (B, C, ...)

        Returns:
            Tenseur dénormalisé
        """
        dtype = x.dtype
        # Cast en float32 pour la précision
        mean = self.mean.astype(mx.float32).reshape(1, -1, 1, 1, 1)
        std = self.std.astype(mx.float32).reshape(1, -1, 1, 1, 1)

        return (x * std + mean).astype(dtype)
