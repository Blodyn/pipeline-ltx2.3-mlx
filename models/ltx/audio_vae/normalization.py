"""Couches de normalisation pour le VAE audio."""

from enum import Enum

import mlx.core as mx
import mlx.nn as nn


class NormType(Enum):
    """Types de couche de normalisation : GROUP (GroupNorm) ou PIXEL (RMS norm par position)."""

    GROUP = "group"
    PIXEL = "pixel"


class PixelNorm(nn.Module):
    """
    Couche de normalisation RMS par pixel (par position).
    Pour chaque élément le long de la dimension choisie, cette couche normalise le tenseur
    par la racine carrée de la moyenne des carrés de ses valeurs sur cette dimension :
        y = x / sqrt(mean(x^2, dim=dim, keepdim=True) + eps)
    """

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        """
        Args:
            dim : dimension le long de laquelle calculer la RMS (typiquement les canaux).
            eps : petite constante ajoutée pour la stabilité numérique.
        """
        super().__init__()
        self.dim = dim
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        """Applique une normalisation RMS le long de la dimension configurée."""
        mean_sq = mx.mean(x**2, axis=self.dim, keepdims=True)
        rms = mx.sqrt(mean_sq + self.eps)
        return x / rms


def build_normalization_layer(
    in_channels: int, *, num_groups: int = 32, normtype: NormType = NormType.GROUP
) -> nn.Module:
    """
    Crée une couche de normalisation en fonction du type de normalisation.
    Args:
        in_channels : nombre de canaux d'entrée
        num_groups : nombre de groupes pour la normalisation par groupe
        normtype : type de normalisation : "group" ou "pixel"
    Returns:
        Une couche de normalisation
    """
    if normtype == NormType.GROUP:
        return nn.GroupNorm(
            num_groups=num_groups, dims=in_channels, eps=1e-6, affine=True
        )
    if normtype == NormType.PIXEL:
        # Pour le format MLX channels-last (B, H, W, C), on normalise sur les canaux (dim=-1).
        # PyTorch utilise dim=1 pour le format channels-first (B, C, H, W).
        return PixelNorm(dim=-1, eps=1e-6)
    raise ValueError(f"Type de normalisation invalide : {normtype}")
