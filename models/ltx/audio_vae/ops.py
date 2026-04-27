"""Utilitaires de traitement audio pour le VAE audio."""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class AudioLatentShape:
    """Descripteur de forme pour les représentations latentes audio."""

    batch: int
    channels: int
    frames: int
    mel_bins: int


class PerChannelStatistics(nn.Module):
    """
    Statistiques par canal pour normaliser et dénormaliser la représentation latente.
    Ces statistiques sont calculées sur l'ensemble du jeu de données et stockées dans
    le checkpoint du modèle.
    """

    def __init__(self, latent_channels: int = 128) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        # Initialisation des buffers — seront chargés depuis les poids.
        # Underscores conservés pour la compatibilité du chargement de poids MLX.
        self.std_of_means = mx.ones((latent_channels,))
        self.mean_of_means = mx.zeros((latent_channels,))

    def un_normalize(self, x: mx.array) -> mx.array:
        """Dénormalise la représentation latente."""
        # Diffusion des statistiques pour correspondre à la forme de x
        # forme de x : (B, C, ...) ou (B, ..., C)
        std = self.std_of_means.astype(x.dtype)
        mean = self.mean_of_means.astype(x.dtype)
        return (x * std) + mean

    def normalize(self, x: mx.array) -> mx.array:
        """Normalise la représentation latente."""
        std = self.std_of_means.astype(x.dtype)
        mean = self.mean_of_means.astype(x.dtype)
        return (x - mean) / std


class AudioPatchifier:
    """
    Patchifier audio permettant de convertir entre les latents audio et les patchs.
    Combine les dimensions canaux et mel_bins pour les statistiques par canal.
    """

    def __init__(
        self,
        patch_size: int = 1,
        audio_latent_downsample_factor: int = 4,
        sample_rate: int = 16000,
        hop_length: int = 160,
        is_causal: bool = True,
    ):
        self.patch_size = patch_size
        self.audio_latent_downsample_factor = audio_latent_downsample_factor
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.is_causal = is_causal

    def patchify(self, x: mx.array) -> mx.array:
        """Convertit des latents audio en patchs.

        Forme d'entrée : (B, T, F, C) au format MLX (canaux à la fin)
        Forme de sortie : (B, T, C*F) — aplatie pour les statistiques par canal

        L'ordre de sortie est (c f) pour correspondre au "b c t f -> b t (c f)" de PyTorch.
        """
        # forme de x : (B, T, F, C) p. ex. (1, 68, 16, 8)
        b, t, f, c = x.shape
        # Transposition vers (B, T, C, F) pour respecter l'ordre (c f)
        x = mx.transpose(x, (0, 1, 3, 2))
        # Reshape vers (B, T, C*F) p. ex. (1, 68, 128)
        return x.reshape(b, t, c * f)

    def unpatchify(self, x: mx.array, latent_shape: AudioLatentShape) -> mx.array:
        """Reconvertit des patchs en latents audio.

        Forme d'entrée : (B, T, C*F)
        Forme de sortie : (B, T, F, C) au format MLX

        Inverse le "b t (c f) -> b c t f" de patchify, puis transpose vers le format MLX.
        """
        # forme de x : (B, T, C*F) p. ex. (1, 68, 128)
        b, t, cf = x.shape
        c = latent_shape.channels
        f = latent_shape.mel_bins
        # Reshape vers (B, T, C, F)
        x = x.reshape(b, t, c, f)
        # Transposition vers le format MLX (B, T, F, C)
        return mx.transpose(x, (0, 1, 3, 2))
