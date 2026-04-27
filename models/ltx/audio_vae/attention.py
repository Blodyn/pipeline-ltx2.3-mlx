"""Blocs d'attention pour le VAE audio."""

from enum import Enum

import mlx.core as mx
import mlx.nn as nn

from .normalization import NormType, build_normalization_layer


class AttentionType(Enum):
    """Énumération précisant le type de mécanisme d'attention."""

    VANILLA = "vanilla"
    LINEAR = "linear"
    NONE = "none"


class AttnBlock(nn.Module):
    """Bloc d'auto-attention pour le VAE audio."""

    def __init__(
        self,
        in_channels: int,
        norm_type: NormType = NormType.GROUP,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        self.norm = build_normalization_layer(in_channels, normtype=norm_type)
        # Utilisation de Conv2d avec kernel_size=1 pour les projections Q, K, V
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Passe avant à travers le bloc d'attention.
        Args:
            x : tenseur d'entrée de forme (B, H, W, C) au format MLX channels-last
        Returns:
            Tenseur de sortie après application de l'attention (connexion résiduelle)
        """
        h_ = x
        h_ = self.norm(h_)

        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Calcul de l'attention
        # forme de x : (B, H, W, C)
        b, h, w, c = q.shape

        # Reshape pour l'attention : (B, H*W, C)
        q = q.reshape(b, h * w, c)
        k = k.reshape(b, h * w, c)
        v = v.reshape(b, h * w, c)

        # Attention : Q @ K^T / sqrt(d)
        # q : (B, HW, C), k : (B, HW, C) -> k^T : (B, C, HW)
        # w_ : (B, HW, HW)
        scale = float(c) ** (-0.5)
        w_ = mx.matmul(q, k.transpose(0, 2, 1)) * scale
        w_ = mx.softmax(w_, axis=-1)

        # Application de l'attention sur les valeurs
        # w_ : (B, HW, HW), v : (B, HW, C) -> h_ : (B, HW, C)
        h_ = mx.matmul(w_, v)

        # Retour aux dimensions spatiales
        h_ = h_.reshape(b, h, w, c)

        h_ = self.proj_out(h_)

        return x + h_


class Identity(nn.Module):
    """Module identité qui renvoie l'entrée inchangée."""

    def __call__(self, x: mx.array) -> mx.array:
        return x


def make_attn(
    in_channels: int,
    attn_type: AttentionType = AttentionType.VANILLA,
    norm_type: NormType = NormType.GROUP,
) -> nn.Module:
    """
    Crée un module d'attention selon le type demandé.
    Args:
        in_channels : nombre de canaux d'entrée
        attn_type : type de mécanisme d'attention
        norm_type : type de normalisation
    Returns:
        Module d'attention
    """
    if attn_type == AttentionType.VANILLA:
        return AttnBlock(in_channels, norm_type=norm_type)
    elif attn_type == AttentionType.NONE:
        return Identity()
    elif attn_type == AttentionType.LINEAR:
        raise NotImplementedError(
            f"Le type d'attention {attn_type.value} n'est pas encore pris en charge."
        )
    else:
        raise ValueError(f"Type d'attention inconnu : {attn_type}")
