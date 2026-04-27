"""Couches de sur-échantillonnage pour le VAE audio."""

from typing import Set, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..config import CausalityAxis
from .attention import AttentionType, make_attn
from .causal_conv_2d import make_conv2d
from .normalization import NormType
from .resnet import ResnetBlock


def nearest_neighbor_upsample(x: mx.array, scale_factor: int = 2) -> mx.array:
    """
    Sur-échantillonnage par plus proche voisin pour des tenseurs 4D.
    Args:
        x : tenseur d'entrée de forme (N, H, W, C)
        scale_factor : facteur de sur-échantillonnage
    Returns:
        Tenseur sur-échantillonné de forme (N, H*scale_factor, W*scale_factor, C)
    """
    n, h, w, c = x.shape
    # Répétition selon la hauteur et la largeur
    x = mx.repeat(x, scale_factor, axis=1)
    x = mx.repeat(x, scale_factor, axis=2)
    return x


class Upsample(nn.Module):
    """Couche de sur-échantillonnage avec convolution optionnelle."""

    def __init__(
        self,
        in_channels: int,
        with_conv: bool,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
    ) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis
        if self.with_conv:
            self.conv = make_conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                causality_axis=causality_axis,
            )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Passe avant avec sur-échantillonnage.
        Args:
            x : tenseur d'entrée de forme (N, H, W, C) au format MLX channels-last
        Returns:
            Tenseur sur-échantillonné
        """
        # Sur-échantillonnage 2x par plus proche voisin
        x = nearest_neighbor_upsample(x, scale_factor=2)

        if self.with_conv:
            x = self.conv(x)
            # On supprime le PREMIER élément sur l'axe causal pour annuler le padding de l'encodeur,
            # tout en gardant une longueur de la forme 1 + 2 * n.
            # Exemple : si l'entrée est [0, 1, 2], après interpolation on obtient [0, 0, 1, 1, 2, 2].
            # La convolution causale ajoute un padding initial : [-, -, 0, 0, 1, 1, 2, 2],
            # ce qui fait que les éléments de sortie reposent sur les fenêtres suivantes :
            # 0 : [-,-,0]
            # 1 : [-,0,0]
            # 2 : [0,0,1]
            # 3 : [0,1,1]
            # 4 : [1,1,2]
            # 5 : [1,2,2]
            # On note que les deux premiers éléments de sortie ne dépendent que du premier élément
            # d'entrée, alors que tous les autres dépendent de deux éléments. On peut donc
            # supprimer le premier élément pour annuler le padding (plutôt que le dernier).
            # C'est un no-op pour les convolutions non causales.
            if self.causality_axis == CausalityAxis.NONE:
                pass  # x reste inchangé
            elif self.causality_axis == CausalityAxis.HEIGHT:
                x = x[:, 1:, :, :]
            elif self.causality_axis == CausalityAxis.WIDTH:
                x = x[:, :, 1:, :]
            elif self.causality_axis == CausalityAxis.WIDTH_COMPATIBILITY:
                pass  # x reste inchangé
            else:
                raise ValueError(f"causality_axis invalide : {self.causality_axis}")

        return x


def build_upsampling_path(
    *,
    ch: int,
    ch_mult: Tuple[int, ...],
    num_resolutions: int,
    num_res_blocks: int,
    resolution: int,
    temb_channels: int,
    dropout: float,
    norm_type: NormType,
    causality_axis: CausalityAxis,
    attn_type: AttentionType,
    attn_resolutions: Set[int],
    resamp_with_conv: bool,
    initial_block_channels: int,
) -> tuple[dict, int]:
    """Construit le chemin de sur-échantillonnage avec blocs résiduels, attention et couches de remontée."""
    up_modules = {}
    block_in = initial_block_channels
    curr_res = resolution // (2 ** (num_resolutions - 1))

    for level in reversed(range(num_resolutions)):
        stage = {}
        stage["block"] = {}
        stage["attn"] = {}
        block_out = ch * ch_mult[level]

        for i_block in range(num_res_blocks + 1):
            stage["block"][i_block] = ResnetBlock(
                in_channels=block_in,
                out_channels=block_out,
                temb_channels=temb_channels,
                dropout=dropout,
                norm_type=norm_type,
                causality_axis=causality_axis,
            )
            block_in = block_out
            if curr_res in attn_resolutions:
                stage["attn"][i_block] = make_attn(
                    block_in, attn_type=attn_type, norm_type=norm_type
                )

        if level != 0:
            stage["upsample"] = Upsample(
                block_in, resamp_with_conv, causality_axis=causality_axis
            )
            curr_res *= 2

        up_modules[level] = stage

    return up_modules, block_in
