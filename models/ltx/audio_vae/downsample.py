"""Couches de sous-échantillonnage pour le VAE audio."""

from typing import Set, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..config import CausalityAxis
from .attention import AttentionType, make_attn
from .normalization import NormType
from .resnet import ResnetBlock


class Downsample(nn.Module):
    """
    Couche de sous-échantillonnage qui peut s'appuyer soit sur une convolution avec
    pas (stride), soit sur un pooling moyen. Prend en charge un padding standard ou
    causal pour le mode convolutionnel.
    """

    def __init__(
        self,
        in_channels: int,
        with_conv: bool,
        causality_axis: CausalityAxis = CausalityAxis.WIDTH,
    ) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis

        if self.causality_axis != CausalityAxis.NONE and not self.with_conv:
            raise ValueError("La causalité n'est prise en charge qu'avec `with_conv=True`.")

        if self.with_conv:
            # Sous-échantillonnage temporel effectué ici.
            # MLX ne propose pas de padding asymétrique sur conv, on le fait nous-mêmes.
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Passe avant avec sous-échantillonnage.
        Args:
            x : tenseur d'entrée de forme (N, H, W, C) au format MLX channels-last
        Returns:
            Tenseur sous-échantillonné
        """
        if self.with_conv:
            # Le tuple de padding suit l'ordre (left, right, top, bottom) côté PyTorch.
            # Pour mx.pad : [(avant_axe0, après_axe0), ...]
            # forme de x : (N, H, W, C) -> on pad sur les axes H et W
            if self.causality_axis == CausalityAxis.NONE:
                # pad : (gauche=0, droite=1, haut=0, bas=1)
                pad = [(0, 0), (0, 1), (0, 1), (0, 0)]
            elif self.causality_axis == CausalityAxis.WIDTH:
                # pad : (gauche=2, droite=0, haut=0, bas=1)
                pad = [(0, 0), (0, 1), (2, 0), (0, 0)]
            elif self.causality_axis == CausalityAxis.HEIGHT:
                # pad : (gauche=0, droite=1, haut=2, bas=0)
                pad = [(0, 0), (2, 0), (0, 1), (0, 0)]
            elif self.causality_axis == CausalityAxis.WIDTH_COMPATIBILITY:
                # pad : (gauche=1, droite=0, haut=0, bas=1)
                pad = [(0, 0), (0, 1), (1, 0), (0, 0)]
            else:
                raise ValueError(f"causality_axis invalide : {self.causality_axis}")

            x = mx.pad(x, pad, constant_values=0)
            x = self.conv(x)
        else:
            # Pooling moyen avec un noyau 2x2 et un pas de 2.
            # MLX ne propose pas avg_pool2d intégré, implémentation manuelle.
            # forme de x : (N, H, W, C)
            n, h, w, c = x.shape
            # Reshape vers (N, H//2, 2, W//2, 2, C) puis moyenne sur les dims de pooling
            x = x.reshape(n, h // 2, 2, w // 2, 2, c)
            x = mx.mean(x, axis=(2, 4))

        return x


def build_downsampling_path(
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
) -> tuple[dict, int]:
    """Construit le chemin de sous-échantillonnage avec blocs résiduels, attention et couches de réduction."""
    down_modules = {}
    curr_res = resolution
    in_ch_mult = (1, *tuple(ch_mult))
    block_in = ch

    for i_level in range(num_resolutions):
        stage = {}
        stage["block"] = {}
        stage["attn"] = {}
        block_in = ch * in_ch_mult[i_level]
        block_out = ch * ch_mult[i_level]

        for i_block in range(num_res_blocks):
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

        if i_level != num_resolutions - 1:
            stage["downsample"] = Downsample(
                block_in, resamp_with_conv, causality_axis=causality_axis
            )
            curr_res = curr_res // 2

        down_modules[i_level] = stage

    return down_modules, block_in
