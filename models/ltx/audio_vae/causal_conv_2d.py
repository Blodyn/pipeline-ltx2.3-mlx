"""Convolutions 2D causales pour le VAE audio."""

from typing import Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from ..config import CausalityAxis


def _pair(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Convertit un int ou un tuple en paire de tuple."""
    if isinstance(x, int):
        return (x, x)
    return x


class CausalConv2d(nn.Module):
    """
    Convolution 2D causale.
    Cette couche garantit que la sortie au temps `t` ne dépend que des entrées au
    temps `t` et antérieures. Elle obtient ce comportement en appliquant un padding
    asymétrique sur la dimension temporelle avant la convolution.

    Note : MLX Conv2d attend des entrées de forme (N, H, W, C) — canaux à la fin.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
    ) -> None:
        super().__init__()

        self.causality_axis = causality_axis

        # On s'assure que kernel_size et dilation sont des tuples
        kernel_size = _pair(kernel_size)
        dilation = _pair(dilation)

        # Calcul des dimensions de padding
        pad_h = (kernel_size[0] - 1) * dilation[0]
        pad_w = (kernel_size[1] - 1) * dilation[1]

        # Stockage du padding pour application manuelle
        # Ordre de mx.pad : [(avant_axe0, après_axe0), (avant_axe1, après_axe1), ...]
        # Pour le format (N, H, W, C) : axe 1 = H (hauteur), axe 2 = W (largeur)
        if self.causality_axis == CausalityAxis.NONE:
            # Non causal : padding symétrique
            self.padding = (
                pad_h // 2,
                pad_h - pad_h // 2,
                pad_w // 2,
                pad_w - pad_w // 2,
            )
        elif self.causality_axis in (
            CausalityAxis.WIDTH,
            CausalityAxis.WIDTH_COMPATIBILITY,
        ):
            # Causal sur la largeur : padding à gauche (avant l'axe largeur)
            self.padding = (pad_h // 2, pad_h - pad_h // 2, pad_w, 0)
        elif self.causality_axis == CausalityAxis.HEIGHT:
            # Causal sur la hauteur : padding en haut (avant l'axe hauteur)
            self.padding = (pad_h, 0, pad_w // 2, pad_w - pad_w // 2)
        else:
            raise ValueError(f"causality_axis invalide : {causality_axis}")

        # La convolution interne n'utilise aucun padding, on le gère manuellement
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Passe avant avec padding causal.
        Args:
            x : tenseur d'entrée de forme (N, H, W, C) au format MLX channels-last
        Returns:
            Tenseur de sortie après convolution causale
        """
        # Application du padding causal avant convolution
        # Format de padding : (pad_h_top, pad_h_bottom, pad_w_left, pad_w_right)
        pad_h_top, pad_h_bottom, pad_w_left, pad_w_right = self.padding

        if any(p > 0 for p in self.padding):
            # mx.pad attend : [(avant_0, après_0), (avant_1, après_1), ...]
            # Pour (N, H, W, C) : axe 0=N, axe 1=H, axe 2=W, axe 3=C
            x = mx.pad(
                x,
                [(0, 0), (pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right), (0, 0)],
            )

        return self.conv(x)


def make_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int]],
    stride: int = 1,
    padding: Union[int, Tuple[int, int], None] = None,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    causality_axis: CausalityAxis | None = None,
) -> nn.Module:
    """
    Crée une couche de convolution 2D, causale ou non causale.
    Args:
        in_channels : nombre de canaux d'entrée
        out_channels : nombre de canaux de sortie
        kernel_size : taille du noyau de convolution
        stride : pas de la convolution
        padding : padding (si None, calculé en fonction du flag causal)
        dilation : taux de dilatation
        groups : nombre de groupes pour la convolution groupée
        bias : si on utilise un biais
        causality_axis : dimension le long de laquelle appliquer la causalité.
    Returns:
        Une couche Conv2d standard ou CausalConv2d
    """
    if causality_axis is not None:
        # En convolution causale, le padding est géré en interne par CausalConv2d
        return CausalConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            groups,
            bias,
            causality_axis,
        )
    else:
        # En convolution non causale, on utilise un padding symétrique si non précisé
        if padding is None:
            if isinstance(kernel_size, int):
                padding = kernel_size // 2
            else:
                padding = tuple(k // 2 for k in kernel_size)

        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
