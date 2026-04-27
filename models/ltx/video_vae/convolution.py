from enum import Enum
from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn


class PaddingModeType(Enum):
    ZEROS = "zeros"
    REFLECT = "reflect"


def reflect_pad_2d(x: mx.array, pad_h: int, pad_w: int) -> mx.array:
    """Applique un padding « reflect » sur les dimensions spatiales d'un tenseur 5D.

    Args:
        x : tenseur d'entrée de forme (B, D, H, W, C) — canaux à la fin
        pad_h : padding sur la dimension hauteur
        pad_w : padding sur la dimension largeur

    Returns:
        Tenseur paddé
    """
    if pad_h == 0 and pad_w == 0:
        return x

    # Padding hauteur (axe 2)
    if pad_h > 0:
        # Indices de réflexion — on exclut la bordure
        top_pad = x[:, :, 1 : pad_h + 1, :, :][:, :, ::-1, :, :]  # Inversion de la portion haute
        bottom_pad = x[:, :, -pad_h - 1 : -1, :, :][
            :, :, ::-1, :, :
        ]  # Inversion de la portion basse
        x = mx.concatenate([top_pad, x, bottom_pad], axis=2)

    # Padding largeur (axe 3)
    if pad_w > 0:
        left_pad = x[:, :, :, 1 : pad_w + 1, :][:, :, :, ::-1, :]  # Inversion de la portion gauche
        right_pad = x[:, :, :, -pad_w - 1 : -1, :][
            :, :, :, ::-1, :
        ]  # Inversion de la portion droite
        x = mx.concatenate([left_pad, x, right_pad], axis=3)

    return x


def make_conv_nd(
    dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, ...]],
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...], str] = 0,
    causal: bool = False,
    spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
) -> nn.Module:

    if dims == 2:
        return CausalConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            causal=causal,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif dims == 3:
        return CausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            causal=causal,
            spatial_padding_mode=spatial_padding_mode,
        )
    else:
        raise ValueError(f"Nombre de dimensions non pris en charge : {dims}")


class CausalConv3d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int], str] = 0,
        causal: bool = False,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        super().__init__()

        self.causal = causal
        self.spatial_padding_mode = spatial_padding_mode

        # Normalisation de kernel_size et stride en tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)

        self.kernel_size = kernel_size
        self.stride = stride
        self.time_kernel_size = kernel_size[0]

        # Calcul du padding spatial (le temporel est géré séparément par réplication de frames)
        height_pad = kernel_size[1] // 2
        width_pad = kernel_size[2] // 2
        self.spatial_padding = (height_pad, width_pad)

        # Création de la convolution de base (sans padding, on le gère manuellement)
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # Padding géré manuellement
            bias=True,
        )

    def __call__(self, x: mx.array, causal: Optional[bool] = None) -> mx.array:

        use_causal = causal if causal is not None else self.causal

        # Application du padding temporel par réplication de frames
        # uniquement si kernel_size > 1
        if self.time_kernel_size > 1:
            if use_causal:
                # Causal : on réplique la première frame kernel_size-1 fois au début
                first_frame_pad = mx.repeat(
                    x[:, :, :1, :, :], self.time_kernel_size - 1, axis=2
                )
                x = mx.concatenate([first_frame_pad, x], axis=2)
            else:
                # Non causal : on réplique la première frame au début, la dernière à la fin
                pad_size = (self.time_kernel_size - 1) // 2
                if pad_size > 0:
                    first_frame_pad = mx.repeat(x[:, :, :1, :, :], pad_size, axis=2)
                    last_frame_pad = mx.repeat(x[:, :, -1:, :, :], pad_size, axis=2)
                    x = mx.concatenate([first_frame_pad, x, last_frame_pad], axis=2)

        # Transposition vers channels last : (B, C, D, H, W) -> (B, D, H, W, C)
        x = mx.transpose(x, (0, 2, 3, 4, 1))

        # Application du padding spatial
        pad_h, pad_w = self.spatial_padding
        if pad_h > 0 or pad_w > 0:
            if self.spatial_padding_mode == PaddingModeType.REFLECT:
                # Padding « reflect » sur les dimensions spatiales
                x = reflect_pad_2d(x, pad_h, pad_w)
            else:
                # Padding zéros sur les dimensions spatiales
                pad_width = [
                    (0, 0),  # Batch
                    (0, 0),  # D (temporel — déjà paddé)
                    (pad_h, pad_h),  # H
                    (pad_w, pad_w),  # W
                    (0, 0),  # C
                ]
                x = mx.pad(x, pad_width)

        # Convolution avec découpage en tronçons pour les gros tenseurs.
        # NB : on a choisi le tronçonnage car la conv3d MLX échoue autour de 33 frames en spatial 192x192
        x = self._chunked_conv3d(x)

        # Transposition de retour vers channels first : (B, D, H, W, C) -> (B, C, D, H, W)
        x = mx.transpose(x, (0, 4, 1, 2, 3))

        return x

    def _chunked_conv3d(self, x: mx.array) -> mx.array:
        """Applique conv3d par tronçons temporels pour contourner un bug MLX avec les gros tenseurs.

        Args:
            x : tenseur d'entrée de forme (B, D, H, W, C) au format channels-last

        Returns:
            Tenseur de sortie après conv3d
        """
        b, d, h, w, c = x.shape

        total_elements = d * h * w * c
        max_safe_elements = 30 * 192 * 192 * 128  # ~140 M éléments par tronçon

        if total_elements <= max_safe_elements:
            return self.conv(x)

        elements_per_frame = h * w * c
        max_frames_per_chunk = max(1, max_safe_elements // elements_per_frame)
        chunk_size = min(max_frames_per_chunk, 24)  # Plafond à 24 frames par tronçon

        kernel_t = self.time_kernel_size

        overlap = kernel_t - 1

        expected_output_frames = d - overlap

        outputs = []
        out_idx = 0

        # Traitement des tronçons
        in_start = 0
        while out_idx < expected_output_frames:
            remaining = expected_output_frames - out_idx
            out_frames_this_chunk = min(chunk_size, remaining)

            in_frames_needed = out_frames_this_chunk + overlap
            in_end = min(in_start + in_frames_needed, d)

            chunk = x[:, in_start:in_end, :, :, :]

            chunk_out = self.conv(chunk)
            mx.eval(chunk_out)

            outputs.append(chunk_out)

            out_idx += chunk_out.shape[1]
            in_start += chunk_out.shape[1]

        # Concaténation de tous les tronçons
        if len(outputs) == 1:
            return outputs[0]
        return mx.concatenate(outputs, axis=1)


class CausalConv2d(nn.Module):
    """Convolution 2D avec padding causal optionnel."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int], str] = 0,
        causal: bool = False,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        """Initialise CausalConv2d."""
        super().__init__()

        self.causal = causal
        self.spatial_padding_mode = spatial_padding_mode

        # Normalisation de kernel_size et stride
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)

        self.kernel_size = kernel_size
        self.stride = stride

        # Calcul du padding
        if isinstance(padding, str) and padding == "same":
            self.padding = (
                (kernel_size[0] - 1) // 2,
                (kernel_size[1] - 1) // 2,
            )
        elif isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=True,
        )

    def __call__(self, x: mx.array, causal: Optional[bool] = None) -> mx.array:
        """Passe avant."""
        # Transposition vers channels last : (B, C, H, W) -> (B, H, W, C)
        x = mx.transpose(x, (0, 2, 3, 1))

        # Application du padding
        pad_h, pad_w = self.padding
        if pad_h != 0 or pad_w != 0:
            pad_width = [
                (0, 0),  # Batch
                (pad_h, pad_h),  # H
                (pad_w, pad_w),  # W
                (0, 0),  # C
            ]
            x = mx.pad(x, pad_width)

        x = self.conv(x)

        # Transposition de retour : (B, H, W, C) -> (B, C, H, W)
        x = mx.transpose(x, (0, 3, 1, 2))

        return x
