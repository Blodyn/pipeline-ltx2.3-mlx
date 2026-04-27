"""Opérations d'échantillonnage pour le Video VAE (sur-échantillonnage / sous-échantillonnage)."""

from typing import Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .convolution import CausalConv3d, PaddingModeType


class SpaceToDepthDownsample(nn.Module):
    """Sous-échantillonnage space-to-depth avec conv 3x3 et connexion résiduelle.

    Implémentation compatible PyTorch :
    1. Conv 3x3 : in_channels -> out_channels // prod(stride)
    2. Space-to-depth sur la sortie de conv : channels * prod(stride)
    3. Space-to-depth sur l'entrée avec moyenne par groupe pour la connexion résiduelle
    4. Ajout de la connexion résiduelle
    """

    def __init__(
        self,
        dims: int,
        in_channels: int,
        out_channels: int,
        stride: Union[int, Tuple[int, int, int]],
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        super().__init__()

        if isinstance(stride, int):
            stride = (stride, stride, stride)

        self.stride = stride
        self.dims = dims
        self.out_channels = out_channels

        # Calcul des canaux
        multiplier = stride[0] * stride[1] * stride[2]
        self.group_size = in_channels * multiplier // out_channels
        conv_out_channels = out_channels // multiplier

        # Convolution 3x3 (et non 1x1)
        self.conv = CausalConv3d(
            in_channels=in_channels,
            out_channels=conv_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            spatial_padding_mode=spatial_padding_mode,
        )

    def _space_to_depth(self, x: mx.array) -> mx.array:
        """Réorganisation : b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w"""
        b, c, d, h, w = x.shape
        st, sh, sw = self.stride

        # Reshape pour regrouper les éléments spatiaux
        x = mx.reshape(x, (b, c, d // st, st, h // sh, sh, w // sw, sw))

        # Permutation : (B, C, D', st, H', sh, W', sw) -> (B, C, st, sh, sw, D', H', W')
        x = mx.transpose(x, (0, 1, 3, 5, 7, 2, 4, 6))

        # Reshape pour combiner les canaux
        new_c = c * st * sh * sw
        new_d = d // st
        new_h = h // sh
        new_w = w // sw
        x = mx.reshape(x, (b, new_c, new_d, new_h, new_w))

        return x

    def __call__(self, x: mx.array, causal: bool = True) -> mx.array:
        b, c, d, h, w = x.shape
        st, sh, sw = self.stride

        # Padding temporel pour le mode causal
        if st == 2:
            # On duplique la première frame pour le padding
            x = mx.concatenate([x[:, :, :1, :, :], x], axis=2)
            d = d + 1

        # Padding éventuel pour rendre les dimensions divisibles par le pas
        pad_d = (st - d % st) % st
        pad_h = (sh - h % sh) % sh
        pad_w = (sw - w % sw) % sw

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = mx.pad(x, [(0, 0), (0, 0), (0, pad_d), (0, pad_h), (0, pad_w)])

        # Connexion résiduelle : space-to-depth sur l'entrée puis moyenne par groupe
        x_in = self._space_to_depth(x)
        # Reshape pour la moyenne par groupe : (b, c*prod(stride), d, h, w) -> (b, out_channels, group_size, d, h, w)
        b2, c2, d2, h2, w2 = x_in.shape
        x_in = mx.reshape(x_in, (b2, self.out_channels, self.group_size, d2, h2, w2))
        x_in = mx.mean(x_in, axis=2)  # (b, out_channels, d, h, w)

        # Branche conv : conv puis space-to-depth
        x_conv = self.conv(x, causal=causal)
        x_conv = self._space_to_depth(x_conv)

        # Ajout de la connexion résiduelle
        return x_conv + x_in


class DepthToSpaceUpsample(nn.Module):

    def __init__(
        self,
        dims: int,
        in_channels: int,
        stride: Union[int, Tuple[int, int, int]],
        residual: bool = False,
        out_channels_reduction_factor: int = 1,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):

        super().__init__()

        if isinstance(stride, int):
            stride = (stride, stride, stride)

        self.stride = stride
        self.dims = dims
        self.residual = residual
        self.out_channels_reduction_factor = out_channels_reduction_factor

        # Calcul des canaux de sortie
        multiplier = stride[0] * stride[1] * stride[2]
        out_channels = in_channels // out_channels_reduction_factor
        self.out_channels = out_channels

        # Convolution 3x3x3 pour préparer les canaux au dépaquetage (correspond à PyTorch)
        self.conv = CausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels * multiplier,
            kernel_size=3,
            stride=1,
            padding=1,
            spatial_padding_mode=spatial_padding_mode,
        )

    def _depth_to_space(self, x: mx.array) -> mx.array:
        b, c_packed, d, h, w = x.shape
        st, sh, sw = self.stride
        c = c_packed // (st * sh * sw)

        # (B, C*st*sh*sw, D, H, W) -> (B, C, st, sh, sw, D, H, W)
        x = mx.reshape(x, (b, c, st, sh, sw, d, h, w))

        # (B, C, st, sh, sw, D, H, W) -> (B, C, D, st, H, sh, W, sw)
        x = mx.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))

        # (B, C, D, st, H, sh, W, sw) -> (B, C, D*st, H*sh, W*sw)
        x = mx.reshape(x, (b, c, d * st, h * sh, w * sw))

        return x

    def __call__(
        self, x: mx.array, causal: bool = True, chunked_conv: bool = False
    ) -> mx.array:

        b, c, d, h, w = x.shape
        st, sh, sw = self.stride

        # Calcul du chemin résiduel si activé
        x_residual = None
        if self.residual:
            # Reshape de l'entrée : on traite les canaux comme des facteurs spatiaux
            # "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)"
            x_residual = self._depth_to_space(x)

            # On tile les canaux pour correspondre à la sortie
            # (.repeat() de PyTorch tile, ce n'est pas une répétition élément par élément !)
            # num_repeat = prod(stride) / out_channels_reduction_factor
            num_repeat = (st * sh * sw) // self.out_channels_reduction_factor
            x_residual = mx.tile(x_residual, (1, num_repeat, 1, 1, 1))

            # On retire la première frame temporelle en cas de sur-échantillonnage temporel
            if st > 1:
                x_residual = x_residual[:, :, 1:, :, :]

        # Mode par tronçons pour les gros tenseurs, afin de réduire le pic mémoire
        if chunked_conv and d > 4:
            x = self._chunked_conv_depth_to_space(x, causal)
        else:
            # Application de la conv
            x = self.conv(x, causal=causal)
            # Réorganisation depth-to-space
            x = self._depth_to_space(x)

        # Suppression de la première frame pour le sur-échantillonnage temporel causal
        if st > 1:
            x = x[:, :, 1:, :, :]

        # Ajout du résidu
        if self.residual and x_residual is not None:
            x = x + x_residual

        return x

    def _chunked_conv_depth_to_space(
        self, x: mx.array, causal: bool = True
    ) -> mx.array:
        """Conv + depth_to_space par tronçons temporels.

        Réduit le pic mémoire en évitant l'allocation du gros tenseur intermédiaire
        à grand nombre de canaux. Au lieu de matérialiser (B, 4096, D, H, W), on
        traite des tronçons temporels et on applique depth_to_space immédiatement.

        Args:
            x : tenseur d'entrée de forme (B, C, D, H, W)
            causal : utiliser des convolutions causales

        Returns:
            Tenseur de sortie après conv + depth_to_space
        """
        b, c, d, h, w = x.shape
        st, sh, sw = self.stride
        out_c = self.out_channels

        # Dimensions de sortie
        out_d = d * st
        out_h = h * sh
        out_w = w * sw

        # Taille de tronçon dans la dimension temporelle (4 frames à la fois)
        chunk_size = 4
        kernel_t = 3  # Taille du noyau temporel

        # En conv causale, il faut (kernel_t - 1) frames de padding au début.
        # En non causal, il faut (kernel_t - 1) // 2 de chaque côté.
        if causal:
            # Padding au début avec la première frame répétée
            pad_start = kernel_t - 1
            pad_end = 0
        else:
            pad_start = (kernel_t - 1) // 2
            pad_end = (kernel_t - 1) // 2

        # Allocation de la sortie
        outputs = []

        # Traitement par tronçons avec recouvrement pour le noyau de conv
        t_pos = 0
        while t_pos < d:
            t_end = min(t_pos + chunk_size, d)

            # Calcul de la plage d'entrée avec padding pour le noyau
            in_start = max(0, t_pos - pad_start)
            in_end = min(d, t_end + pad_end)

            # Extraction du tronçon
            chunk = x[:, :, in_start:in_end, :, :]

            # Application de la conv au tronçon
            chunk_conv = self.conv(chunk, causal=causal)

            # Application du depth_to_space
            chunk_out = self._depth_to_space(chunk_conv)

            # Calcul de la plage de sortie valide (en excluant les effets du padding)
            # Chaque frame d'entrée produit st frames de sortie
            out_start = (t_pos - in_start) * st
            out_end = out_start + (t_end - t_pos) * st

            # Extraction de la portion valide
            chunk_out = chunk_out[:, :, out_start:out_end, :, :]

            outputs.append(chunk_out)

            # Évaluation pour libérer la mémoire intermédiaire
            mx.eval(outputs[-1])

            t_pos = t_end

        # Concaténation de tous les tronçons
        if len(outputs) == 1:
            return outputs[0]
        return mx.concatenate(outputs, axis=2)
