import math
from typing import List, Optional, Tuple

import mlx.core as mx

from .config import LTXRopeType


def apply_rotary_emb(
    input_tensor: mx.array,
    freqs_cis: Tuple[mx.array, mx.array],
    rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
) -> mx.array:
    """Applique des embeddings de position rotatifs à un tenseur d'entrée.

    Args:
        input_tensor : tenseur d'entrée auquel appliquer le RoPE
        freqs_cis : tuple (cos_freqs, sin_freqs)
        rope_type : type de RoPE à appliquer (INTERLEAVED ou SPLIT)

    Returns:
        Tenseur avec les embeddings rotatifs appliqués
    """
    if rope_type == LTXRopeType.INTERLEAVED:
        return apply_interleaved_rotary_emb(input_tensor, freqs_cis[0], freqs_cis[1])
    elif rope_type == LTXRopeType.SPLIT:
        return apply_split_rotary_emb(input_tensor, freqs_cis[0], freqs_cis[1])
    else:
        raise ValueError(f"Type de RoPE invalide : {rope_type}")


def apply_interleaved_rotary_emb(
    input_tensor: mx.array,
    cos_freqs: mx.array,
    sin_freqs: mx.array,
) -> mx.array:
    """Applique des embeddings rotatifs entrelacés.

    Apparie des dimensions adjacentes et applique une rotation.
    Motif : [x0, x1, x2, x3, ...] -> rotation des paires (x0,x1), (x2,x3), ...

    Args:
        input_tensor : tenseur d'entrée de forme (..., dim)
        cos_freqs : fréquences cosinus
        sin_freqs : fréquences sinus

    Returns:
        Tenseur avec les embeddings rotatifs entrelacés appliqués
    """
    # Calcul en float32 pour une meilleure précision
    input_dtype = input_tensor.dtype
    input_tensor = input_tensor.astype(mx.float32)
    cos_freqs = cos_freqs.astype(mx.float32)
    sin_freqs = sin_freqs.astype(mx.float32)

    # Reshape pour apparier les dimensions adjacentes : (..., dim) -> (..., dim/2, 2)
    shape = input_tensor.shape
    input_tensor = mx.reshape(input_tensor, shape[:-1] + (shape[-1] // 2, 2))

    # Extraction des paires
    t1 = input_tensor[..., 0]  # Indices pairs
    t2 = input_tensor[..., 1]  # Indices impairs

    # Application de la rotation : motif (-t2, t1)
    t_rot = mx.stack([-t2, t1], axis=-1)

    # Reflatten : (..., dim/2, 2) -> (..., dim)
    input_tensor = mx.reshape(input_tensor, shape)
    t_rot = mx.reshape(t_rot, shape)

    # Application des embeddings rotatifs
    out = input_tensor * cos_freqs + t_rot * sin_freqs

    return out.astype(input_dtype)


def rotate_half_interleaved(x: mx.array) -> mx.array:
    """Rotation pour le RoPE entrelacé : [x0, x1, x2, x3] -> [-x1, x0, -x3, x2].

    Équivalent PyTorch :
        t_dup = rearrange(x, "... (d r) -> ... d r", r=2)
        t1, t2 = t_dup.unbind(dim=-1)
        t_dup = torch.stack((-t2, t1), dim=-1)
        return rearrange(t_dup, "... d r -> ... (d r)")
    """
    # x : (..., dim) avec dim pair
    x_even = x[..., 0::2]  # [x0, x2, x4, ...]
    x_odd = x[..., 1::2]  # [x1, x3, x5, ...]
    # Empilement : [[-x1, x0], [-x3, x2], ...] puis aplatissement en [-x1, x0, -x3, x2, ...]
    rotated = mx.stack([-x_odd, x_even], axis=-1)
    return mx.reshape(rotated, x.shape)


def apply_rotary_emb_1d(
    q: mx.array,
    k: mx.array,
    freqs_cis: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Applique des embeddings rotatifs 1D à partir de fréquences précalculées (entrelacé)."""
    # freqs_cis : (1, seq_len, num_heads, head_dim, 2) où [..., 0] = cos, [..., 1] = sin
    cos = freqs_cis[..., 0]  # (1, seq_len, num_heads, head_dim)
    sin = freqs_cis[..., 1]

    # q, k : (batch, seq_len, num_heads, head_dim)
    # RoPE entrelacé : les paires de dimensions adjacentes tournent ensemble
    q_r = q * cos + rotate_half_interleaved(q) * sin
    k_r = k * cos + rotate_half_interleaved(k) * sin

    return q_r, k_r


def apply_split_rotary_emb(
    input_tensor: mx.array,
    cos_freqs: mx.array,
    sin_freqs: mx.array,
) -> mx.array:
    """Applique des embeddings rotatifs split.

    Sépare les dimensions en deux moitiés et applique la rotation.
    Motif : split en première moitié et seconde moitié.

    Args:
        input_tensor : tenseur d'entrée
        cos_freqs : fréquences cosinus de forme (B, H, T, D//2)
        sin_freqs : fréquences sinus de forme (B, H, T, D//2)

    Returns:
        Tenseur avec les embeddings rotatifs split appliqués
    """
    input_dtype = input_tensor.dtype
    needs_reshape = False
    original_shape = input_tensor.shape

    # Gestion d'une éventuelle incohérence de dimension
    if input_tensor.ndim != 4 and cos_freqs.ndim == 4:
        b, h, t, _ = cos_freqs.shape
        # Reshape de (B, T, H*D) vers (B, H, T, D)
        input_tensor = mx.reshape(input_tensor, (b, t, h, -1))
        input_tensor = mx.swapaxes(input_tensor, 1, 2)
        needs_reshape = True

    # Cast en float32 pour la précision du calcul
    input_tensor = input_tensor.astype(mx.float32)
    cos_freqs = cos_freqs.astype(mx.float32)
    sin_freqs = sin_freqs.astype(mx.float32)

    # Split en deux moitiés : (..., dim) -> (..., 2, dim//2)
    dim = input_tensor.shape[-1]
    split_input = mx.reshape(input_tensor, input_tensor.shape[:-1] + (2, dim // 2))

    # Récupération des deux moitiés
    first_half = split_input[..., 0, :]  # (..., dim//2)
    second_half = split_input[..., 1, :]  # (..., dim//2)

    # Application du cosinus aux deux moitiés
    output_first = first_half * cos_freqs
    output_second = second_half * cos_freqs

    # Application des termes croisés en sinus (motif addcmul)
    output_first = output_first - sin_freqs * second_half
    output_second = output_second + sin_freqs * first_half

    # Empilement final
    output = mx.stack([output_first, output_second], axis=-2)

    # Aplatissement : (..., 2, dim//2) -> (..., dim)
    output = mx.reshape(output, input_tensor.shape)

    if needs_reshape:
        # Reshape de retour : (B, H, T, D) -> (B, T, H*D)
        b, h, t, d = output.shape
        output = mx.swapaxes(output, 1, 2)
        output = mx.reshape(output, (b, t, h * d))

    return output.astype(input_dtype)


def generate_freq_grid(
    positional_embedding_theta: float,
    positional_embedding_max_pos_count: int,
    inner_dim: int,
) -> mx.array:
    """Génère la grille de fréquences pour le RoPE.

    Args:
        positional_embedding_theta : valeur de base theta
        positional_embedding_max_pos_count : nombre de dimensions de position
        inner_dim : dimension interne du modèle

    Returns:
        Tenseur d'indices de fréquences
    """
    theta = positional_embedding_theta
    start = 1.0
    end = theta

    n_elem = 2 * positional_embedding_max_pos_count

    # Espacement logarithmique
    log_start = math.log(start) / math.log(theta)
    log_end = math.log(end) / math.log(theta)

    num_indices = inner_dim // n_elem
    if num_indices == 0:
        num_indices = 1

    # Création de valeurs espacées linéairement en espace log
    lin_space = mx.linspace(log_start, log_end, num_indices)

    # Calcul des indices puissance
    pow_indices = mx.power(theta, lin_space)

    # Mise à l'échelle par pi/2
    return pow_indices * (math.pi / 2)


def get_fractional_positions(
    indices_grid: mx.array,
    max_pos: List[int],
) -> mx.array:
    """Convertit des indices en positions fractionnaires.

    Args:
        indices_grid : grille d'indices de position de forme (B, n_pos_dims, ...)
        max_pos : position maximale pour chaque dimension

    Returns:
        Positions fractionnaires dans [-1, 1] après mise à l'échelle
    """
    n_pos_dims = indices_grid.shape[1]
    assert n_pos_dims == len(
        max_pos
    ), f"Le nombre de dimensions de position ({n_pos_dims}) doit correspondre à la longueur de max_pos ({len(max_pos)})"

    # Division de chaque dimension par sa position max
    fractional_positions = []
    for i in range(n_pos_dims):
        frac = indices_grid[:, i] / max_pos[i]
        fractional_positions.append(frac)

    return mx.stack(fractional_positions, axis=-1)


def generate_freqs(
    indices: mx.array,
    indices_grid: mx.array,
    max_pos: List[int],
    use_middle_indices_grid: bool,
) -> mx.array:
    """Génère les fréquences à partir des indices et de la grille de positions.

    Args:
        indices : indices de fréquences
        indices_grid : grille d'indices de position
        max_pos : positions maximales par dimension
        use_middle_indices_grid : utiliser le milieu des plages d'indices

    Returns:
        Tenseur de fréquences
    """
    # Gestion de la grille des indices médians
    if use_middle_indices_grid:
        # Forme de indices_grid : (B, n_dims, T, 2) où la dernière dim est [start, end]
        assert len(indices_grid.shape) == 4
        assert indices_grid.shape[-1] == 2
        indices_grid_start = indices_grid[..., 0]
        indices_grid_end = indices_grid[..., 1]
        indices_grid = (indices_grid_start + indices_grid_end) / 2.0
    elif len(indices_grid.shape) == 4:
        indices_grid = indices_grid[..., 0]

    # Récupération des positions fractionnaires
    fractional_positions = get_fractional_positions(indices_grid, max_pos)

    # Calcul des fréquences
    # fractional_positions : (B, T, n_dims)
    # indices : (inner_dim // n_elem,)
    # Résultat : (B, T, inner_dim // n_elem * n_dims)

    # Mise à l'échelle des positions fractionnaires vers [-1, 1]
    scaled_positions = fractional_positions * 2 - 1  # (B, T, n_dims)

    # Produit extérieur avec les indices
    # (B, T, n_dims, 1) * (1, 1, 1, n_indices) -> (B, T, n_dims, n_indices)
    freqs = mx.expand_dims(scaled_positions, axis=-1) * mx.expand_dims(
        mx.expand_dims(mx.expand_dims(indices, axis=0), axis=0), axis=0
    )

    # Transposition + flatten : (B, T, n_dims, n_indices) -> (B, T, n_indices * n_dims)
    freqs = mx.swapaxes(freqs, -1, -2)  # (B, T, n_indices, n_dims)
    freqs = mx.reshape(freqs, freqs.shape[:-2] + (-1,))

    return freqs


def split_freqs_cis(
    freqs: mx.array,
    pad_size: int,
    num_attention_heads: int,
) -> Tuple[mx.array, mx.array]:
    """Prépare les fréquences cos/sin pour le RoPE split.

    Args:
        freqs : tenseur de fréquences
        pad_size : taille de padding pour aligner la dimension
        num_attention_heads : nombre de têtes d'attention

    Returns:
        Tuple (cos_freq, sin_freq) de forme (B, H, T, D//2)
    """
    cos_freq = mx.cos(freqs)
    sin_freq = mx.sin(freqs)

    # Ajout du padding si nécessaire
    if pad_size != 0:
        cos_padding = mx.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = mx.zeros_like(sin_freq[:, :, :pad_size])

        cos_freq = mx.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = mx.concatenate([sin_padding, sin_freq], axis=-1)

    # Reshape pour l'attention multi-têtes
    b, t = cos_freq.shape[0], cos_freq.shape[1]

    cos_freq = mx.reshape(cos_freq, (b, t, num_attention_heads, -1))
    sin_freq = mx.reshape(sin_freq, (b, t, num_attention_heads, -1))

    # Inversion d'axes : (B, T, H, D//2) -> (B, H, T, D//2)
    cos_freq = mx.swapaxes(cos_freq, 1, 2)
    sin_freq = mx.swapaxes(sin_freq, 1, 2)

    return cos_freq, sin_freq


def interleaved_freqs_cis(
    freqs: mx.array,
    pad_size: int,
) -> Tuple[mx.array, mx.array]:
    """Prépare les fréquences cos/sin pour le RoPE entrelacé.

    Args:
        freqs : tenseur de fréquences de forme (B, T, dim//2)
        pad_size : taille de padding pour aligner la dimension

    Returns:
        Tuple (cos_freq, sin_freq) de forme (B, T, dim)
    """
    # Calcul de cos et sin
    cos_freq = mx.cos(freqs)
    sin_freq = mx.sin(freqs)

    # Repeat-interleave : chaque élément répété deux fois
    # (B, T, D) -> (B, T, 2*D) avec le motif [c0, c0, c1, c1, ...]
    cos_freq = mx.repeat(cos_freq, 2, axis=-1)
    sin_freq = mx.repeat(sin_freq, 2, axis=-1)

    # Ajout du padding si nécessaire
    if pad_size != 0:
        cos_padding = mx.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = mx.zeros_like(sin_freq[:, :, :pad_size])
        cos_freq = mx.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = mx.concatenate([sin_padding, sin_freq], axis=-1)

    return cos_freq, sin_freq


def precompute_freqs_cis(
    indices_grid: mx.array,
    dim: int,
    theta: float = 10000.0,
    max_pos: Optional[List[int]] = None,
    use_middle_indices_grid: bool = False,
    num_attention_heads: int = 32,
    rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
    double_precision: bool = False,
) -> Tuple[mx.array, mx.array]:
    """Précalcule les fréquences RoPE.

    Args:
        indices_grid : grille d'indices de position
        dim : dimension du RoPE
        theta : valeur de base theta pour le calcul des fréquences
        max_pos : position maximale par dimension
        use_middle_indices_grid : utiliser les indices médians
        num_attention_heads : nombre de têtes d'attention
        rope_type : type de RoPE (INTERLEAVED ou SPLIT)
        double_precision : si True, calcule les fréquences en float64 pour plus de précision

    Returns:
        Tuple de tenseurs (cos_freq, sin_freq)
    """
    if max_pos is None:
        max_pos = [20, 2048, 2048]

    if double_precision:
        return _precompute_freqs_cis_double_precision(
            indices_grid,
            dim,
            theta,
            max_pos,
            use_middle_indices_grid,
            num_attention_heads,
            rope_type,
        )

    # On garde les positions en float32 pour le calcul du RoPE.
    # Bien que PyTorch convertisse nominalement les positions vers le dtype du modèle (bfloat16),
    # la comparaison empirique montre que des positions en float32 produisent des valeurs RoPE
    # qui matchent PyTorch exactement (cosine=1.000). Le bfloat16 perd de la précision dans le
    # calcul des positions fractionnaires, qui est amplifiée par les indices haute fréquence
    # (jusqu'à 15708), provoquant des inversions de signe en cos/sin et une similarité cosinus
    # qui chute à 0.88 seulement.
    indices_grid = indices_grid.astype(mx.float32)

    # Génération des indices de fréquences
    indices = generate_freq_grid(theta, indices_grid.shape[1], dim)

    # Génération des fréquences
    freqs = generate_freqs(indices, indices_grid, max_pos, use_middle_indices_grid)

    # Préparation cos/sin selon le type de RoPE
    if rope_type == LTXRopeType.SPLIT:
        expected_freqs = dim // 2
        current_freqs = freqs.shape[-1]
        pad_size = expected_freqs - current_freqs
        cos_freq, sin_freq = split_freqs_cis(freqs, pad_size, num_attention_heads)
    else:
        # Entrelacé
        n_elem = 2 * indices_grid.shape[1]
        cos_freq, sin_freq = interleaved_freqs_cis(freqs, dim % n_elem)

    return cos_freq, sin_freq


def _precompute_freqs_cis_double_precision(
    indices_grid: mx.array,
    dim: int,
    theta: float,
    max_pos: List[int],
    use_middle_indices_grid: bool,
    num_attention_heads: int,
    rope_type: LTXRopeType,
) -> Tuple[mx.array, mx.array]:
    """Calcule les fréquences RoPE en plus haute précision (float64) pour la grille de fréquences.

    Reproduit le `generate_freq_grid_np` de PyTorch : utilise NumPy float64 pour le calcul
    critique de la grille de fréquences (valeurs espacées en log), puis convertit en float32.
    La grille de positions reste en bfloat16 pour reproduire le comportement PyTorch (les
    positions sont dans le dtype du modèle tout au long de generate_freqs).
    """
    import numpy as np

    # On garde les positions en float32 — même raisonnement que la voie sans double précision.
    indices_grid_f32 = indices_grid.astype(mx.float32)

    n_pos_dims = indices_grid_f32.shape[1]
    n_elem = 2 * n_pos_dims

    # Calcul des fréquences espacées en log en float64 (pour matcher generate_freq_grid_np de PyTorch)
    # C'est l'étape de précision critique — PyTorch utilise np.float64 ici
    log_start = np.log(1.0) / np.log(theta)
    log_end = np.log(theta) / np.log(theta)  # = 1.0
    num_indices = dim // n_elem
    if num_indices == 0:
        num_indices = 1

    # Utilisation de numpy float64 pour le linspace (correspond à PyTorch)
    pow_indices = np.power(
        theta,
        np.linspace(log_start, log_end, num_indices, dtype=np.float64),
    )
    # Conversion en tenseur float32 (correspond à torch.tensor(..., dtype=torch.float32))
    freq_indices = mx.array(pow_indices * (math.pi / 2), dtype=mx.float32)

    # Gestion de la grille des indices médians
    # Forme d'entrée : (B, n_dims, T, 2) pour les indices médians, (B, n_dims, T, 1) sinon
    if use_middle_indices_grid:
        assert len(indices_grid_f32.shape) == 4
        assert indices_grid_f32.shape[-1] == 2
        indices_grid_start = indices_grid_f32[..., 0]
        indices_grid_end = indices_grid_f32[..., 1]
        indices_grid_f32 = (indices_grid_start + indices_grid_end) / 2.0
    elif len(indices_grid_f32.shape) == 4:
        indices_grid_f32 = indices_grid_f32[..., 0]
    # Après gestion : la forme de indices_grid_f32 est (B, n_dims, T)

    # Positions fractionnaires : (B, n_dims, T) -> (B, T, n_dims)
    # Calcul des positions fractionnaires pour chaque dimension
    fractional_list = []
    for i in range(n_pos_dims):
        frac = indices_grid_f32[:, i, :] / max_pos[i]  # (B, T)
        fractional_list.append(frac)

    # Empilement : (B, T, n_dims)
    fractional_positions = mx.stack(fractional_list, axis=-1)

    # Mise à l'échelle vers [-1, 1]
    scaled_positions = fractional_positions * 2 - 1

    # Calcul des fréquences : produit extérieur
    # scaled_positions : (B, T, n_dims) -> (B, T, n_dims, 1)
    # freq_indices : (num_indices,) -> (1, 1, 1, num_indices)
    freqs = mx.expand_dims(scaled_positions, axis=-1) * mx.reshape(
        freq_indices, (1, 1, 1, -1)
    )
    # freqs : (B, T, n_dims, num_indices)

    # Transposition + flatten : (B, T, n_dims, num_indices) -> (B, T, num_indices, n_dims) -> (B, T, num_indices * n_dims)
    freqs = mx.swapaxes(freqs, -1, -2)
    freqs = mx.reshape(freqs, (freqs.shape[0], freqs.shape[1], -1))

    # Calcul de cos/sin
    cos_freq = mx.cos(freqs)
    sin_freq = mx.sin(freqs)

    # Préparation selon le type de RoPE
    if rope_type == LTXRopeType.SPLIT:
        expected_freqs = dim // 2
        current_freqs = cos_freq.shape[-1]
        pad_size = expected_freqs - current_freqs

        # Ajout du padding
        if pad_size > 0:
            cos_padding = mx.ones((*cos_freq.shape[:-1], pad_size), dtype=mx.float32)
            sin_padding = mx.zeros((*sin_freq.shape[:-1], pad_size), dtype=mx.float32)
            cos_freq = mx.concatenate([cos_padding, cos_freq], axis=-1)
            sin_freq = mx.concatenate([sin_padding, sin_freq], axis=-1)

        # Reshape pour l'attention multi-têtes : (B, T, dim//2) -> (B, H, T, dim//2//H)
        b, t = cos_freq.shape[0], cos_freq.shape[1]
        cos_freq = mx.reshape(cos_freq, (b, t, num_attention_heads, -1))
        sin_freq = mx.reshape(sin_freq, (b, t, num_attention_heads, -1))
        cos_freq = mx.swapaxes(cos_freq, 1, 2)
        sin_freq = mx.swapaxes(sin_freq, 1, 2)
    else:
        # Entrelacé
        cos_freq = mx.repeat(cos_freq, 2, axis=-1)
        sin_freq = mx.repeat(sin_freq, 2, axis=-1)

        pad_size = dim % n_elem
        if pad_size > 0:
            cos_padding = mx.ones((*cos_freq.shape[:-1], pad_size), dtype=mx.float32)
            sin_padding = mx.zeros((*sin_freq.shape[:-1], pad_size), dtype=mx.float32)
            cos_freq = mx.concatenate([cos_padding, cos_freq], axis=-1)
            sin_freq = mx.concatenate([sin_padding, sin_freq], axis=-1)

    return cos_freq, sin_freq
