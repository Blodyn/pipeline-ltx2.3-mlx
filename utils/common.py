import math
from functools import partial
from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image


def get_model_path(model_repo: str):
    """Récupère ou télécharge le chemin du modèle LTX-2."""
    try:
        if Path(model_repo).exists():
            return Path(model_repo)
        return Path(snapshot_download(repo_id=model_repo, local_files_only=True))
    except Exception:
        print("Téléchargement des poids du modèle LTX-2…")
        return Path(
            snapshot_download(
                repo_id=model_repo,
                local_files_only=False,
                resume_download=True,
                allow_patterns=["*.safetensors", "*.json"],
            )
        )


def apply_quantization(model: nn.Module, weights: mx.array, quantization: dict):
    if quantization is not None:

        def get_class_predicate(p, m):
            # Gestion des quantifications personnalisées par couche
            if p in quantization:
                return quantization[p]
            if not hasattr(m, "to_quantized"):
                return False
            # Saute les couches dont la première dim n'est pas divisible par 64
            if hasattr(m, "weight") and m.weight.shape[0] % 64 != 0:
                return False
            # Compatibilité avec les anciens modèles dont tout n'est pas quantifié
            return f"{p}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=get_class_predicate,
        )


@partial(mx.compile, shapeless=True)
def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return mx.fast.rms_norm(x, mx.ones((x.shape[-1],), dtype=x.dtype), eps)


@partial(mx.compile, shapeless=True)
def to_denoised(
    noisy: mx.array, velocity: mx.array, sigma: mx.array | float
) -> mx.array:
    """Convertit une prédiction de vélocité en sortie débruitée.

    Étant donné l'entrée bruitée x_t et la prédiction de vélocité v, calcule x_0 :
    x_0 = x_t - sigma * v

    Calcule en float32 pour la précision (comportement identique à PyTorch),
    puis re-cast vers le dtype d'entrée.

    Args:
        noisy : tenseur d'entrée bruité x_t
        velocity : prédiction de vélocité v
        sigma : niveau de bruit (scalaire ou par échantillon)

    Returns:
        Tenseur débruité x_0
    """
    original_dtype = noisy.dtype

    # Cast en float32 pour la précision (PyTorch utilise calc_dtype=torch.float32)
    noisy_f32 = noisy.astype(mx.float32)
    velocity_f32 = velocity.astype(mx.float32)

    if isinstance(sigma, (int, float)):
        sigma_f32 = mx.array(sigma, dtype=mx.float32)
    else:
        sigma_f32 = sigma.astype(mx.float32)
        while sigma_f32.ndim < velocity_f32.ndim:
            sigma_f32 = mx.expand_dims(sigma_f32, axis=-1)

    result = noisy_f32 - sigma_f32 * velocity_f32
    return result.astype(original_dtype)


def repeat_interleave(x: mx.array, repeats: int, axis: int = -1) -> mx.array:
    """Répète les éléments d'un tenseur sur un axe, similaire à torch.repeat_interleave.

    Args:
        x : tenseur d'entrée
        repeats : nombre de répétitions pour chaque élément
        axis : axe sur lequel répéter les valeurs

    Returns:
        Tenseur avec les valeurs répétées
    """
    # Gestion d'un axe négatif
    if axis < 0:
        axis = x.ndim + axis

    # Récupération de la shape
    shape = list(x.shape)

    # Expansion, répétition, puis reshape
    x = mx.expand_dims(x, axis=axis + 1)

    # Construction du motif de tile
    tile_pattern = [1] * x.ndim
    tile_pattern[axis + 1] = repeats

    x = mx.tile(x, tile_pattern)

    # Reshape pour fusionner la dimension répétée
    new_shape = shape.copy()
    new_shape[axis] *= repeats

    return mx.reshape(x, new_shape)


class PixelNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return x / mx.sqrt(mx.mean(x * x, axis=1, keepdims=True) + self.eps)


def get_timestep_embedding(
    timesteps: mx.array,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> mx.array:
    """Crée des embeddings sinusoïdaux de timesteps.

    Args:
        timesteps : tenseur 1D de timesteps
        embedding_dim : dimension des embeddings à créer
        flip_sin_to_cos : si True, inverse l'ordre sin/cos
        downscale_freq_shift : facteur de décalage de fréquence
        scale : facteur d'échelle pour les timesteps
        max_period : période maximale des sinusoïdes

    Returns:
        Tenseur de forme (len(timesteps), embedding_dim)
    """
    assert timesteps.ndim == 1, "Les timesteps doivent être en 1D"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(0, half_dim, dtype=mx.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = mx.exp(exponent)
    emb = (timesteps[:, None].astype(mx.float32) * scale) * emb[None, :]

    # Calcul des embeddings sin et cos
    if flip_sin_to_cos:
        emb = mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)
    else:
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)

    # Padding zéros si la dimension d'embedding est impaire
    if embedding_dim % 2 == 1:
        emb = mx.pad(emb, [(0, 0), (0, 1)])

    return emb


def load_image(
    image_path: Union[str, Path],
    height: Optional[int] = None,
    width: Optional[int] = None,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Charge et prétraite une image pour le conditionnement I2V.

    Args:
        image_path : chemin du fichier image
        height : hauteur cible (doit être divisible par 32). Si None, conserve l'originale.
        width : largeur cible (doit être divisible par 32). Si None, conserve l'originale.

    Returns:
        Tenseur image de forme (H, W, 3) dans [0, 1]
    """
    image = Image.open(image_path).convert("RGB")

    # Redimensionnement si les dimensions sont précisées
    if height is not None and width is not None:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    elif height is not None or width is not None:
        # Si une seule dimension est précisée, on redimensionne en conservant l'aspect
        orig_w, orig_h = image.size
        if height is not None:
            scale = height / orig_h
            new_w = int(orig_w * scale)
            new_w = (new_w // 32) * 32  # Arrondi au multiple de 32 le plus proche
            image = image.resize((new_w, height), Image.Resampling.LANCZOS)
        else:
            scale = width / orig_w
            new_h = int(orig_h * scale)
            new_h = (new_h // 32) * 32  # Arrondi au multiple de 32 le plus proche
            image = image.resize((width, new_h), Image.Resampling.LANCZOS)
    else:
        # Arrondi au multiple de 32 le plus proche
        orig_w, orig_h = image.size
        new_w = (orig_w // 32) * 32
        new_h = (orig_h // 32) * 32
        if new_w != orig_w or new_h != orig_h:
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Conversion vers numpy puis MLX
    image_np = np.array(image).astype(np.float32) / 255.0
    return mx.array(image_np, dtype=dtype)


def resize_image_aspect_ratio(
    image: mx.array,
    long_side: int = 512,
) -> mx.array:
    """Redimensionne une image en conservant le ratio, en fixant le grand côté à long_side.

    Args:
        image : tenseur image de forme (H, W, 3)
        long_side : taille cible pour la plus grande dimension

    Returns:
        Tenseur image redimensionné
    """
    h, w = image.shape[:2]

    if h > w:
        new_h = long_side
        new_w = int(w * long_side / h)
    else:
        new_w = long_side
        new_h = int(h * long_side / w)

    # Arrondi au multiple de 32 le plus proche
    new_h = (new_h // 32) * 32
    new_w = (new_w // 32) * 32

    # Utilisation de PIL pour un redimensionnement de qualité
    image_np = np.array(image)
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    pil_image = Image.fromarray(image_np)
    pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return mx.array(np.array(pil_image).astype(np.float32) / 255.0)


def prepare_image_for_encoding(
    image: mx.array,
    target_height: int,
    target_width: int,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Prépare une image pour l'encodage VAE en la redimensionnant et la normalisant.

    Args:
        image : tenseur image de forme (H, W, 3) dans [0, 1]
        target_height : hauteur cible pour la vidéo
        target_width : largeur cible pour la vidéo

    Returns:
        Tenseur image prêt pour l'encodage, forme (1, 3, 1, H, W) dans [-1, 1]
    """
    h, w = image.shape[:2]

    # Redimensionnement si nécessaire
    if h != target_height or w != target_width:
        image_np = np.array(image)
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        pil_image = pil_image.resize(
            (target_width, target_height), Image.Resampling.LANCZOS
        )
        image = mx.array(np.array(pil_image).astype(np.float32) / 255.0)

    # Normalisation vers [-1, 1]
    image = image * 2.0 - 1.0

    # Conversion en (B, C, 1, H, W)
    image = mx.transpose(image, (2, 0, 1))  # (3, H, W)
    image = mx.expand_dims(image, axis=0)  # (1, 3, H, W)
    image = mx.expand_dims(image, axis=2)  # (1, 3, 1, H, W)

    return image.astype(dtype)
