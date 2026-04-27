"""Conditionnement par latents pour la génération I2V (Image-to-Video).

Ce module fournit un mécanisme de conditionnement qui injecte des latents d'image
encodés dans le processus de génération vidéo à des positions de frame spécifiques.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx


@dataclass
class VideoConditionByLatentIndex:
    """Conditionne la génération vidéo en injectant des latents à un index de frame précis.

    Remplace le latent à l'index de frame indiqué par le latent conditionnant et
    contrôle la quantité de débruitage via le paramètre strength.

    Args:
        latent : latent d'image encodé de forme (B, C, 1, H, W)
        frame_idx : index de la frame à conditionner (0 = première frame)
        strength : intensité du débruitage (1.0 = débruitage complet, 0.0 = conserve l'original)
    """

    latent: mx.array
    frame_idx: int = 0
    strength: float = 1.0

    def get_num_latent_frames(self) -> int:
        """Renvoie le nombre de frames latentes du conditionnement."""
        return self.latent.shape[2]


@dataclass
class LatentState:
    """État de la diffusion latente avec prise en charge du conditionnement.

    Attributes:
        latent : latent bruité courant (B, C, F, H, W)
        clean_latent : latent de conditionnement « propre » (B, C, F, H, W)
        denoise_mask : masque de débruitage par frame (B, 1, F, 1, 1) où
                      1.0 = débruitage complet, 0.0 = conserver le latent propre
    """

    latent: mx.array
    clean_latent: mx.array
    denoise_mask: mx.array

    def clone(self) -> "LatentState":
        """Crée une copie de l'état."""
        return LatentState(
            latent=self.latent,
            clean_latent=self.clean_latent,
            denoise_mask=self.denoise_mask,
        )


def create_initial_state(
    shape: Tuple[int, ...],
    seed: Optional[int] = None,
    noise_scale: float = 1.0,
) -> LatentState:
    """Crée un état latent bruité initial.

    Args:
        shape : forme du latent (B, C, F, H, W)
        seed : graine aléatoire optionnelle
        noise_scale : échelle du bruit initial (sigma)

    Returns:
        LatentState initial avec du bruit aléatoire
    """
    if seed is not None:
        mx.random.seed(seed)

    noise = mx.random.normal(shape)

    return LatentState(
        latent=noise * noise_scale,
        clean_latent=mx.zeros(shape),
        denoise_mask=mx.ones((shape[0], 1, shape[2], 1, 1)),  # Débruitage complet par défaut
    )


def apply_conditioning(
    state: LatentState,
    conditionings: List[VideoConditionByLatentIndex],
) -> LatentState:
    """Applique des éléments de conditionnement à un état latent.

    Args:
        state : état latent courant
        conditionings : liste des éléments de conditionnement à appliquer

    Returns:
        LatentState mis à jour, conditionnement appliqué
    """
    state = state.clone()
    dtype = state.latent.dtype
    b, c, f, h, w = state.latent.shape

    for cond in conditionings:
        cond_latent = cond.latent
        frame_idx = cond.frame_idx
        strength = cond.strength

        # Validation des formes
        _, cond_c, cond_f, cond_h, cond_w = cond_latent.shape
        if (cond_c, cond_h, cond_w) != (c, h, w):
            raise ValueError(
                f"La forme spatiale du latent de conditionnement ({cond_c}, {cond_h}, {cond_w}) "
                f"ne correspond pas à la forme cible ({c}, {h}, {w})"
            )

        if frame_idx >= f:
            raise ValueError(
                f"L'index de frame {frame_idx} dépasse les bornes pour un latent à {f} frames"
            )

        # Nombre de frames de conditionnement
        num_cond_frames = cond_f
        end_idx = min(frame_idx + num_cond_frames, f)

        # Remplacement du latent à la position de conditionnement
        # state.latent[:, :, frame_idx:end_idx] = cond_latent[:, :, :end_idx - frame_idx]
        latent_list = []
        clean_list = []
        mask_list = []

        for i in range(f):
            if frame_idx <= i < end_idx:
                # On utilise le latent de conditionnement
                cond_idx = i - frame_idx
                latent_list.append(cond_latent[:, :, cond_idx : cond_idx + 1])
                clean_list.append(cond_latent[:, :, cond_idx : cond_idx + 1])
                # Réglage du masque : 1.0 - strength signifie moins de débruitage pour les frames conditionnées
                mask_list.append(mx.full((b, 1, 1, 1, 1), 1.0 - strength, dtype=dtype))
            else:
                # On conserve l'original
                latent_list.append(state.latent[:, :, i : i + 1])
                clean_list.append(state.clean_latent[:, :, i : i + 1])
                mask_list.append(state.denoise_mask[:, :, i : i + 1])

        state.latent = mx.concatenate(latent_list, axis=2)
        state.clean_latent = mx.concatenate(clean_list, axis=2)
        state.denoise_mask = mx.concatenate(mask_list, axis=2)

    return state


def apply_denoise_mask(
    denoised: mx.array,
    clean: mx.array,
    denoise_mask: mx.array,
) -> mx.array:
    """Mélange la sortie débruitée et l'état propre selon le masque.

    Args:
        denoised : latent débruité (B, C, F, H, W)
        clean : latent de conditionnement propre (B, C, F, H, W)
        denoise_mask : masque où 1.0 = utiliser le débruité, 0.0 = utiliser le propre

    Returns:
        Latent mélangé
    """
    one = mx.array(1.0, dtype=denoised.dtype)
    return denoised * denoise_mask + clean * (one - denoise_mask)


def add_noise_with_state(
    state: LatentState,
    noise_scale: float,
) -> LatentState:
    """Ajoute du bruit à l'état tout en respectant le conditionnement.

    Pour les frames conditionnées (masque < 1.0), on ajoute du bruit de manière
    proportionnelle pour permettre un peu d'affinage tout en préservant le conditionnement.

    Args:
        state : état latent courant
        noise_scale : échelle du bruit (sigma)

    Returns:
        État mis à jour avec du bruit ajouté
    """
    state = state.clone()

    # Génération du bruit
    noise = mx.random.normal(state.latent.shape)

    # Pour les frames totalement conditionnées (masque=0), on veut ajouter un bruit minimal.
    # Pour les frames non conditionnées (masque=1), on veut un bruit complet.
    # noisy = noise * sigma + latent * (1 - sigma)
    # Mais on module sigma par le masque pour les régions conditionnées.

    effective_scale = noise_scale * state.denoise_mask
    one = mx.array(1.0, dtype=state.latent.dtype)
    state.latent = noise * effective_scale + state.latent * (one - effective_scale)

    return state
