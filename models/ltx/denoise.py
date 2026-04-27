"""Boucle de débruitage distillée pour LTX-2.3.

Ce module isole la boucle de débruitage par pas d'Euler utilisée par le pipeline
distillé. La CFG, l'APG, le STG, le scale par modalité et l'échantillonneur d'ordre
deux ``res_2s`` (utilisés par les pipelines dev / dev-two-stage / dev-two-stage-hq)
ont tous été retirés : le checkpoint distillé est entraîné pour fonctionner sans
classifier-free guidance.
"""

from typing import Callable, Optional

import mlx.core as mx
import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from .conditioning.latent import (
    LatentState,
    apply_denoise_mask,
)
from .ltx import LTXModel
from .transformer import Modality

console = Console()


# Plannings sigma pour le modèle distillé. Les deux étapes effectuent quelques pas
# d'Euler avec des valeurs de sigma fixes — aucun réglage de scheduler nécessaire.
STAGE_1_SIGMAS = [
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
    0.0,
]
STAGE_2_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]


# Constantes audio utilisées partout dans le pipeline.
AUDIO_SAMPLE_RATE = 24000  # Fréquence d'échantillonnage de l'audio en sortie
AUDIO_LATENT_SAMPLE_RATE = 16000  # Fréquence d'échantillonnage interne du VAE
AUDIO_HOP_LENGTH = 160
AUDIO_LATENT_DOWNSAMPLE_FACTOR = 4
AUDIO_LATENT_CHANNELS = 8  # Canaux latents avant patchification
AUDIO_MEL_BINS = 16
AUDIO_LATENTS_PER_SECOND = (
    AUDIO_LATENT_SAMPLE_RATE / AUDIO_HOP_LENGTH / AUDIO_LATENT_DOWNSAMPLE_FACTOR
)  # 25


def create_position_grid(
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    temporal_scale: int = 8,
    spatial_scale: int = 32,
    fps: float = 24.0,
    causal_fix: bool = True,
) -> mx.array:
    """Crée la grille de positions pour le RoPE en espace pixel.

    Args:
        batch_size : taille du batch.
        num_frames : nombre de frames latentes.
        height : hauteur latente.
        width : largeur latente.
        temporal_scale : facteur d'échelle temporel du VAE (8 par défaut).
        spatial_scale : facteur d'échelle spatial du VAE (32 par défaut).
        fps : frames par seconde (24.0 par défaut).
        causal_fix : applique la correction causale pour la première frame (True par défaut).

    Returns:
        Grille de positions de forme ``(B, 3, num_patches, 2)`` en espace pixel,
        où la dim 2 contient la borne ``[start, end)`` pour chaque patch.
    """
    patch_size_t, patch_size_h, patch_size_w = 1, 1, 1

    t_coords = np.arange(0, num_frames, patch_size_t)
    h_coords = np.arange(0, height, patch_size_h)
    w_coords = np.arange(0, width, patch_size_w)

    t_grid, h_grid, w_grid = np.meshgrid(t_coords, h_coords, w_coords, indexing="ij")
    patch_starts = np.stack([t_grid, h_grid, w_grid], axis=0)

    patch_size_delta = np.array([patch_size_t, patch_size_h, patch_size_w]).reshape(
        3, 1, 1, 1
    )
    patch_ends = patch_starts + patch_size_delta

    latent_coords = np.stack([patch_starts, patch_ends], axis=-1)
    num_patches = num_frames * height * width
    latent_coords = latent_coords.reshape(3, num_patches, 2)
    latent_coords = np.tile(latent_coords[np.newaxis, ...], (batch_size, 1, 1, 1))

    scale_factors = np.array([temporal_scale, spatial_scale, spatial_scale]).reshape(
        1, 3, 1, 1
    )
    pixel_coords = (latent_coords * scale_factors).astype(np.float32)

    if causal_fix:
        pixel_coords[:, 0, :, :] = np.clip(
            pixel_coords[:, 0, :, :] + 1 - temporal_scale, a_min=0, a_max=None
        )

    # Division des coordonnées temporelles par fps
    pixel_coords[:, 0, :, :] = pixel_coords[:, 0, :, :] / fps

    # Cast via bfloat16 pour reproduire le comportement de PyTorch. PyTorch fait
    # `positions = positions.to(bfloat16)` sur TOUTES les coordonnées avant de les
    # passer au transformer/RoPE. Cette quantification fait partie de l'entraînement,
    # il faut donc la reproduire pour la fidélité numérique.
    positions_bf16 = mx.array(pixel_coords, dtype=mx.bfloat16)
    mx.eval(positions_bf16)
    return positions_bf16.astype(mx.float32)


def create_audio_position_grid(
    batch_size: int,
    audio_frames: int,
    sample_rate: int = AUDIO_LATENT_SAMPLE_RATE,
    hop_length: int = AUDIO_HOP_LENGTH,
    downsample_factor: int = AUDIO_LATENT_DOWNSAMPLE_FACTOR,
    is_causal: bool = True,
) -> mx.array:
    """Crée la grille de positions temporelles pour le RoPE audio."""

    def get_audio_latent_time_in_sec(start_idx: int, end_idx: int) -> np.ndarray:
        latent_frame = np.arange(start_idx, end_idx, dtype=np.float32)
        mel_frame = latent_frame * downsample_factor
        if is_causal:
            mel_frame = np.clip(mel_frame + 1 - downsample_factor, 0, None)
        return mel_frame * hop_length / sample_rate

    start_times = get_audio_latent_time_in_sec(0, audio_frames)
    end_times = get_audio_latent_time_in_sec(1, audio_frames + 1)

    positions = np.stack([start_times, end_times], axis=-1)
    positions = positions[np.newaxis, np.newaxis, :, :]
    positions = np.tile(positions, (batch_size, 1, 1, 1))

    # Cast via bfloat16 pour respecter la précision PyTorch
    positions_bf16 = mx.array(positions, dtype=mx.bfloat16)
    mx.eval(positions_bf16)
    return positions_bf16.astype(mx.float32)


def compute_audio_frames(num_video_frames: int, fps: float) -> int:
    """Calcule le nombre de frames latentes audio pour une durée vidéo donnée."""
    duration = num_video_frames / fps
    return round(duration * AUDIO_LATENTS_PER_SECOND)


def denoise_distilled(
    latents: mx.array,
    positions: mx.array,
    text_embeddings: mx.array,
    transformer: LTXModel,
    sigmas: list,
    verbose: bool = True,
    state: Optional[LatentState] = None,
    audio_latents: Optional[mx.array] = None,
    audio_positions: Optional[mx.array] = None,
    audio_embeddings: Optional[mx.array] = None,
    audio_frozen: bool = False,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    progress_stage: str = "denoising",
) -> tuple[mx.array, Optional[mx.array]]:
    """Exécute une boucle de débruitage par pas d'Euler à sigma fixe pour le pipeline distillé.

    Args:
        latents : latents initiaux ``(B, C, F, H, W)``.
        positions : grille de positions pour le RoPE vidéo.
        text_embeddings : embeddings du prompt textuel encodé.
        transformer : transformer prédicteur de vélocité (:class:`LTXModel`).
        sigmas : planning sigma, p. ex. :data:`STAGE_1_SIGMAS`.
        verbose : afficher la barre de progression rich.
        state : :class:`LatentState` optionnel portant les frames de conditionnement.
        audio_latents : latents audio optionnels ``(B, 8, T, 16)`` pour la génération A/V conjointe.
        audio_positions : grille de positions audio optionnelle.
        audio_embeddings : embeddings textuels audio optionnels.
        audio_frozen : si True (mode Audio-to-Video), les latents audio sont gelés
            (timesteps=0, aucun pas d'Euler ne leur est appliqué).
        on_progress : callback optionnel ``(step, total, stage)`` invoqué à chaque pas
            de débruitage. ``stage`` correspond à la valeur de ``progress_stage``.
        progress_stage : étiquette transmise à ``on_progress`` pour identifier l'étape.

    Returns:
        Tuple ``(video_latents, audio_latents)``, ce dernier valant None quand l'audio
        est désactivé.
    """
    dtype = latents.dtype
    enable_audio = audio_latents is not None

    if state is not None:
        latents = state.latent

    # On garde les latents en float32 partout pour éviter l'accumulation de bruit
    # de quantification entre plusieurs pas d'Euler.
    latents = latents.astype(mx.float32)
    if enable_audio:
        audio_latents = audio_latents.astype(mx.float32)

    desc = "[cyan]Débruitage A/V[/]" if enable_audio else "[cyan]Débruitage[/]"
    num_steps = len(sigmas) - 1

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=not verbose,
    ) as progress:
        task = progress.add_task(desc, total=num_steps)

        for i in range(num_steps):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]

            b, c, f, h, w = latents.shape
            num_tokens = f * h * w
            # Cast vers le dtype du modèle pour l'entrée du transformer
            latents_flat = mx.transpose(
                mx.reshape(latents, (b, c, -1)), (0, 2, 1)
            ).astype(dtype)

            if state is not None:
                denoise_mask_flat = mx.reshape(state.denoise_mask, (b, 1, f, 1, 1))
                denoise_mask_flat = mx.broadcast_to(denoise_mask_flat, (b, 1, f, h, w))
                denoise_mask_flat = mx.reshape(denoise_mask_flat, (b, num_tokens))
                timesteps = mx.array(sigma, dtype=dtype) * denoise_mask_flat
            else:
                timesteps = mx.full((b, num_tokens), sigma, dtype=dtype)

            video_modality = Modality(
                latent=latents_flat,
                timesteps=timesteps,
                positions=positions,
                context=text_embeddings,
                context_mask=None,
                enabled=True,
                sigma=mx.full((b,), sigma, dtype=dtype),
            )

            audio_modality = None
            if enable_audio:
                ab, ac, at, af = audio_latents.shape
                audio_flat = mx.transpose(audio_latents, (0, 2, 1, 3))
                audio_flat = mx.reshape(audio_flat, (ab, at, ac * af)).astype(dtype)

                # A2V : l'audio gelé utilise timesteps=0 (signale au modèle que l'audio est propre)
                a_ts = (
                    mx.zeros((ab, at), dtype=dtype)
                    if audio_frozen
                    else mx.full((ab, at), sigma, dtype=dtype)
                )
                a_sig = (
                    mx.zeros((ab,), dtype=dtype)
                    if audio_frozen
                    else mx.full((ab,), sigma, dtype=dtype)
                )
                audio_modality = Modality(
                    latent=audio_flat,
                    timesteps=a_ts,
                    positions=audio_positions,
                    context=audio_embeddings,
                    context_mask=None,
                    enabled=True,
                    sigma=a_sig,
                )

            velocity, audio_velocity = transformer(
                video=video_modality, audio=audio_modality
            )
            mx.eval(velocity)
            if audio_velocity is not None:
                mx.eval(audio_velocity)

            # Calcul du débruité (x0) avec des timesteps par token en float32
            sigma_f32 = mx.array(sigma, dtype=mx.float32)
            latents_flat_f32 = mx.transpose(mx.reshape(latents, (b, c, -1)), (0, 2, 1))
            timesteps_f32 = mx.expand_dims(timesteps.astype(mx.float32), axis=-1)
            x0_f32 = latents_flat_f32 - timesteps_f32 * velocity.astype(mx.float32)
            denoised = mx.reshape(mx.transpose(x0_f32, (0, 2, 1)), (b, c, f, h, w))

            audio_denoised = None
            if enable_audio and audio_velocity is not None and not audio_frozen:
                ab, ac, at, af = audio_latents.shape
                audio_velocity = mx.reshape(audio_velocity, (ab, at, ac, af))
                audio_velocity = mx.transpose(audio_velocity, (0, 2, 1, 3))
                audio_denoised = audio_latents - sigma_f32 * audio_velocity.astype(
                    mx.float32
                )

            if state is not None:
                denoised = apply_denoise_mask(
                    denoised, state.clean_latent.astype(mx.float32), state.denoise_mask
                )

            mx.eval(denoised)
            if audio_denoised is not None:
                mx.eval(audio_denoised)

            # Pas d'Euler en float32
            if sigma_next > 0:
                sigma_next_f32 = mx.array(sigma_next, dtype=mx.float32)
                latents = denoised + sigma_next_f32 * (latents - denoised) / sigma_f32
                if enable_audio and audio_denoised is not None and not audio_frozen:
                    audio_latents = (
                        audio_denoised
                        + sigma_next_f32 * (audio_latents - audio_denoised) / sigma_f32
                    )
            else:
                latents = denoised
                if enable_audio and audio_denoised is not None and not audio_frozen:
                    audio_latents = audio_denoised

            mx.eval(latents)
            if enable_audio:
                mx.eval(audio_latents)

            progress.advance(task)
            if on_progress is not None:
                on_progress(i + 1, num_steps, progress_stage)

    return (
        latents.astype(dtype),
        audio_latents.astype(dtype) if enable_audio else None,
    )
