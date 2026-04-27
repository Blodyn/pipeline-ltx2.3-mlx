from dataclasses import dataclass, replace
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .attention import Attention
from .config import LTXRopeType, TransformerConfig
from .feed_forward import FeedForward
from ...utils.common import rms_norm


@dataclass(frozen=True)
class Modality:
    latent: mx.array
    timesteps: mx.array
    positions: mx.array
    context: mx.array
    enabled: bool = True
    context_mask: Optional[mx.array] = None
    # Embeddings de position précalculés (RoPE) pour éviter de les recalculer
    positional_embeddings: Optional[Tuple[mx.array, mx.array]] = None
    # Valeur de sigma brute (scalaire par batch) pour le prompt adaln (LTX-2.3)
    sigma: Optional[mx.array] = None


@dataclass(frozen=True)
class TransformerArgs:
    x: mx.array
    context: mx.array
    context_mask: Optional[mx.array]
    timesteps: mx.array
    embedded_timestep: mx.array
    positional_embeddings: Tuple[mx.array, mx.array]
    cross_positional_embeddings: Optional[Tuple[mx.array, mx.array]]
    cross_scale_shift_timestep: Optional[mx.array]
    cross_gate_timestep: Optional[mx.array]
    enabled: bool
    # LTX-2.3 : embeddings de timestep conditionnés par le prompt pour la cross-attention
    prompt_timesteps: Optional[mx.array] = None
    prompt_embedded_timestep: Optional[mx.array] = None


class BasicAVTransformerBlock(nn.Module):
    """Bloc transformer Audio-Vidéo avec cross-attention multi-modale.

    Prend en charge le traitement vidéo seul, audio seul ou audio-vidéo combiné
    avec une cross-attention bidirectionnelle entre les modalités.
    """

    def __init__(
        self,
        idx: int,
        video: Optional[TransformerConfig] = None,
        audio: Optional[TransformerConfig] = None,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        norm_eps: float = 1e-6,
        has_prompt_adaln: bool = False,
    ):
        super().__init__()

        self.idx = idx
        self.norm_eps = norm_eps
        self.has_prompt_adaln = has_prompt_adaln

        # Composants vidéo
        if video is not None:
            self.attn1 = Attention(
                query_dim=video.dim,
                heads=video.heads,
                dim_head=video.d_head,
                context_dim=None,  # Auto-attention
                rope_type=rope_type,
                norm_eps=norm_eps,
                has_gate_logits=has_prompt_adaln,
            )
            self.attn2 = Attention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                has_gate_logits=has_prompt_adaln,
            )
            self.ff = FeedForward(video.dim, dim_out=video.dim)
            # 9 paramètres pour LTX-2.3 (self-attn + cross-attn + FFN), 6 pour LTX-2
            num_ada_params = 9 if has_prompt_adaln else 6
            self.scale_shift_table = mx.zeros((num_ada_params, video.dim))

            if has_prompt_adaln:
                self.prompt_scale_shift_table = mx.zeros((2, video.dim))

        # Composants audio
        if audio is not None:
            self.audio_attn1 = Attention(
                query_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                has_gate_logits=has_prompt_adaln,
            )
            self.audio_attn2 = Attention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                has_gate_logits=has_prompt_adaln,
            )
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim)
            num_audio_ada_params = 9 if has_prompt_adaln else 6
            self.audio_scale_shift_table = mx.zeros((num_audio_ada_params, audio.dim))

            if has_prompt_adaln:
                self.audio_prompt_scale_shift_table = mx.zeros((2, audio.dim))

        # Cross-attention multi-modale (quand vidéo et audio sont actifs tous les deux)
        if audio is not None and video is not None:
            # Audio-vers-Vidéo : Q vient de la vidéo, K/V de l'audio
            self.audio_to_video_attn = Attention(
                query_dim=video.dim,
                context_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                has_gate_logits=has_prompt_adaln,
            )
            # Vidéo-vers-Audio : Q vient de l'audio, K/V de la vidéo
            self.video_to_audio_attn = Attention(
                query_dim=audio.dim,
                context_dim=video.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                has_gate_logits=has_prompt_adaln,
            )
            # Tables scale-shift pour la cross-attention
            self.scale_shift_table_a2v_ca_audio = mx.zeros((5, audio.dim))
            self.scale_shift_table_a2v_ca_video = mx.zeros((5, video.dim))

    def get_ada_values(
        self,
        scale_shift_table: mx.array,
        batch_size: int,
        timestep: mx.array,
        indices: slice,
    ) -> Tuple[mx.array, ...]:
        """Récupère les valeurs de normalisation adaptative depuis la table scale-shift.

        Args:
            scale_shift_table : table de forme (num_params, dim)
            batch_size : taille du batch
            timestep : embeddings de timestep de forme (B, 1, num_params * dim) ou similaire
            indices : slice indiquant les paramètres à extraire

        Returns:
            Tuple des valeurs scale-shift
        """
        num_ada_params = scale_shift_table.shape[0]

        # scale_shift_table[indices] : (num_selected, dim)
        # Ajout des dims batch et séquence : (1, 1, num_selected, dim)
        table_slice = scale_shift_table[indices]
        table_expanded = mx.expand_dims(mx.expand_dims(table_slice, axis=0), axis=0)

        # timestep : (B, seq, num_params * dim) -> reshape en (B, seq, num_params, dim)
        timestep_reshaped = mx.reshape(
            timestep, (batch_size, timestep.shape[1], num_ada_params, -1)
        )

        # Extraction des indices pertinents
        timestep_slice = timestep_reshaped[:, :, indices, :]

        # Ajout des valeurs de la table au timestep
        ada_values = table_expanded + timestep_slice

        # Détachement le long de la dimension paramètres
        # Résultat : tuple de tenseurs, chacun de forme (B, seq, dim)
        num_sliced = ada_values.shape[2]
        result = tuple(ada_values[:, :, i, :] for i in range(num_sliced))

        return result

    def get_av_ca_ada_values(
        self,
        scale_shift_table: mx.array,
        batch_size: int,
        scale_shift_timestep: mx.array,
        gate_timestep: mx.array,
        num_scale_shift_values: int = 4,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        """Récupère les valeurs adaptatives pour la cross-attention multi-modale.

        Args:
            scale_shift_table : table à 5 paramètres (4 scale-shift + 1 gate)
            batch_size : taille du batch
            scale_shift_timestep : timestep pour le scale-shift
            gate_timestep : timestep pour le gating
            num_scale_shift_values : nombre de valeurs scale-shift (4 par défaut)

        Returns:
            Tuple de 5 tenseurs : (scale1, shift1, scale2, shift2, gate)
        """
        # Récupération des valeurs scale-shift
        scale_shift_ada = self.get_ada_values(
            scale_shift_table[:num_scale_shift_values, :],
            batch_size,
            scale_shift_timestep,
            slice(None, None),
        )

        # Récupération des valeurs de gate
        gate_ada = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :],
            batch_size,
            gate_timestep,
            slice(None, None),
        )

        # Squeeze de la dimension séquence si elle vaut 1
        scale_shift_squeezed = tuple(
            mx.squeeze(t, axis=1) if t.shape[1] == 1 else t for t in scale_shift_ada
        )
        gate_squeezed = tuple(
            mx.squeeze(t, axis=1) if t.shape[1] == 1 else t for t in gate_ada
        )

        return (*scale_shift_squeezed, *gate_squeezed)

    def __call__(
        self,
        video: Optional[TransformerArgs] = None,
        audio: Optional[TransformerArgs] = None,
        skip_video_self_attn: bool = False,
        skip_audio_self_attn: bool = False,
        skip_cross_modal: bool = False,
    ) -> Tuple[Optional[TransformerArgs], Optional[TransformerArgs]]:
        """Passe avant à travers le bloc transformer.

        Args:
            video : arguments de la modalité vidéo
            audio : arguments de la modalité audio
            skip_video_self_attn : saute l'auto-attention vidéo (pour la perturbation STG)
            skip_audio_self_attn : saute l'auto-attention audio (pour la perturbation STG)
            skip_cross_modal : saute toute la cross-attention multi-modale (isolation des modalités)

        Returns:
            Tuple (updated_video, updated_audio) de TransformerArgs
        """
        batch_size = video.x.shape[0] if video is not None else audio.x.shape[0]

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        # Détermination des modalités à exécuter
        run_vx = video is not None and video.enabled and vx.size > 0
        run_ax = audio is not None and audio.enabled and ax.size > 0
        run_a2v = (
            run_vx
            and (audio is not None and audio.enabled and ax.size > 0)
            and not skip_cross_modal
        )
        run_v2a = (
            run_ax
            and (video is not None and video.enabled and vx.size > 0)
            and not skip_cross_modal
        )

        # Auto-attention vidéo + cross-attention avec le texte
        if run_vx:
            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
            )

            # Auto-attention avec RoPE (skip_attention=True pour la perturbation STG)
            norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_msa) + vshift_msa
            vx = (
                vx
                + self.attn1(
                    norm_vx,
                    pe=video.positional_embeddings,
                    skip_attention=skip_video_self_attn,
                )
                * vgate_msa
            )

            # Cross-attention avec le contexte texte
            if self.has_prompt_adaln:
                # LTX-2.3 : Q modulé par le timestep (indices 6-8), contexte modulé par prompt_adaln
                vshift_q, vscale_q, vgate_q = self.get_ada_values(
                    self.scale_shift_table, vx.shape[0], video.timesteps, slice(6, 9)
                )
                vprompt_shift_kv, vprompt_scale_kv = self.get_ada_values(
                    self.prompt_scale_shift_table,
                    vx.shape[0],
                    video.prompt_timesteps,
                    slice(0, 2),
                )
                attn_input = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_q) + vshift_q
                encoder_hidden_states = (
                    video.context * (1 + vprompt_scale_kv) + vprompt_shift_kv
                )
                vx = (
                    vx
                    + self.attn2(
                        attn_input,
                        context=encoder_hidden_states,
                        mask=video.context_mask,
                    )
                    * vgate_q
                )
            else:
                vx = vx + self.attn2(
                    rms_norm(vx, eps=self.norm_eps),
                    context=video.context,
                    mask=video.context_mask,
                )

        # Auto-attention audio + cross-attention avec le texte
        if run_ax:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
            )

            # Auto-attention avec RoPE (skip_attention=True pour la perturbation STG)
            norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa
            ax = (
                ax
                + self.audio_attn1(
                    norm_ax,
                    pe=audio.positional_embeddings,
                    skip_attention=skip_audio_self_attn,
                )
                * agate_msa
            )

            # Cross-attention avec le contexte texte
            if self.has_prompt_adaln:
                # LTX-2.3 : Q modulé par le timestep (indices 6-8), contexte modulé par prompt_adaln
                ashift_q, ascale_q, agate_q = self.get_ada_values(
                    self.audio_scale_shift_table,
                    ax.shape[0],
                    audio.timesteps,
                    slice(6, 9),
                )
                aprompt_shift_kv, aprompt_scale_kv = self.get_ada_values(
                    self.audio_prompt_scale_shift_table,
                    ax.shape[0],
                    audio.prompt_timesteps,
                    slice(0, 2),
                )
                attn_input_a = (
                    rms_norm(ax, eps=self.norm_eps) * (1 + ascale_q) + ashift_q
                )
                encoder_hidden_states_a = (
                    audio.context * (1 + aprompt_scale_kv) + aprompt_shift_kv
                )
                ax = (
                    ax
                    + self.audio_attn2(
                        attn_input_a,
                        context=encoder_hidden_states_a,
                        mask=audio.context_mask,
                    )
                    * agate_q
                )
            else:
                ax = ax + self.audio_attn2(
                    rms_norm(ax, eps=self.norm_eps),
                    context=audio.context,
                    mask=audio.context_mask,
                )

        # Cross-attention multi-modale Audio-Vidéo
        if run_a2v or run_v2a:
            vx_norm3 = rms_norm(vx, eps=self.norm_eps)
            ax_norm3 = rms_norm(ax, eps=self.norm_eps)

            # Récupération des valeurs adaptatives pour la cross-attention audio
            (
                scale_ca_audio_a2v,
                shift_ca_audio_a2v,
                scale_ca_audio_v2a,
                shift_ca_audio_v2a,
                gate_out_v2a,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_audio,
                ax.shape[0],
                audio.cross_scale_shift_timestep,
                audio.cross_gate_timestep,
            )

            # Récupération des valeurs adaptatives pour la cross-attention vidéo
            (
                scale_ca_video_a2v,
                shift_ca_video_a2v,
                scale_ca_video_v2a,
                shift_ca_video_v2a,
                gate_out_a2v,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_video,
                vx.shape[0],
                video.cross_scale_shift_timestep,
                video.cross_gate_timestep,
            )

            # Cross-attention Audio-vers-Vidéo
            if run_a2v:
                vx_scaled = vx_norm3 * (1 + scale_ca_video_a2v) + shift_ca_video_a2v
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_a2v) + shift_ca_audio_a2v
                vx = vx + (
                    self.audio_to_video_attn(
                        vx_scaled,
                        context=ax_scaled,
                        pe=video.cross_positional_embeddings,
                        k_pe=audio.cross_positional_embeddings,
                    )
                    * gate_out_a2v
                )

            # Cross-attention Vidéo-vers-Audio
            if run_v2a:
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_v2a) + shift_ca_audio_v2a
                vx_scaled = vx_norm3 * (1 + scale_ca_video_v2a) + shift_ca_video_v2a
                ax = ax + (
                    self.video_to_audio_attn(
                        ax_scaled,
                        context=vx_scaled,
                        pe=audio.cross_positional_embeddings,
                        k_pe=video.cross_positional_embeddings,
                    )
                    * gate_out_v2a
                )

        # Feed-forward vidéo
        if run_vx:
            vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, 6)
            )
            vx_scaled = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_mlp) + vshift_mlp
            vx = vx + self.ff(vx_scaled) * vgate_mlp

        # Feed-forward audio
        if run_ax:
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, 6)
            )
            ax_scaled = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_mlp) + ashift_mlp
            ax = ax + self.audio_ff(ax_scaled) * agate_mlp

        # Renvoi des TransformerArgs mis à jour
        video_out = replace(video, x=vx) if video is not None else None
        audio_out = replace(audio, x=ax) if audio is not None else None

        return video_out, audio_out
