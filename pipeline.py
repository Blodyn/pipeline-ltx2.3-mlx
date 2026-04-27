"""High-level inference API for LTX-2.3.

Exposes :class:`LTXPipeline`, a stateful wrapper that loads the LTX-2.3
transformer, video VAE, audio VAE, vocoder, text encoder and spatial
upscaler once, then lets callers run multiple ``generate(...)`` invocations
without paying the model-loading cost (~30-60s on Apple Silicon) each time.

The pipeline returns raw numpy arrays (frames + optional audio samples).
Saving to disk, muxing audio with video, or streaming to a frontend is the
responsibility of the calling application.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import mlx.core as mx
import numpy as np
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from .models.ltx.conditioning import (
    VideoConditionByLatentIndex,
    apply_conditioning,
)
from .models.ltx.conditioning.latent import LatentState
from .models.ltx.denoise import (
    AUDIO_LATENT_CHANNELS,
    AUDIO_LATENT_SAMPLE_RATE,
    AUDIO_MEL_BINS,
    AUDIO_SAMPLE_RATE,
    STAGE_1_SIGMAS,
    STAGE_2_SIGMAS,
    compute_audio_frames,
    create_audio_position_grid,
    create_position_grid,
    denoise_distilled,
)
from .models.ltx.lora import load_and_merge_lora
from .models.ltx.ltx import LTXModel
from .models.ltx.upsampler import load_upsampler, upsample_latents
from .models.ltx.utils import get_model_path
from .models.ltx.video_vae import VideoEncoder
from .models.ltx.video_vae.decoder import VideoDecoder
from .models.ltx.video_vae.tiling import TilingConfig
from .utils.common import load_image, prepare_image_for_encoding

console = Console()


# Type aliases
ProgressCallback = Callable[[int, int, str], None]
"""Signature: ``on_progress(step, total, stage_name)``."""

FramesCallback = Callable[[np.ndarray, int], None]
"""Signature: ``on_frames_ready(frames_uint8, start_index)`` where
``frames_uint8`` has shape ``(N, H, W, 3)`` in 0-255 range."""


@dataclass
class GenerationResult:
    """Result of a :meth:`LTXPipeline.generate` call.

    Attributes:
        frames: Video frames as ``np.uint8`` of shape ``(T, H, W, 3)`` in
            RGB order, range 0-255.
        audio: Audio samples as ``np.float32`` of shape ``(channels, samples)``
            in range [-1, 1], or None when audio is disabled.
        audio_sample_rate: Sample rate of ``audio`` in Hz (typically 24000
            for synthesized audio, 16000 for A2V passthrough). 0 when no
            audio is returned.
        fps: Frame rate the video was generated at.
        elapsed_seconds: Wall-clock time spent inside ``generate()``.
        peak_memory_gb: Peak MLX memory usage observed during generation.
        metadata: Free-form dictionary of additional metadata (mode, prompt,
            seed, ...).
    """

    frames: np.ndarray
    audio: Optional[np.ndarray] = None
    audio_sample_rate: int = 0
    fps: int = 24
    elapsed_seconds: float = 0.0
    peak_memory_gb: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class LTXPipeline:
    """Stateful inference pipeline for LTX-2.3 (distilled checkpoint).

    Typical usage::

        pipeline = LTXPipeline(model_repo="prince-canuma/LTX-2.3-distilled")
        pipeline.load()
        result = pipeline.generate(prompt="A cinematic ocean scene", num_frames=33)
        frames = result.frames                    # (T, H, W, 3) uint8
        audio = result.audio                      # None for muted T2V
        # ... pipeline.generate(...) repeatedly without re-loading ...
        pipeline.unload()

    The class is intentionally small in surface area: a single
    :meth:`generate` method accepts every combination of T2V, I2V, A2V, and
    joint audio-video synthesis. All optional inputs default to None and
    are activated by passing the corresponding argument.

    Args:
        model_repo: Local path or HuggingFace repo id holding the LTX-2.3
            transformer + VAE + audio VAE + vocoder. Default targets the
            pre-converted MLX checkpoint.
        text_encoder_repo: Local path or HuggingFace repo id of the Gemma 3
            text encoder. ``None`` falls back to the model_repo (some
            checkpoints bundle Gemma alongside the transformer).
        verbose: When True, rich panels and progress bars are emitted to the
            terminal. Set False for silent operation; the per-step
            ``on_progress`` callback still fires either way.
    """

    DEFAULT_MODEL_REPO = "prince-canuma/LTX-2.3-distilled"
    DEFAULT_TEXT_ENCODER_REPO = "google/gemma-3-12b-it"

    def __init__(
        self,
        model_repo: str = DEFAULT_MODEL_REPO,
        text_encoder_repo: Optional[str] = DEFAULT_TEXT_ENCODER_REPO,
        verbose: bool = True,
    ):
        self.model_repo = model_repo
        self.text_encoder_repo = text_encoder_repo
        self.verbose = verbose

        # Resolved at load() time
        self.model_path: Optional[Path] = None
        self.text_encoder_path: Optional[Path] = None

        # Lazily loaded components
        self._text_encoder = None  # LTX2TextEncoder
        self._transformer: Optional[LTXModel] = None
        self._has_prompt_adaln: bool = True

        # LoRA bookkeeping: we track whether a LoRA was merged so the user
        # can ask for a clean reload.
        self._lora_merged: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load all model components into memory.

        Idempotent: a second call is a no-op. The text encoder, transformer
        and VAE encoders/decoders are kept in memory until :meth:`unload`
        is called.
        """
        if self._transformer is not None:
            return  # already loaded

        self._log_panel(
            f"[bold cyan]🎬 Loading LTX-2.3 from[/] {self.model_repo}"
        )

        # Resolve model paths (downloads if not present locally).
        self.model_path = get_model_path(self.model_repo)
        self.text_encoder_path = (
            self.model_path
            if self.text_encoder_repo is None
            else get_model_path(self.text_encoder_repo)
        )

        # Detect LTX-2.3 vs LTX-2 from transformer config (distilled is 2.3
        # in our supported case but we still respect the file).
        import json

        cfg_path = self.model_path / "transformer" / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                self._has_prompt_adaln = json.load(f).get("has_prompt_adaln", True)

        # ---- Text encoder -------------------------------------------------
        from .models.ltx.text_encoder import LTX2TextEncoder

        with self._console_status("[blue]📝 Loading text encoder...[/]"):
            self._text_encoder = LTX2TextEncoder(
                has_prompt_adaln=self._has_prompt_adaln
            )
            self._text_encoder.load(
                model_path=self.model_path,
                text_encoder_path=self.text_encoder_path,
            )
            mx.eval(self._text_encoder.parameters())
        self._log("[green]✓[/] Text encoder loaded")

        # ---- Transformer --------------------------------------------------
        with self._console_status("[blue]🤖 Loading transformer...[/]"):
            self._transformer = LTXModel.from_pretrained(
                model_path=self.model_path / "transformer", strict=True
            )
        self._log("[green]✓[/] Transformer loaded")

    def unload(self) -> None:
        """Release all loaded models and their MLX buffers."""
        self._text_encoder = None
        self._transformer = None
        self.model_path = None
        self.text_encoder_path = None
        self._lora_merged = None
        mx.clear_cache()

    @property
    def is_loaded(self) -> bool:
        """True iff at least the transformer is loaded in memory."""
        return self._transformer is not None

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        # Resolution / length
        height: int = 512,
        width: int = 512,
        num_frames: int = 33,
        fps: int = 24,
        # Image conditioning (I2V)
        image: Optional[Union[str, Path]] = None,
        image_strength: float = 1.0,
        image_frame_idx: int = 0,
        # Audio conditioning (A2V) and audio synthesis
        audio: bool = False,
        audio_file: Optional[Union[str, Path]] = None,
        audio_start_time: float = 0.0,
        # Sampling
        seed: int = 42,
        # LoRA (external, e.g. camera-control)
        lora_path: Optional[str] = None,
        lora_strength: float = 1.0,
        # Spatial upscaler (between stage 1 and stage 2)
        spatial_upscaler: Optional[str] = None,
        # Decoder behavior
        tiling: str = "auto",
        # Streaming + progress hooks
        on_progress: Optional[ProgressCallback] = None,
        on_frames_ready: Optional[FramesCallback] = None,
        stream: bool = False,
        # Debugging
        save_frames: bool = False,
        save_frames_dir: Optional[Union[str, Path]] = None,
        verbose: Optional[bool] = None,
    ) -> GenerationResult:
        """Generate a video (and optionally audio) from a prompt.

        See module docstring and :class:`GenerationResult` for return shape.

        Args:
            prompt: Text description of the desired output.
            height: Output height in pixels. Must be divisible by 64.
            width: Output width in pixels. Must be divisible by 64.
            num_frames: Number of frames to generate. Must be ``1 + 8*k``.
            fps: Frame rate for position grid + audio length computation.
            image: Path to a conditioning image for image-to-video.
            image_strength: Conditioning strength in [0, 1] for I2V.
            image_frame_idx: Frame index where the conditioning is injected.
            audio: When True, synthesizes audio jointly with video. Mutually
                exclusive with ``audio_file``.
            audio_file: Path to a WAV/MP3/FLAC/OGG/video file used for
                Audio-to-Video conditioning. Frozen during denoising.
            audio_start_time: Start offset (seconds) inside ``audio_file``.
            seed: Random seed for reproducibility.
            lora_path: Optional LoRA path/HF repo to merge into the
                transformer for this run.
            lora_strength: Scalar multiplier applied to the LoRA delta.
            spatial_upscaler: Filename of the latent upscaler safetensors
                inside ``model_repo``. ``None`` auto-detects an x2 file.
            tiling: VAE decoder tiling mode (``"auto"``, ``"none"``,
                ``"default"``, ``"aggressive"``, ``"conservative"``,
                ``"spatial"`` or ``"temporal"``).
            on_progress: Callback ``(step, total, stage)`` fired during
                denoising, upsampling and VAE decoding.
            on_frames_ready: Callback ``(frames_uint8, start_idx)`` fired
                during the VAE decode when ``stream`` is True. Useful for
                preview UIs.
            stream: When True, decoded frames are emitted progressively via
                ``on_frames_ready`` instead of (in addition to) being
                returned in bulk.
            save_frames: When True, individual PNG frames are written to
                ``save_frames_dir``.
            save_frames_dir: Directory where frames are saved when
                ``save_frames=True``. Defaults to ``./frames``.
            verbose: Override the pipeline-level verbosity for this call.

        Returns:
            :class:`GenerationResult`.
        """
        if not self.is_loaded:
            self.load()

        v = self.verbose if verbose is None else verbose
        start_time = time.time()

        # ---- Validate dimensions -----------------------------------------
        # The distilled pipeline is two-stage so requires divisor 64
        # (stage 1 runs at half resolution and must still be VAE-aligned).
        divisor = 64
        if height % divisor != 0:
            raise ValueError(f"height must be divisible by {divisor}, got {height}")
        if width % divisor != 0:
            raise ValueError(f"width must be divisible by {divisor}, got {width}")

        if num_frames % 8 != 1:
            adjusted = round((num_frames - 1) / 8) * 8 + 1
            if v:
                console.print(
                    f"[yellow]⚠️  num_frames must be 1 + 8*k. Using: {adjusted}[/]"
                )
            num_frames = adjusted

        is_i2v = image is not None
        is_a2v = audio_file is not None
        if is_a2v and audio:
            raise ValueError(
                "audio_file (A2V) and audio (synthesize audio) are mutually "
                "exclusive. Choose one."
            )
        # A2V implicitly enables the audio branch in the transformer.
        if is_a2v:
            audio = True

        mode = "I2V" if is_i2v else "T2V"
        if is_a2v:
            mode = "A2V" + ("+I2V" if is_i2v else "")
        elif audio:
            mode = mode + "+Audio"

        if v:
            console.print(
                Panel(
                    f"[bold cyan]🎬 [DISTILLED] [{mode}] "
                    f"{width}x{height} • {num_frames} frames[/]",
                    expand=False,
                )
            )
            console.print(
                f"[dim]Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}[/]"
            )

        # ---- LoRA merge --------------------------------------------------
        if lora_path is not None:
            if v:
                console.print(
                    f"[bold magenta]🎨 Merging LoRA[/] from {lora_path}"
                )
            load_and_merge_lora(
                self._transformer,
                lora_path,
                strength=lora_strength,
                verbose=v,
            )
            self._lora_merged = lora_path

        # ---- Spatial upscaler resolution ---------------------------------
        upscaler_path, upscaler_scale = self._resolve_upscaler(spatial_upscaler)

        # ---- Calculate latent dimensions ---------------------------------
        # Distilled is two-stage: stage 1 always at half spatial resolution.
        stage1_h = height // 2 // 32
        stage1_w = width // 2 // 32
        stage2_h = int(stage1_h * upscaler_scale)
        stage2_w = int(stage1_w * upscaler_scale)
        latent_frames = 1 + (num_frames - 1) // 8

        mx.random.seed(seed)
        audio_frames = compute_audio_frames(num_frames, fps)
        if audio and v:
            console.print(
                f"[dim]Audio: {audio_frames} latent frames @ {AUDIO_SAMPLE_RATE}Hz[/]"
            )

        # ---- Encode prompt ----------------------------------------------
        text_embeddings, audio_embeddings = self._text_encoder(
            prompt, return_audio_embeddings=True
        )
        mx.eval(text_embeddings, audio_embeddings)
        model_dtype = text_embeddings.dtype

        # ---- A2V: encode input audio to frozen latents -------------------
        a2v_audio_latents = None
        a2v_waveform = None
        a2v_sr = None
        if is_a2v:
            a2v_audio_latents, a2v_waveform, a2v_sr = self._encode_input_audio(
                audio_file=audio_file,
                audio_start_time=audio_start_time,
                num_frames=num_frames,
                fps=fps,
                audio_frames=audio_frames,
                model_dtype=model_dtype,
                verbose=v,
            )

        # ---- Stage 1 ----------------------------------------------------
        if v:
            console.print(
                f"\n[bold yellow]⚡ Stage 1:[/] Generating at "
                f"{stage1_w * 32}x{stage1_h * 32} (8 steps)"
            )

        # I2V conditioning encoding (encode at both stage1 and stage2 sizes)
        stage1_image_latent = None
        stage2_image_latent = None
        if is_i2v:
            stage1_image_latent, stage2_image_latent = self._encode_image(
                image=image,
                stage1_h=stage1_h,
                stage1_w=stage1_w,
                stage2_h=stage2_h,
                stage2_w=stage2_w,
                model_dtype=model_dtype,
                verbose=v,
            )

        mx.random.seed(seed)
        positions = create_position_grid(1, latent_frames, stage1_h, stage1_w, fps=fps)
        mx.eval(positions)

        audio_positions = create_audio_position_grid(1, audio_frames)
        if is_a2v:
            audio_latents = a2v_audio_latents
        else:
            audio_latents = mx.random.normal(
                (1, AUDIO_LATENT_CHANNELS, audio_frames, AUDIO_MEL_BINS)
            ).astype(model_dtype)
        mx.eval(audio_positions, audio_latents)

        latent_shape_s1 = (1, 128, latent_frames, stage1_h, stage1_w)
        latents, state1 = self._init_latents(
            shape=latent_shape_s1,
            image_latent=stage1_image_latent,
            image_strength=image_strength,
            image_frame_idx=image_frame_idx,
            initial_sigma=STAGE_1_SIGMAS[0],
            dtype=model_dtype,
        )

        latents, audio_latents = denoise_distilled(
            latents,
            positions,
            text_embeddings,
            self._transformer,
            STAGE_1_SIGMAS,
            verbose=v,
            state=state1,
            audio_latents=audio_latents,
            audio_positions=audio_positions,
            audio_embeddings=audio_embeddings,
            audio_frozen=is_a2v,
            on_progress=on_progress,
            progress_stage="stage_1",
        )

        # ---- Spatial upsampling -----------------------------------------
        with self._console_status(
            f"[magenta]🔍 Upsampling latents {upscaler_scale}x...[/]"
        ):
            if upscaler_path is None or not upscaler_path.exists():
                raise FileNotFoundError(
                    f"No spatial upscaler found in {self.model_path}"
                )
            upsampler, upscaler_scale = load_upsampler(str(upscaler_path))
            mx.eval(upsampler.parameters())

            vae_decoder = VideoDecoder.from_pretrained(
                str(self.model_path / "vae" / "decoder")
            )

            latents = upsample_latents(
                latents,
                upsampler,
                vae_decoder.per_channel_statistics.mean,
                vae_decoder.per_channel_statistics.std,
            )
            mx.eval(latents)

            del upsampler
            mx.clear_cache()
        if v:
            console.print("[green]✓[/] Latents upsampled")
        if on_progress is not None:
            on_progress(1, 1, "upsample")

        # ---- Stage 2 ----------------------------------------------------
        if v:
            console.print(
                f"\n[bold yellow]⚡ Stage 2:[/] Refining at "
                f"{stage2_w * 32}x{stage2_h * 32} (3 steps)"
            )

        positions = create_position_grid(1, latent_frames, stage2_h, stage2_w, fps=fps)
        mx.eval(positions)

        latents, state2 = self._renoise_for_stage2(
            latents=latents,
            image_latent=stage2_image_latent,
            image_strength=image_strength,
            image_frame_idx=image_frame_idx,
            initial_sigma=STAGE_2_SIGMAS[0],
            dtype=model_dtype,
        )

        # Re-noise audio at sigma=0.909375 for joint refinement
        # (matches PyTorch reference implementation).
        if audio_latents is not None and not is_a2v:
            audio_noise = mx.random.normal(audio_latents.shape, dtype=model_dtype)
            audio_noise_scale = mx.array(STAGE_2_SIGMAS[0], dtype=model_dtype)
            audio_latents = audio_noise * audio_noise_scale + audio_latents * (
                mx.array(1.0, dtype=model_dtype) - audio_noise_scale
            )
            mx.eval(audio_latents)

        latents, audio_latents = denoise_distilled(
            latents,
            positions,
            text_embeddings,
            self._transformer,
            STAGE_2_SIGMAS,
            verbose=v,
            state=state2,
            audio_latents=audio_latents,
            audio_positions=audio_positions,
            audio_embeddings=audio_embeddings,
            audio_frozen=is_a2v,
            on_progress=on_progress,
            progress_stage="stage_2",
        )

        # ---- Decode video -----------------------------------------------
        if v:
            console.print("\n[blue]🎞️  Decoding video...[/]")

        tiling_config = self._build_tiling_config(tiling, height, width, num_frames, v)

        video_np = self._decode_video(
            vae_decoder=vae_decoder,
            latents=latents,
            tiling_config=tiling_config,
            tiling_mode=tiling,
            stream=stream,
            on_frames_ready=on_frames_ready,
            on_progress=on_progress,
            num_frames=num_frames,
            verbose=v,
        )
        del vae_decoder
        mx.clear_cache()

        # ---- Decode audio ----------------------------------------------
        audio_np = None
        audio_sr = 0
        if audio and audio_latents is not None:
            audio_np, audio_sr = self._decode_audio(
                audio_latents=audio_latents,
                is_a2v=is_a2v,
                a2v_waveform=a2v_waveform,
                a2v_sr=a2v_sr,
                verbose=v,
            )

        # ---- Optionally save individual frames -------------------------
        if save_frames:
            target_dir = Path(save_frames_dir) if save_frames_dir else Path("frames")
            target_dir.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(video_np):
                Image.fromarray(frame).save(target_dir / f"frame_{i:04d}.png")
            if v:
                console.print(
                    f"[green]✓[/] Saved {len(video_np)} frames to {target_dir}"
                )

        elapsed = time.time() - start_time
        peak_mem = mx.get_peak_memory() / (1024**3)
        if v:
            minutes, seconds = divmod(elapsed, 60)
            time_str = (
                f"{int(minutes)}m {seconds:.1f}s" if minutes >= 1 else f"{seconds:.1f}s"
            )
            console.print(
                Panel(
                    f"[bold green]🎉 Done![/] Generated in {time_str} "
                    f"({elapsed / num_frames:.2f}s/frame)\n"
                    f"[bold green]✨ Peak memory:[/] {peak_mem:.2f}GB",
                    expand=False,
                )
            )

        return GenerationResult(
            frames=video_np,
            audio=audio_np,
            audio_sample_rate=audio_sr,
            fps=fps,
            elapsed_seconds=elapsed,
            peak_memory_gb=peak_mem,
            metadata={
                "prompt": prompt,
                "mode": mode,
                "seed": seed,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "upscaler_scale": upscaler_scale,
                "lora_path": lora_path,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_upscaler(self, spatial_upscaler: Optional[str]):
        """Locate the spatial upscaler safetensors file inside model_path."""
        if spatial_upscaler is not None:
            user_path = Path(spatial_upscaler)
            if user_path.is_absolute():
                upscaler_path = user_path
            else:
                upscaler_path = self.model_path / spatial_upscaler

            scale = 1.5 if "x1.5" in str(upscaler_path) else 2.0
            return upscaler_path, scale

        # Auto-detect: prefer x2
        candidates = sorted(self.model_path.glob("*spatial-upscaler-x2*.safetensors"))
        if candidates:
            return candidates[0], 2.0
        # Fallback to x1.5 if no x2 is shipped
        candidates = sorted(self.model_path.glob("*spatial-upscaler-x1.5*.safetensors"))
        if candidates:
            return candidates[0], 1.5
        return None, 2.0

    def _encode_image(
        self,
        image,
        stage1_h: int,
        stage1_w: int,
        stage2_h: int,
        stage2_w: int,
        model_dtype,
        verbose: bool,
    ):
        """Run the VAE encoder on the conditioning image at both resolutions."""
        with self._console_status(
            "[blue]🖼️  Loading VAE encoder and encoding image...[/]"
        ):
            vae_encoder = VideoEncoder.from_pretrained(
                self.model_path / "vae" / "encoder"
            )

            s1_h, s1_w = stage1_h * 32, stage1_w * 32
            input_image = load_image(image, height=s1_h, width=s1_w, dtype=model_dtype)
            stage1_tensor = prepare_image_for_encoding(
                input_image, s1_h, s1_w, dtype=model_dtype
            )
            stage1_latent = vae_encoder(stage1_tensor)
            mx.eval(stage1_latent)

            s2_h, s2_w = stage2_h * 32, stage2_w * 32
            input_image = load_image(image, height=s2_h, width=s2_w, dtype=model_dtype)
            stage2_tensor = prepare_image_for_encoding(
                input_image, s2_h, s2_w, dtype=model_dtype
            )
            stage2_latent = vae_encoder(stage2_tensor)
            mx.eval(stage2_latent)

            del vae_encoder
            mx.clear_cache()

        if verbose:
            console.print("[green]✓[/] VAE encoder loaded and image encoded")
        return stage1_latent, stage2_latent

    def _encode_input_audio(
        self,
        audio_file,
        audio_start_time: float,
        num_frames: int,
        fps: int,
        audio_frames: int,
        model_dtype,
        verbose: bool,
    ):
        """Encode an input audio file to frozen latents for A2V."""
        from .models.ltx.audio_vae import AudioEncoder
        from .models.ltx.audio_vae.audio_processor import (
            ensure_stereo,
            load_audio,
            waveform_to_mel,
        )
        from .models.ltx.utils import convert_audio_encoder

        with self._console_status(
            "[blue]Loading and encoding input audio (A2V)...[/]"
        ):
            video_duration = num_frames / fps

            waveform, sr = load_audio(
                audio_file,
                target_sr=AUDIO_LATENT_SAMPLE_RATE,
                start_time=audio_start_time,
                max_duration=video_duration,
            )
            waveform = ensure_stereo(waveform)
            a2v_waveform = waveform.copy()
            a2v_sr = sr

            mel = waveform_to_mel(
                waveform,
                sample_rate=sr,
                n_fft=1024,
                hop_length=160,
                n_mels=64,
            )

            # Lazily extract the audio encoder weights from the original
            # Lightricks checkpoint if not already present locally.
            encoder_dir = convert_audio_encoder(
                self.model_path, source_repo="Lightricks/LTX-2"
            )
            audio_encoder = AudioEncoder.from_pretrained(encoder_dir)
            mx.eval(audio_encoder.parameters())

            encoded = audio_encoder(mel)
            mx.eval(encoded)

            # MLX layout (B, T', mel_bins', z_channels) -> (B, C, T, mel_bins)
            a2v_audio_latents = mx.transpose(encoded, (0, 3, 1, 2)).astype(model_dtype)

            t_encoded = a2v_audio_latents.shape[2]
            if t_encoded > audio_frames:
                a2v_audio_latents = a2v_audio_latents[:, :, :audio_frames, :]
            elif t_encoded < audio_frames:
                pad_size = audio_frames - t_encoded
                padding = mx.zeros(
                    (1, AUDIO_LATENT_CHANNELS, pad_size, AUDIO_MEL_BINS),
                    dtype=model_dtype,
                )
                a2v_audio_latents = mx.concatenate(
                    [a2v_audio_latents, padding], axis=2
                )
            mx.eval(a2v_audio_latents)

            del audio_encoder
            mx.clear_cache()

        if verbose:
            console.print(
                f"[green]✓[/] Audio encoded "
                f"({a2v_audio_latents.shape[2]} frames from {audio_file})"
            )
        return a2v_audio_latents, a2v_waveform, a2v_sr

    def _init_latents(
        self,
        shape,
        image_latent,
        image_strength: float,
        image_frame_idx: int,
        initial_sigma: float,
        dtype,
    ):
        """Initialize stage-1 latents with optional I2V conditioning."""
        if image_latent is not None:
            state = LatentState(
                latent=mx.zeros(shape, dtype=dtype),
                clean_latent=mx.zeros(shape, dtype=dtype),
                denoise_mask=mx.ones((1, 1, shape[2], 1, 1), dtype=dtype),
            )
            cond = VideoConditionByLatentIndex(
                latent=image_latent,
                frame_idx=image_frame_idx,
                strength=image_strength,
            )
            state = apply_conditioning(state, [cond])

            noise = mx.random.normal(shape, dtype=dtype)
            noise_scale = mx.array(initial_sigma, dtype=dtype)
            scaled_mask = state.denoise_mask * noise_scale
            state = LatentState(
                latent=noise * scaled_mask
                + state.latent * (mx.array(1.0, dtype=dtype) - scaled_mask),
                clean_latent=state.clean_latent,
                denoise_mask=state.denoise_mask,
            )
            latents = state.latent
            mx.eval(latents)
            return latents, state

        latents = mx.random.normal(shape, dtype=dtype)
        mx.eval(latents)
        return latents, None

    def _renoise_for_stage2(
        self,
        latents,
        image_latent,
        image_strength: float,
        image_frame_idx: int,
        initial_sigma: float,
        dtype,
    ):
        """Add stage-2 starting noise, applying I2V conditioning when present."""
        latent_frames = latents.shape[2]
        if image_latent is not None:
            state = LatentState(
                latent=latents,
                clean_latent=mx.zeros_like(latents),
                denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=dtype),
            )
            cond = VideoConditionByLatentIndex(
                latent=image_latent,
                frame_idx=image_frame_idx,
                strength=image_strength,
            )
            state = apply_conditioning(state, [cond])

            noise = mx.random.normal(latents.shape).astype(dtype)
            noise_scale = mx.array(initial_sigma, dtype=dtype)
            scaled_mask = state.denoise_mask * noise_scale
            state = LatentState(
                latent=noise * scaled_mask
                + state.latent * (mx.array(1.0, dtype=dtype) - scaled_mask),
                clean_latent=state.clean_latent,
                denoise_mask=state.denoise_mask,
            )
            mx.eval(state.latent)
            return state.latent, state

        noise_scale = mx.array(initial_sigma, dtype=dtype)
        one_minus_scale = mx.array(1.0 - initial_sigma, dtype=dtype)
        noise = mx.random.normal(latents.shape).astype(dtype)
        latents = noise * noise_scale + latents * one_minus_scale
        mx.eval(latents)
        return latents, None

    def _build_tiling_config(self, tiling: str, height: int, width: int,
                             num_frames: int, verbose: bool) -> Optional[TilingConfig]:
        """Map the tiling string to a :class:`TilingConfig` (or None for off)."""
        if tiling == "none":
            return None
        if tiling == "auto":
            return TilingConfig.auto(height, width, num_frames)
        if tiling == "default":
            return TilingConfig.default()
        if tiling == "aggressive":
            return TilingConfig.aggressive()
        if tiling == "conservative":
            return TilingConfig.conservative()
        if tiling == "spatial":
            return TilingConfig.spatial_only()
        if tiling == "temporal":
            return TilingConfig.temporal_only()
        if verbose:
            console.print(
                f"[yellow]  Unknown tiling mode '{tiling}', using auto[/]"
            )
        return TilingConfig.auto(height, width, num_frames)

    def _decode_video(
        self,
        vae_decoder,
        latents,
        tiling_config: Optional[TilingConfig],
        tiling_mode: str,
        stream: bool,
        on_frames_ready: Optional[FramesCallback],
        on_progress: Optional[ProgressCallback],
        num_frames: int,
        verbose: bool,
    ) -> np.ndarray:
        """Decode video latents to ``np.uint8`` frames, with optional streaming."""
        # Build a tile-decoder callback that converts MLX -> numpy and forwards
        # to the user's on_frames_ready hook (if streaming).
        stream_progress = None
        stream_task = None
        if stream and tiling_config is not None and on_frames_ready is not None:
            stream_progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
                disable=not verbose,
            )
            stream_progress.start()
            stream_task = stream_progress.add_task(
                "[cyan]Streaming frames[/]", total=num_frames
            )

            def _on_chunk_ready(frames: mx.array, start_idx: int):
                f = mx.squeeze(frames, axis=0)
                f = mx.transpose(f, (1, 2, 3, 0))
                f = mx.clip((f + 1.0) / 2.0, 0.0, 1.0)
                f = (f * 255).astype(mx.uint8)
                np_chunk = np.array(f)
                on_frames_ready(np_chunk, start_idx)
                stream_progress.advance(stream_task, advance=np_chunk.shape[0])

            tile_callback = _on_chunk_ready
        else:
            tile_callback = None

        if tiling_config is not None:
            spatial_info = (
                f"{tiling_config.spatial_config.tile_size_in_pixels}px"
                if tiling_config.spatial_config
                else "none"
            )
            temporal_info = (
                f"{tiling_config.temporal_config.tile_size_in_frames}f"
                if tiling_config.temporal_config
                else "none"
            )
            if verbose:
                console.print(
                    f"[dim]  Tiling ({tiling_mode}): "
                    f"spatial={spatial_info}, temporal={temporal_info}[/]"
                )
            video = vae_decoder.decode_tiled(
                latents,
                tiling_config=tiling_config,
                tiling_mode=tiling_mode,
                debug=False,
                on_frames_ready=tile_callback,
            )
        else:
            if verbose:
                console.print("[dim]  Tiling: disabled[/]")
            video = vae_decoder(latents)
        mx.eval(video)
        mx.clear_cache()

        if stream_progress is not None:
            stream_progress.stop()

        # Convert final tensor to numpy uint8 (T, H, W, 3)
        video = mx.squeeze(video, axis=0)
        video = mx.transpose(video, (1, 2, 3, 0))
        video = mx.clip((video + 1.0) / 2.0, 0.0, 1.0)
        video = (video * 255).astype(mx.uint8)

        if on_progress is not None:
            on_progress(1, 1, "vae_decode")

        return np.array(video)

    def _decode_audio(
        self,
        audio_latents,
        is_a2v: bool,
        a2v_waveform,
        a2v_sr,
        verbose: bool,
    ):
        """Decode the audio latents to a waveform, or pass-through for A2V."""
        if is_a2v and a2v_waveform is not None:
            audio_np = a2v_waveform
            if audio_np.ndim == 1:
                audio_np = audio_np[np.newaxis, :]
            sr = a2v_sr or AUDIO_LATENT_SAMPLE_RATE
            if verbose:
                console.print("[green]✓[/] Using original input audio (A2V)")
            return audio_np, sr

        from .models.ltx.audio_vae import AudioDecoder
        from .models.ltx.audio_vae.vocoder import (
            load_vocoder as _load_vocoder,
        )

        with self._console_status("[blue]Decoding audio...[/]"):
            audio_decoder = AudioDecoder.from_pretrained(
                self.model_path / "audio_vae" / "decoder"
            )
            vocoder = _load_vocoder(self.model_path / "vocoder")
            mx.eval(audio_decoder.parameters(), vocoder.parameters())

            mel_spectrogram = audio_decoder(audio_latents)
            mx.eval(mel_spectrogram)

            audio_waveform = vocoder(mel_spectrogram)
            mx.eval(audio_waveform)

            audio_np = np.array(audio_waveform.astype(mx.float32))
            if audio_np.ndim == 3:
                audio_np = audio_np[0]

            sr = getattr(vocoder, "output_sampling_rate", AUDIO_SAMPLE_RATE)

            del audio_decoder, vocoder
            mx.clear_cache()
        if verbose:
            console.print("[green]✓[/] Audio decoded")
        return audio_np, sr

    # ------------------------------------------------------------------
    # Logging helpers (centralized so we can tune verbose handling)
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            console.print(msg)

    def _log_panel(self, msg: str) -> None:
        if self.verbose:
            console.print(Panel(msg, expand=False))

    def _console_status(self, msg: str):
        """Return a context manager: rich status when verbose, no-op otherwise."""
        if self.verbose:
            return console.status(msg, spinner="dots")
        return _NullContext()


class _NullContext:
    """Trivial no-op context manager used to silence rich status output."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
