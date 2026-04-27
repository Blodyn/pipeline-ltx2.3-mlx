"""Utilitaires partagés pour le chargement du modèle LTX-2.3."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import mlx.core as mx
from huggingface_hub import snapshot_download


def get_model_path(
    path_or_hf_repo: str,
    revision: Optional[str] = None,
) -> Path:
    """Renvoie le chemin local d'un modèle, en le téléchargeant si nécessaire.

    Args:
        path_or_hf_repo : chemin local ou identifiant de dépôt HuggingFace
        revision : révision Git pour le dépôt HF

    Returns:
        Chemin du dossier modèle
    """
    model_path = Path(path_or_hf_repo)

    if model_path.exists():
        return model_path

    model_path = Path(
        snapshot_download(
            repo_id=path_or_hf_repo,
            revision=revision,
            allow_patterns=[
                "*.safetensors",
                "*.json",
                "config.json",
            ],
        )
    )

    return model_path


def load_safetensors(path: Path) -> Dict[str, mx.array]:
    """Charge les poids depuis un (ou plusieurs) fichier(s) safetensors via MLX.

    Args:
        path : chemin du dossier modèle ou d'un fichier safetensors unique

    Returns:
        Dictionnaire de poids
    """
    if path.is_file():
        return mx.load(str(path))

    weights = {}
    for sf_path in path.glob("*.safetensors"):
        weights.update(mx.load(str(sf_path)))
    return weights


def load_config(model_path: Path) -> Dict[str, Any]:
    """Charge la configuration du modèle depuis config.json.

    Args:
        model_path : chemin du dossier modèle

    Returns:
        Dictionnaire de configuration
    """
    config_path = model_path / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


def save_weights(path: Path, weights: Dict[str, mx.array]) -> None:
    """Enregistre des poids au format safetensors.

    Args:
        path : dossier de sortie
        weights : dictionnaire de poids
    """
    path.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path / "model.safetensors"), weights)


def convert_audio_encoder(
    model_path,
    source_repo: str = "Lightricks/LTX-2",
) -> Path:
    """Convertit et sauvegarde les poids de l'encodeur audio depuis le checkpoint HF d'origine.

    Certains dépôts LTX-2.3 pré-convertis ne fournissent que le décodeur audio + le vocoder
    (utilisés pour la génération audio/vidéo conjointe). Pour l'Audio-to-Video (A2V) on a
    aussi besoin de l'encodeur audio. Cet utilitaire extrait les poids de l'encodeur depuis
    le checkpoint Lightricks d'origine, transpose les Conv2d pour MLX et les sauvegarde
    localement pour AudioEncoder.from_pretrained().

    Args:
        model_path : dossier modèle local (emplacement de sortie).
        source_repo : dépôt HF contenant audio_vae/diffusion_pytorch_model.safetensors.

    Returns:
        Chemin du dossier audio_vae/encoder.
    """
    model_path = Path(model_path)
    encoder_dir = model_path / "audio_vae" / "encoder"

    if (encoder_dir / "model.safetensors").exists():
        return encoder_dir

    from huggingface_hub import hf_hub_download

    vae_path = hf_hub_download(
        source_repo,
        "audio_vae/diffusion_pytorch_model.safetensors",
    )

    raw_weights = mx.load(vae_path)

    from .audio_vae import AudioEncoder
    from .config import AudioEncoderModelConfig

    # Construction de la config à partir de celle du décodeur (même architecture du VAE audio)
    decoder_config_path = model_path / "audio_vae" / "decoder" / "config.json"
    if decoder_config_path.exists():
        with open(decoder_config_path) as f:
            dec_cfg = json.load(f)
        enc_config = {
            "ch": dec_cfg.get("ch", 128),
            "in_channels": dec_cfg.get("out_ch", 2),
            "ch_mult": dec_cfg.get("ch_mult", [1, 2, 4]),
            "num_res_blocks": dec_cfg.get("num_res_blocks", 2),
            "attn_resolutions": dec_cfg.get("attn_resolutions", []),
            "resolution": dec_cfg.get("resolution", 256),
            "z_channels": dec_cfg.get("z_channels", 8),
            "double_z": True,
            "n_fft": 1024,
            "norm_type": dec_cfg.get("norm_type", "pixel"),
            "causality_axis": dec_cfg.get("causality_axis", "height"),
            "dropout": dec_cfg.get("dropout", 0.0),
            "mid_block_add_attention": dec_cfg.get("mid_block_add_attention", False),
            "sample_rate": dec_cfg.get("sample_rate", 16000),
            "mel_hop_length": dec_cfg.get("mel_hop_length", 160),
            "is_causal": dec_cfg.get("is_causal", True),
            "mel_bins": dec_cfg.get("mel_bins", 64) or 64,
            "resamp_with_conv": dec_cfg.get("resamp_with_conv", True),
            "attn_type": dec_cfg.get("attn_type", "vanilla"),
        }
    else:
        enc_config = {"in_channels": 2, "double_z": True, "n_fft": 1024, "mel_bins": 64}

    config = AudioEncoderModelConfig.from_dict(enc_config)
    encoder = AudioEncoder(config)
    sanitized = encoder.sanitize(raw_weights)

    encoder_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(encoder_dir / "model.safetensors"), sanitized)
    with open(encoder_dir / "config.json", "w") as f:
        json.dump(enc_config, f, indent=2)

    return encoder_dir
