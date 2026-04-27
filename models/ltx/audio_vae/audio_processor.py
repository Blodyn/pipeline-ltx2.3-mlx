"""Utilitaires de traitement audio pour charger des fichiers audio et calculer des spectrogrammes mel.

Reproduit le comportement de l'AudioProcessor PyTorch de LTX-2 (torchaudio.transforms.MelSpectrogram)
en s'appuyant sur librosa pour la compatibilité macOS/MLX.
"""


import mlx.core as mx
import numpy as np


def load_audio(
    path: str,
    target_sr: int = 16000,
    start_time: float = 0.0,
    max_duration: float | None = None,
    mono: bool = False,
) -> tuple[np.ndarray, int]:
    """Charge un fichier audio et le rééchantillonne à la fréquence cible.

    Args:
        path : chemin du fichier audio (WAV, FLAC, MP3, OGG ou vidéo contenant une piste audio).
        target_sr : fréquence d'échantillonnage cible (16000 Hz par défaut).
        start_time : instant de départ en secondes.
        max_duration : durée maximale en secondes. None = lit jusqu'à la fin.
        mono : si True, convertit en mono. Par défaut False (préserve les canaux).

    Returns:
        (waveform, sample_rate) où waveform est un tableau numpy float32 de forme (canaux, échantillons).
    """
    import librosa

    # librosa.load renvoie du mono par défaut ; on veut préserver le stéréo
    y, sr = librosa.load(
        path,
        sr=target_sr,
        mono=mono,
        offset=start_time,
        duration=max_duration,
    )

    # On garantit un tableau 2D : (canaux, échantillons)
    if y.ndim == 1:
        y = y[np.newaxis, :]  # (1, échantillons)

    return y.astype(np.float32), sr


def ensure_stereo(waveform: np.ndarray) -> np.ndarray:
    """S'assure que la forme d'onde est stéréo (2, échantillons). Duplique si mono."""
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    if waveform.shape[0] == 1:
        waveform = np.concatenate([waveform, waveform], axis=0)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    return waveform


def waveform_to_mel(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 160,
    win_length: int = 1024,
    n_mels: int = 64,
    fmin: float = 0.0,
    fmax: float = 8000.0,
) -> mx.array:
    """Convertit une forme d'onde en spectrogramme log-mel équivalent à PyTorch MelSpectrogram.

    Référence PyTorch :
        MelSpectrogram(sample_rate=16000, n_fft=1024, win_length=1024, hop_length=160,
                       f_min=0.0, f_max=8000.0, n_mels=64, power=1.0,
                       mel_scale="slaney", norm="slaney", center=True, pad_mode="reflect")

    Args:
        waveform : tableau numpy float32 de forme (canaux, échantillons).
        sample_rate : fréquence d'échantillonnage de la forme d'onde.
        n_fft : taille de la FFT.
        hop_length : pas (hop length).
        win_length : taille de la fenêtre.
        n_mels : nombre de bandes mel.
        fmin : fréquence minimale du banc de filtres mel.
        fmax : fréquence maximale du banc de filtres mel.

    Returns:
        Spectrogramme log-mel sous forme de mx.array de shape (1, canaux, temps, n_mels).
    """
    import librosa

    # On garantit un tableau 2D
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]

    channels = waveform.shape[0]
    mels = []

    for ch in range(channels):
        # Spectrogramme de magnitude (power=1.0)
        S = np.abs(
            librosa.stft(
                waveform[ch],
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                center=True,
                pad_mode="reflect",
            )
        )

        # Banc de filtres mel avec normalisation slaney
        mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            norm="slaney",
        )
        mel = mel_basis @ S

        # Échelle logarithmique
        mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))

        # Transposition : (n_mels, temps) -> (temps, n_mels)
        mel = mel.T
        mels.append(mel)

    # Empilement des canaux : (canaux, temps, n_mels)
    mel_spec = np.stack(mels, axis=0)

    # Ajout de la dimension batch : (1, canaux, temps, n_mels)
    mel_spec = mel_spec[np.newaxis, ...]

    return mx.array(mel_spec, dtype=mx.float32)
