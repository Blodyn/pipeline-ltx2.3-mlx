# ltx2_3

**Génération vidéo (et audio) LTX-2.3 accélérée par MLX pour Apple Silicon.**

`ltx2_3` est un paquet Python qui exécute le pipeline de diffusion distillée
LTX-2.3 nativement sur les GPU Apple Silicon via [MLX](https://github.com/ml-explore/mlx).
Il expose un unique point d'entrée haut niveau — `LTXPipeline` — qui prend en
charge la génération texte-vers-vidéo, image-vers-vidéo, audio-vers-vidéo, ainsi
que la synthèse audio + vidéo conjointe.

> ⚠️ **Statut :** travail en cours. Le code du paquet s'exécute de bout en bout
> sur Apple Silicon dès lors que les poids du modèle sont disponibles localement,
> mais l'API peut encore évoluer entre versions mineures.

---

## Fonctionnalités

- **Texte-vers-Vidéo (T2V)** — génère une vidéo à partir d'un prompt.
- **Image-vers-Vidéo (I2V)** — anime une image fixe à partir d'un prompt textuel.
- **Audio-vers-Vidéo (A2V)** — génère une vidéo synchronisée à une piste audio existante.
- **Audio + vidéo conjoints** — synthétise les deux modalités en un seul passage.
- **Fusion de LoRA** — charge à la volée des LoRA de contrôle de caméra ou stylistiques.
- **Sur-échantillonneur spatial** — sur-échantillonneur latent ×2 optionnel entre les deux étapes de distillation.
- **Tuilage du VAE** — tuilage spatial / temporel automatique pour les sorties dépassant la VRAM disponible.
- **Callbacks de streaming** — réception progressive des frames décodées pour les UI de prévisualisation.

Le paquet renvoie des tableaux NumPy bruts. La sauvegarde sur disque, l'encodage
vidéo ou le multiplexage audio sont laissés à l'application appelante (cela
permet au paquet de rester libre de toute dépendance ffmpeg / codec).

---

## Prérequis

- macOS sur Apple Silicon (puces série M).
- Python 3.10 ou plus récent.
- Les paquets Python suivants :

  | Paquet            | Rôle                                          |
  | ----------------- | --------------------------------------------- |
  | `mlx`             | Moteur de tenseurs                            |
  | `mlx-vlm`         | Réutilise les poids Gemma 3 pour l'encodage texte |
  | `numpy`           | Échange des tableaux de frames / audio        |
  | `pillow`          | Chargement des images pour l'I2V              |
  | `rich`            | Barres de progression et panneaux             |
  | `huggingface_hub` | Téléchargement des poids                      |
  | `safetensors`     | Chargement des poids (transitive)             |

Un checkpoint de modèle compatible LTX-2.3 (par défaut :
`prince-canuma/LTX-2.3-distilled`) ainsi qu'un encodeur de texte Gemma 3
(par défaut : `google/gemma-3-12b-it`).

---

## Démarrage rapide

```python
from ltx2_3 import LTXPipeline

pipeline = LTXPipeline(
    model_repo="prince-canuma/LTX-2.3-distilled",
    text_encoder_repo="google/gemma-3-12b-it",
)
pipeline.load()

result = pipeline.generate(
    prompt="Une scène océanique cinématique au coucher du soleil, prise de drone lente",
    height=512,
    width=512,
    num_frames=33,   # doit satisfaire num_frames = 1 + 8*k
    fps=24,
    seed=42,
)

frames = result.frames          # np.uint8, forme (T, H, W, 3), RGB 0..255
audio  = result.audio           # None pour un T2V muet
print(f"Généré en {result.elapsed_seconds:.1f} s, "
      f"pic mémoire : {result.peak_memory_gb:.2f} Go")

pipeline.unload()
```

### Image-vers-Vidéo

```python
result = pipeline.generate(
    prompt="Le chat tourne lentement la tête et cligne des yeux",
    image="chemin/vers/chat.png",
    image_strength=1.0,    # 0..1, force avec laquelle la première frame ancre la génération
    num_frames=49,
)
```

### Synthèse audio (conjointe)

```python
result = pipeline.generate(
    prompt="Pluie sur un toit en tôle, tonnerre lointain",
    audio=True,             # synthèse audio conjointe
    num_frames=33,
)
# result.audio          -> np.float32, forme (canaux, échantillons), plage [-1, 1]
# result.audio_sample_rate -> 24000
```

### Audio-vers-Vidéo (A2V)

```python
result = pipeline.generate(
    prompt="Un danseur suivant le rythme",
    audio_file="chemin/vers/piste.wav",
    audio_start_time=0.0,
    num_frames=33,
)
```

### Prévisualisation en streaming

```python
def on_frames_ready(frames_uint8, start_index):
    # Pousse les frames vers un visualiseur / WebSocket / encodeur…
    print(f"Reçu {len(frames_uint8)} frames à partir de l'index {start_index}")

def on_progress(step, total, stage):
    print(f"[{stage}] {step}/{total}")

result = pipeline.generate(
    prompt="Une rue baignée de néons la nuit",
    stream=True,
    on_frames_ready=on_frames_ready,
    on_progress=on_progress,
)
```

### Fusion de LoRA

```python
result = pipeline.generate(
    prompt="plan en orbite autour d'une sculpture science-fiction",
    lora_path="chemin/vers/camera_control.safetensors",
    lora_strength=1.0,
)
```

---

## Surface d'API

```python
from ltx2_3 import (
    LTXPipeline,        # la classe pipeline
    GenerationResult,   # dataclass renvoyée par .generate(...)
    ProgressCallback,   # alias de type : Callable[[int, int, str], None]
    FramesCallback,     # alias de type : Callable[[np.ndarray, int], None]
)
```

### `LTXPipeline.__init__`

| Argument            | Type            | Valeur par défaut                    |
| ------------------- | --------------- | ------------------------------------ |
| `model_repo`        | `str`           | `"prince-canuma/LTX-2.3-distilled"`  |
| `text_encoder_repo` | `Optional[str]` | `"google/gemma-3-12b-it"`            |
| `verbose`           | `bool`          | `True`                               |

### `LTXPipeline.generate` — paramètres principaux

| Argument            | Type                       | Défaut  | Notes                                            |
| ------------------- | -------------------------- | ------- | ------------------------------------------------ |
| `prompt`            | `str`                      | —       | requis                                           |
| `height`, `width`   | `int`                      | `512`   | doivent être divisibles par 64                   |
| `num_frames`        | `int`                      | `33`    | doit valoir `1 + 8*k`                            |
| `fps`               | `int`                      | `24`    |                                                  |
| `image`             | `str \| Path`              | `None`  | active l'I2V                                     |
| `image_strength`    | `float`                    | `1.0`   | force du conditionnement I2V dans `[0, 1]`       |
| `audio`             | `bool`                     | `False` | synthétise l'audio conjointement                 |
| `audio_file`        | `str \| Path`              | `None`  | active l'A2V                                     |
| `seed`              | `int`                      | `42`    |                                                  |
| `lora_path`         | `str`                      | `None`  | fusionné à la volée                              |
| `lora_strength`     | `float`                    | `1.0`   |                                                  |
| `spatial_upscaler`  | `str`                      | `None`  | nom de fichier dans `model_repo`                 |
| `tiling`            | `str`                      | `"auto"`| `"none"`, `"default"`, `"aggressive"`, …         |
| `on_progress`       | `ProgressCallback`         | `None`  |                                                  |
| `on_frames_ready`   | `FramesCallback`           | `None`  |                                                  |
| `stream`            | `bool`                     | `False` | émission progressive des frames                  |

`generate()` renvoie un `GenerationResult` :

```python
@dataclass
class GenerationResult:
    frames: np.ndarray                 # (T, H, W, 3) uint8
    audio: Optional[np.ndarray] = None # (canaux, échantillons) float32
    audio_sample_rate: int = 0
    fps: int = 24
    elapsed_seconds: float = 0.0
    peak_memory_gb: float = 0.0
    metadata: Dict[str, Any] = ...
```

---

## Arborescence du dépôt

```
ltx2_3/
├── __init__.py                    # ré-exports publics (LTXPipeline, …)
├── pipeline.py                    # API d'inférence haut niveau
├── py.typed                       # marqueur PEP 561
├── utils/
│   ├── __init__.py
│   └── common.py                  # rms_norm, chargement d'image, helpers de quantification, …
└── models/
    ├── __init__.py
    └── ltx/
        ├── __init__.py            # ré-exports LTXModel, LTXModelConfig, …
        ├── adaln.py               # AdaLayerNormSingle + embedding de timestep
        ├── attention.py           # scaled_dot_product_attention + Attention
        ├── config.py              # toutes les configs (dataclasses)
        ├── conditioning/          # helpers de conditionnement latent
        ├── denoise.py             # échantillonneur distillé en 2 étapes
        ├── feed_forward.py
        ├── lora.py                # fusion LoRA / réécriture des poids
        ├── ltx.py                 # LTXModel (wrapper transformer)
        ├── postprocess.py
        ├── rope.py                # embeddings de position rotatifs
        ├── text_encoder.py        # wrapper Gemma 3 (LTX2TextEncoder)
        ├── text_projection.py     # PixArtAlphaTextProjection
        ├── transformer.py         # blocs de base du transformer
        ├── upsampler.py           # sur-échantillonneur latent spatial
        ├── utils.py               # get_model_path, conversions de format
        ├── audio_vae/             # VAE mel + waveform + vocoder
        │   ├── audio_processor.py
        │   ├── audio_vae.py
        │   ├── attention.py
        │   ├── causal_conv_2d.py
        │   ├── downsample.py
        │   ├── normalization.py
        │   ├── ops.py
        │   ├── resnet.py
        │   ├── upsample.py
        │   └── vocoder.py
        └── video_vae/             # VAE 3-D avec tuilage
            ├── convolution.py
            ├── decoder.py
            ├── encoder.py
            ├── ops.py
            ├── resnet.py
            ├── sampling.py
            ├── tiling.py
            └── video_vae.py
```

---

## `pyproject.toml` suggéré

Si le dépôt n'en a pas encore, vous pouvez déposer celui-ci à la racine du projet :

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ltx2_3"
version = "0.2.0"
description = "Génération vidéo LTX-2.3 accélérée par MLX pour Apple Silicon."
readme = "README.md"
requires-python = ">=3.10"
license = {text = "Apache-2.0"}
dependencies = [
    "mlx>=0.18",
    "mlx-vlm>=0.1.0",
    "numpy>=1.24",
    "pillow>=10.0",
    "rich>=13.0",
    "huggingface_hub>=0.20",
    "safetensors>=0.4",
]

[tool.setuptools.packages.find]
include = ["ltx2_3*"]
```

---

## Contribuer

Les issues et les pull requests sont les bienvenues. Avant d'ouvrir une PR :

1. Assurez-vous que tous les imports sont **relatifs** à l'intérieur du paquet
   (pas d'imports `from ltx2_3.xxx` au sein du paquet lui-même ; utilisez
   `from .xxx` ou `from ..xxx`).
2. Lancez `python -m pyflakes ltx2_3` et nettoyez les nouveaux avertissements.
3. Lancez `python -c "import ltx2_3; print(ltx2_3.__version__)"` pour confirmer
   que le paquet s'importe toujours correctement.
