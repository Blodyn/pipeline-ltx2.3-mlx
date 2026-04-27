"""ltx2_3 : inférence LTX-2.3 accélérée par MLX pour Apple Silicon.

API publique ::

    from ltx2_3 import LTXPipeline, GenerationResult

    pipeline = LTXPipeline(model_repo="prince-canuma/LTX-2.3-distilled")
    pipeline.load()
    result = pipeline.generate(prompt="Une scène cinématique d'océan", num_frames=33)
    frames = result.frames                 # np.uint8 (T, H, W, 3)
    audio = result.audio                   # np.float32 (canaux, échantillons) ou None
    pipeline.unload()
"""

from .pipeline import (
    FramesCallback,
    GenerationResult,
    LTXPipeline,
    ProgressCallback,
)

__version__ = "0.2.0"

__all__ = [
    "LTXPipeline",
    "GenerationResult",
    "ProgressCallback",
    "FramesCallback",
    "__version__",
]
