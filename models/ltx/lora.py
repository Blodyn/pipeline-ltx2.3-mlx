"""Chargement et fusion de LoRA pour le transformeur LTX-2.3.

Ce module est autonome : il ne dépend plus du paquet `mlx_video.lora` (supprimé)
qui ne ciblait que les modèles Wan2.x.
"""

from pathlib import Path

import mlx.core as mx
from rich.console import Console

from .ltx import LTXModel
from .utils import get_model_path

console = Console()


# Correspondance entre les clés brutes PyTorch d'un LoRA et les noms assainis utilisés
# par les modules MLX. Les substitutions s'appliquent quand la clé LoRA se termine par
# le suffixe ou quand un segment de chemin la suit.
_LORA_KEY_REPLACEMENTS = [
    (".to_out.0", ".to_out"),
    (".ff.net.0.proj", ".ff.proj_in"),
    (".ff.net.2", ".ff.proj_out"),
    (".audio_ff.net.0.proj", ".audio_ff.proj_in"),
    (".audio_ff.net.2", ".audio_ff.proj_out"),
    (".linear_1", ".linear1"),
    (".linear_2", ".linear2"),
]


def _resolve_lora_path(lora_path: str) -> Path:
    """Résout une référence LoRA (fichier / dossier / dépôt HF) vers un chemin de fichier.

    Args:
        lora_path : un fichier local (.safetensors), un dossier local en contenant un,
            ou un identifiant de dépôt HuggingFace.

    Returns:
        Chemin absolu du fichier .safetensors résolu.
    """
    p = Path(lora_path)

    if p.is_file():
        return p

    if p.is_dir():
        candidates = sorted(p.glob("*.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"Aucun fichier .safetensors trouvé dans {lora_path}")
        # On préfère les fichiers distilled-lora aux poids complets quand les deux existent
        lora_candidates = [c for c in candidates if "distilled-lora" in c.name]
        chosen = lora_candidates[0] if lora_candidates else candidates[0]
        console.print(f"[dim]Fichier LoRA utilisé : {chosen.name}[/]")
        return chosen

    # Traité comme un identifiant de dépôt HuggingFace
    lora_dir = get_model_path(lora_path)
    candidates = sorted(lora_dir.glob("*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"Aucun fichier .safetensors trouvé dans {lora_dir}")
    lora_candidates = [c for c in candidates if "distilled-lora" in c.name]
    chosen = lora_candidates[0] if lora_candidates else candidates[0]
    console.print(f"[dim]LoRA utilisé depuis le dépôt : {lora_path} ({chosen.name})[/]")
    return chosen


def load_and_merge_lora(
    model: LTXModel,
    lora_path: str,
    strength: float = 1.0,
    verbose: bool = True,
) -> int:
    """Charge des poids LoRA et les fusionne dans le transformeur en place.

    Deux formats sur disque sont pris en charge :

    - PyTorch brut : clés comme ``diffusion_model.{module}.lora_A.weight``
      (nécessite un assainissement des clés vers la convention MLX).
    - MLX pré-converti : clés comme ``{module}.lora_A.weight`` déjà assainies.

    La formule de fusion est ::

        weight += (lora_B * strength) @ lora_A

    Args:
        model : le transformeur :class:`LTXModel` dans lequel fusionner.
        lora_path : chemin / dossier / id de dépôt HF ; voir :func:`_resolve_lora_path`.
        strength : multiplicateur scalaire par LoRA appliqué au delta de poids.
        verbose : indique s'il faut afficher la progression via rich.

    Returns:
        Nombre de paires LoRA fusionnées.
    """
    lora_file = _resolve_lora_path(lora_path)

    # Chargement des poids LoRA
    lora_weights = mx.load(str(lora_file))

    # Détection du format : le PyTorch brut comporte le préfixe 'diffusion_model.'
    has_prefix = any(k.startswith("diffusion_model.") for k in lora_weights)

    # Regroupement en paires A/B par nom de module
    lora_pairs = {}
    for key in lora_weights:
        module_key = key
        if has_prefix:
            if not key.startswith("diffusion_model."):
                continue
            module_key = key.replace("diffusion_model.", "")

        if module_key.endswith(".lora_A.weight"):
            base_key = module_key.replace(".lora_A.weight", "")
            lora_pairs.setdefault(base_key, {})["A"] = lora_weights[key]
        elif module_key.endswith(".lora_B.weight"):
            base_key = module_key.replace(".lora_B.weight", "")
            lora_pairs.setdefault(base_key, {})["B"] = lora_weights[key]

    # Assainissement des clés uniquement pour le format PyTorch brut
    if has_prefix:
        sanitized_pairs = {}
        for key, pair in lora_pairs.items():
            new_key = key
            for old, new in _LORA_KEY_REPLACEMENTS:
                if new_key.endswith(old):
                    new_key = new_key[: -len(old)] + new
                else:
                    new_key = new_key.replace(old + ".", new + ".")
            sanitized_pairs[new_key] = pair
    else:
        sanitized_pairs = lora_pairs

    # Aplatissement du dict des paramètres du modèle (style chemin) pour des recherches rapides
    def flatten_params(params, prefix=""):
        flat = {}
        for k, v in params.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(flatten_params(v, full_key))
            else:
                flat[full_key] = v
        return flat

    flat_weights = flatten_params(dict(model.parameters()))

    # Fusion des deltas LoRA par lots pour éviter de doubler la mémoire d'un coup
    merged_count = 0
    batch = []
    batch_size = 100  # fusion par lots de 100 poids, puis eval pour libérer les intermédiaires

    for module_key, pair in sanitized_pairs.items():
        if "A" not in pair or "B" not in pair:
            continue

        weight_key = f"{module_key}.weight"
        if weight_key not in flat_weights:
            continue

        lora_a = pair["A"].astype(mx.float32)  # (rang, in_features)
        lora_b = pair["B"].astype(mx.float32)  # (out_features, rang)

        # delta = (lora_B * strength) @ lora_A
        delta = (lora_b * strength) @ lora_a

        base_weight = flat_weights.pop(weight_key)
        merged_weight = (base_weight.astype(mx.float32) + delta).astype(
            base_weight.dtype
        )
        batch.append((weight_key, merged_weight))
        del base_weight
        merged_count += 1

        if len(batch) >= batch_size:
            model.load_weights(batch, strict=False)
            mx.eval(model.parameters())
            batch.clear()

    if batch:
        model.load_weights(batch, strict=False)
        mx.eval(model.parameters())
        batch.clear()

    del flat_weights, lora_weights
    mx.clear_cache()

    if verbose:
        console.print(
            f"[green]✓[/] {merged_count} paires LoRA fusionnées (strength={strength})"
        )
    return merged_count
