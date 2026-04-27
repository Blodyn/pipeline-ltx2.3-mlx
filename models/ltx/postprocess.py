
import numpy as np


def bilateral_filter(
    image: np.ndarray, d: int = 5, sigma_color: float = 75, sigma_space: float = 75
) -> np.ndarray:
    """Applique un filtre bilatéral pour réduire les artefacts de grille tout en préservant les bords.

    Args:
        image : image d'entrée sous forme de tableau numpy uint8 (H, W, C)
        d : diamètre du voisinage de chaque pixel
        sigma_color : sigma du filtre dans l'espace couleur
        sigma_space : sigma du filtre dans l'espace des coordonnées

    Returns:
        Image filtrée
    """
    try:
        import cv2

        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    except ImportError:
        # Repli sur un simple flou gaussien si cv2 n'est pas disponible
        return gaussian_blur(image, kernel_size=3)


def gaussian_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Applique un flou gaussien.

    Args:
        image : image d'entrée sous forme de tableau numpy uint8 (H, W, C)
        kernel_size : taille du noyau gaussien (doit être impaire)

    Returns:
        Image floutée
    """
    try:
        import cv2

        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    except ImportError:
        # Repli sur un simple box blur
        from scipy.ndimage import uniform_filter

        return uniform_filter(image, size=(kernel_size, kernel_size, 1)).astype(
            np.uint8
        )


def unsharp_mask(
    image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0, amount: float = 1.0
) -> np.ndarray:
    """Applique un masquage flou (unsharp mask) pour rehausser les bords après un flou.

    Args:
        image : image d'entrée sous forme de tableau numpy uint8
        kernel_size : taille du noyau gaussien
        sigma : sigma gaussien
        amount : intensité du rehaussement

    Returns:
        Image rehaussée
    """
    try:
        import cv2

        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    except ImportError:
        return image


def reduce_grid_artifacts(
    video: np.ndarray,
    method: str = "bilateral",
    strength: float = 1.0,
) -> np.ndarray:
    """Réduit les artefacts de grille dans les images d'une vidéo.

    Args:
        video : vidéo sous forme de tableau numpy (F, H, W, C) uint8
        method : "bilateral", "gaussian" ou "frequency"
        strength : intensité d'application du filtre (0-1)

    Returns:
        Vidéo traitée
    """
    if method == "bilateral":
        d = max(3, int(5 * strength))
        sigma = 50 + 50 * strength
        processed = np.stack(
            [
                bilateral_filter(frame, d=d, sigma_color=sigma, sigma_space=sigma)
                for frame in video
            ]
        )
    elif method == "gaussian":
        kernel_size = max(3, int(3 + 4 * strength))
        if kernel_size % 2 == 0:
            kernel_size += 1
        processed = np.stack(
            [gaussian_blur(frame, kernel_size=kernel_size) for frame in video]
        )
    elif method == "frequency":
        processed = np.stack(
            [remove_grid_frequency(frame, grid_size=8) for frame in video]
        )
    else:
        raise ValueError(f"Méthode inconnue : {method}")

    # Rehaussement optionnel pour récupérer un peu de détail
    if strength < 1.0:
        # Mélange avec l'original en fonction de l'intensité
        alpha = strength
        processed = (alpha * processed + (1 - alpha) * video).astype(np.uint8)

    return processed


def remove_grid_frequency(frame: np.ndarray, grid_size: int = 8) -> np.ndarray:
    """Supprime les composantes fréquentielles correspondant à la grille via FFT.

    Args:
        frame : image d'entrée (H, W, C) uint8
        grid_size : périodicité attendue de la grille en pixels

    Returns:
        Image filtrée
    """
    result = np.zeros_like(frame)

    for c in range(frame.shape[2]):
        channel = frame[:, :, c].astype(np.float32)
        h, w = channel.shape

        # FFT
        fft = np.fft.fft2(channel)
        fft_shifted = np.fft.fftshift(fft)

        # Création d'un filtre coupe-bande aux fréquences de la grille
        cy, cx = h // 2, w // 2
        mask = np.ones((h, w), dtype=np.float32)

        # Atténuation des fréquences correspondant à la périodicité de la grille
        freq_y = h // grid_size
        freq_x = w // grid_size

        for fy in range(-2, 3):
            for fx in range(-2, 3):
                if fy == 0 and fx == 0:
                    continue
                y_pos = cy + fy * freq_y
                x_pos = cx + fx * freq_x
                if 0 <= y_pos < h and 0 <= x_pos < w:
                    # Atténuation gaussienne autour de la fréquence
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            yy, xx = y_pos + dy, x_pos + dx
                            if 0 <= yy < h and 0 <= xx < w:
                                dist = np.sqrt(dy**2 + dx**2)
                                mask[yy, xx] *= min(1.0, dist / 3.0)

        # Application du masque puis FFT inverse
        fft_filtered = fft_shifted * mask
        channel_filtered = np.fft.ifft2(np.fft.ifftshift(fft_filtered)).real

        result[:, :, c] = np.clip(channel_filtered, 0, 255).astype(np.uint8)

    return result
