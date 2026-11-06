"""Drawing helpers shared by the demos."""
from __future__ import annotations

import logging

import matplotlib
import numpy as np
import viser


def quiet_browser_preconnects() -> None:
    """Chrome opens speculative connections that viser's server logs as handshake errors."""
    logging.getLogger("websockets.server").setLevel(logging.CRITICAL)


def sample_colours(k: int) -> list[tuple[int, int, int]]:
    """K viridis colours. The dark end would vanish against a dim mesh, so skip it."""
    cmap = matplotlib.colormaps["viridis"]
    return [tuple(int(c * 255) for c in cmap(t)[:3]) for t in np.linspace(0.25, 1.0, k)]


def set_background(server: viser.ViserServer, colour) -> None:
    tile = np.tile(np.asarray(colour, dtype=np.uint8), (16, 16, 1))
    server.scene.set_background_image(image=tile, format="png")


def flatten_rgba(rgba: np.ndarray, background) -> np.ndarray:
    """Composite a PNG render onto the background: empty pixels come back transparent."""
    rgba = rgba.astype(np.float32)
    alpha = rgba[:, :, 3:] / 255.0
    return (rgba[:, :, :3] * alpha + np.asarray(background) * (1.0 - alpha)).astype(np.uint8)
