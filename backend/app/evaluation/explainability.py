from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def image_to_base64(img: np.ndarray, cmap: str = "gray", vmin=0.0, vmax=1.0) -> str:
    fig, ax = plt.subplots(figsize=(2.2, 2.2), dpi=100)
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


def perturbation_heatmap_base64(original: np.ndarray, adversarial: np.ndarray) -> str:
    diff = np.abs(adversarial - original)
    fig, ax = plt.subplots(figsize=(2.2, 2.2), dpi=100)
    im = ax.imshow(diff, cmap="inferno", vmin=0.0, vmax=max(diff.max(), 1e-6))
    ax.axis("off")
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


def tabular_feature_deltas(original: np.ndarray, adversarial: np.ndarray, feature_names: list[str]) -> list[dict]:
    deltas = adversarial - original
    rows = []
    for name, orig_val, delta in zip(feature_names, original, deltas):
        rows.append(
            {
                "feature": name,
                "original_value": round(float(orig_val), 4),
                "change": round(float(delta), 4),
                "significant": bool(abs(delta) > 0.05),
            }
        )
    # sort by magnitude of change, largest first, for a more useful readout
    rows.sort(key=lambda r: abs(r["change"]), reverse=True)
    return rows
