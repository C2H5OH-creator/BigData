import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

X_PATH = Path("/home/c2h5oh/dev/projects/BigData/dataset/X_test_sat4_R_G_B.csv")
Y_PATH = Path("/home/c2h5oh/dev/projects/BigData/dataset/y_test_sat4.csv")
OUTPUT_PATH = Path("/home/c2h5oh/dev/projects/BigData/sat4_sample.png")

CLASS_NAMES = ["barren_land", "trees", "grassland", "none"]
IDX = 4


def save_or_show(fig: plt.Figure) -> None:
    backend = plt.get_backend().lower()
    if "agg" not in backend and (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        plt.show()
    else:
        fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
        print(f"Saved preview to: {OUTPUT_PATH} (backend={backend})")


def main() -> None:
    X = pd.read_csv(X_PATH, header=None)
    y = pd.read_csv(Y_PATH, header=None)

    row = X.iloc[IDX].to_numpy(dtype=np.uint8)
    label = CLASS_NAMES[y.iloc[IDX].to_numpy().argmax()]
    n_features = row.size

    if n_features == 784:
        img = row.reshape(28, 28)
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{X_PATH.stem}: {label}")
        ax.axis("off")
    elif n_features == 2352:
        img = row.reshape(28, 28, 3)
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.imshow(img)
        ax.set_title(f"{X_PATH.stem}: {label}")
        ax.axis("off")
    elif n_features == 3136:
        img = row.reshape(28, 28, 4)
        rgb = img[:, :, :3]
        nir = img[:, :, 3]

        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax[0].imshow(rgb)
        ax[0].set_title(f"RGB: {label}")
        ax[0].axis("off")

        ax[1].imshow(nir, cmap="gray")
        ax[1].set_title("NIR")
        ax[1].axis("off")
    else:
        raise ValueError(
            f"Unsupported row width: {n_features}. Expected 784, 2352, or 3136."
        )

    plt.tight_layout()
    save_or_show(fig)


if __name__ == "__main__":
    main()
