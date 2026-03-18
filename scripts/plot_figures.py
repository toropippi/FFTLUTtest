import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-style figures for the FFT twiddle comparison experiment.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Experiment output directory containing summary.csv.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for generated figure PNGs. Defaults to <input-dir>/figures.")
    parser.add_argument(
        "--figure-c-case",
        type=str,
        default="",
        help="Optional case directory name for Figure C, for example case_checkerboard_0_256x256.",
    )
    return parser.parse_args()


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def representative_rows(summary: pd.DataFrame) -> pd.DataFrame:
    gpu_fast = summary[summary["mode"] == "gpu_fast"].copy()
    gpu_fast["variant_id"] = gpu_fast["variant_id"].astype(int)
    gpu_fast["width"] = gpu_fast["width"].astype(int)
    gpu_fast["height"] = gpu_fast["height"].astype(int)
    gpu_fast["area"] = gpu_fast["width"] * gpu_fast["height"]
    gpu_fast = gpu_fast.sort_values(["image_type", "area", "variant_id"])
    return gpu_fast.groupby("image_type", as_index=False).first()


def plot_figure_a(summary: pd.DataFrame, output_dir: Path) -> None:
    reps = representative_rows(summary)
    columns = [
        ("original.png", "Original"),
        ("recon_gpu_lut.png", "GPU LUT"),
        ("recon_gpu_fast.png", "GPU Fast"),
        ("absdiff_fast_vs_lut_log.png", "AbsDiff Fast vs LUT (log)"),
    ]

    fig, axes = plt.subplots(len(reps), len(columns), figsize=(12, max(3, 2.4 * len(reps))))
    if len(reps) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, (_, row) in enumerate(reps.iterrows()):
        case_dir = Path(row["case_dir"])
        for col_idx, (filename, title) in enumerate(columns):
            ax = axes[row_idx, col_idx]
            ax.imshow(load_image(case_dir / filename), cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(title)
            if col_idx == 0:
                ax.set_ylabel(row["image_type"], rotation=0, labelpad=60, va="center")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_a_comparison_grid.png", dpi=180)
    plt.close(fig)


def plot_figure_b(summary: pd.DataFrame, output_dir: Path) -> None:
    reps = representative_rows(summary)
    lut = summary[summary["mode"] == "gpu_lut"].copy()
    fast = summary[summary["mode"] == "gpu_fast"].copy()
    key_cols = ["image_type", "variant_id", "width", "height"]
    reps_key = reps[key_cols]
    lut = reps_key.merge(lut, on=key_cols, how="left")
    fast = reps_key.merge(fast, on=key_cols, how="left")

    metrics = [
        ("max_abs_error", "Max Abs Error"),
        ("rmse_vs_ref", "RMSE vs Ref"),
        ("psnr_vs_ref", "PSNR vs Ref"),
    ]
    labels = lut["image_type"].tolist()
    x = np.arange(len(labels))
    width = 0.38

    fig, axes = plt.subplots(len(metrics), 1, figsize=(16, 11), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (column, title) in zip(axes, metrics):
        lut_values = pd.to_numeric(lut[column], errors="coerce").fillna(0.0).to_numpy()
        fast_values = pd.to_numeric(fast[column], errors="coerce").fillna(0.0).to_numpy()
        ax.bar(x - width / 2, lut_values, width=width, label="GPU LUT")
        ax.bar(x + width / 2, fast_values, width=width, label="GPU Fast")
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper right")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_b_metrics.png", dpi=180)
    plt.close(fig)


def plot_figure_c(summary: pd.DataFrame, output_dir: Path, override_case: str) -> None:
    gpu_fast = summary[summary["mode"] == "gpu_fast"].copy()
    gpu_fast["max_abs_error_in_spectrum"] = pd.to_numeric(gpu_fast["max_abs_error_in_spectrum"], errors="coerce")
    if override_case:
        match = gpu_fast[gpu_fast["case_dir"].map(lambda p: Path(p).name == override_case)]
        if match.empty:
            raise ValueError(f"Figure C case not found: {override_case}")
        row = match.iloc[0]
    else:
        row = gpu_fast.sort_values("max_abs_error_in_spectrum", ascending=False).iloc[0]

    case_dir = Path(row["case_dir"])
    image = load_image(case_dir / "spectrum_absdiff_fast_vs_ref_log.png")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Figure C: {row['image_type']} v{row['variant_id']} {row['width']}x{row['height']}")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_c_spectrum_example.png", dpi=180)
    plt.close(fig)


def plot_figure_d(summary: pd.DataFrame, output_dir: Path) -> None:
    targets = [
        "horizontal_gradient",
        "sharp_edge_vertical",
        "text_small",
        "game_like_scene_simple",
    ]
    reps = representative_rows(summary).set_index("image_type")
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for ax, image_type in zip(axes.flat, targets):
        row = reps.loc[image_type]
        case_dir = Path(row["case_dir"])
        ax.imshow(load_image(case_dir / "absdiff_fast_vs_lut_log.png"), cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(image_type)

    fig.tight_layout()
    fig.savefig(output_dir / "figure_d_error_heatmaps.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_path = args.input_dir / "summary.csv"
    summary = pd.read_csv(summary_path)
    output_dir = args.output_dir or (args.input_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_figure_a(summary, output_dir)
    plot_figure_b(summary, output_dir)
    plot_figure_c(summary, output_dir, args.figure_c_case)
    plot_figure_d(summary, output_dir)

    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
