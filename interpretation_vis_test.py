import os
import json, re
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

"""
Utility to visualize model attention at high level and compare 
to caption. Produces maps of average attention and 95th percentile attention
next to caption as well as a panel with several timesteps. 

This is only for human qualitative evalutation. This is NOT the data passed
to the model for synthetic caption generation. 
"""

# --------------------------------------------------------
# CONFIG
# --------------------------------------------------------

IMAGE_ROOT = "train2014/train2014"  # original COCO images
ATTN_ROOT = "output_train2014"  # <image_id>/alphas.pt
INTERP_ROOT = "llm_interpret_test_4"  # <coco_image_name>/interpretation.json
OUT_ROOT = "interpretation_vis_test_4"  # where output images will be saved

N_TIMESTEPS_TO_SHOW = 6  # number of sampled timesteps


# --------------------------------------------------------
# UTILITIES
# --------------------------------------------------------


def load_image_and_attn(coco_name):
    """
    coco_name = 'COCO_train2014_000000506766.jpg'
    """
    # image_id = coco_name.split("_")[-1].replace(".jpg", "")

    img_path = os.path.join(IMAGE_ROOT, coco_name)
    img = Image.open(img_path).convert("RGB")

    attn_path = os.path.join(ATTN_ROOT, coco_name, "alphas.pt")
    alphas = torch.load(attn_path)  # shape (T, H, W)
    return img, alphas


def load_interpretation(coco_name):
    json_path = Path(INTERP_ROOT) / coco_name / "interpretation.json"
    raw = json_path.read_text()

    # remove code fences
    cleaned = re.sub(r"```(?:json)?", "", raw)
    cleaned = re.sub(r"```", "", cleaned)

    data = json.loads(cleaned)
    return data["interpretation"]


def attention_overlay_on_image(img, attn_2d, alpha=0.45):
    """Return an attention heatmap overlaid on the image as a numpy array."""
    attn = attn_2d / attn_2d.max()
    heat = Image.fromarray((attn * 255).astype(np.uint8)).resize(
        img.size, Image.BILINEAR
    )
    return heat


# --------------------------------------------------------
# MAIN PANEL CREATION
# --------------------------------------------------------


def save_main_panel(coco_name, img, alphas, interp, out_dir):
    alphas_valid = alphas[1:, ...]
    avg_attn = alphas_valid.mean(dim=0).numpy()
    perc95 = np.percentile(alphas_valid, 95, axis=0)

    avg_heat = attention_overlay_on_image(img, avg_attn)
    perc95_heat = attention_overlay_on_image(
        img, perc95
    )  # Highlight pixels with often-strong attention

    # Create panel
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[4, 4, 2])

    # Original image
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img)
    ax_img.set_title("Original")
    ax_img.axis("off")

    # Average attention
    ax_avg = fig.add_subplot(gs[1, 0])
    ax_avg.imshow(img)
    ax_avg.imshow(avg_heat, cmap="jet", alpha=0.45)
    ax_avg.set_title("Average Attention")
    ax_avg.axis("off")

    # Max attention
    ax_max = fig.add_subplot(gs[2, 0])
    ax_max.imshow(img)
    ax_max.imshow(perc95_heat, cmap="jet", alpha=0.45)
    ax_max.set_title("95th Percentile Attention")
    ax_max.axis("off")

    # Interpretation text
    ax_text = fig.add_subplot(gs[:, 1:])
    ax_text.axis("off")
    ax_text.set_title("Generated Interpretation", fontsize=14, pad=10)
    ax_text.text(0.01, 0.98, interp, ha="left", va="top", wrap=True, fontsize=12)

    out_path = os.path.join(out_dir, "panel_main.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def save_timestep_panel(coco_name, img, alphas, out_dir):
    T = alphas.shape[0]
    timesteps = np.linspace(1, T - 1, N_TIMESTEPS_TO_SHOW).astype(int)

    fig = plt.figure(figsize=(16, 3))
    cols = len(timesteps)

    for i, t in enumerate(timesteps):
        ax = fig.add_subplot(1, cols, i + 1)
        att = alphas[t].numpy()
        heat = attention_overlay_on_image(img, att)

        ax.imshow(img)
        ax.imshow(heat, cmap="jet", alpha=0.45)
        ax.set_title(f"t={t}")
        ax.axis("off")

    out_path = os.path.join(out_dir, "timesteps.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# --------------------------------------------------------
# FULL IMAGE RENDER
# --------------------------------------------------------


def process_image_folder(coco_name):
    print(f"Processing {coco_name}...")

    out_dir = os.path.join(OUT_ROOT, coco_name)
    os.makedirs(out_dir, exist_ok=True)

    img, alphas = load_image_and_attn(coco_name)
    interp = load_interpretation(coco_name)

    # Save the combined panel
    save_main_panel(coco_name, img, alphas, interp, out_dir)

    # Save timestep highlights
    save_timestep_panel(coco_name, img, alphas, out_dir)

    print(f"Saved results to: {out_dir}")


# --------------------------------------------------------
# SCRIPT ENTRY POINT
# --------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(OUT_ROOT, exist_ok=True)

    # Generate visualizations for all images in folder
    coco_folders = [
        d
        for d in os.listdir(INTERP_ROOT)
        if d.startswith("COCO_train2014_")
        and os.path.isdir(os.path.join(INTERP_ROOT, d))
    ]

    for coco_name in coco_folders:
        process_image_folder(coco_name)
