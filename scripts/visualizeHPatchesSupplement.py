import os
import math
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================================================
# CONFIG
# ============================================================

# HPATCHES_ROOT = "/home/boat/proxyISP/pytorch-superpoint/datasets/HPatches_caches_DGC/eval_sl_v16.2-chroma-HumanTunedInitialHype_replicate-s21fe_sunlit_lr0.0005_schedulerPlateauTo0.00001_bs1_ga8_adjust_defaultcolorhuesat_120000_HpatchesV4.1"
HPATCHES_ROOT = "/home/boat/proxyISP/pytorch-superpoint/datasets/HPatches_caches_DGC/eval_ll_v16.2-chroma-HumanTunedInitialHype_replicate-s21fe_lowlight_lr0.0005_schedulerPlateauTo0.00001_bs1_ga8_adjust_defaultcolorhuesat_denoise_45000_HpatchesV4.1"
target_sequences = [
    "stone", "starrynight", "brick", "seanema", "professor",
    "reddoor", "cement", "taladcat", "nobag", "moss2",
    "melon", "thaicult"
]

SEQUENCES_PER_FIG = 12
MAX_IMAGES_PER_SEQ = None
SAVE_FIGS = True
OUT_DIR = "hpatches_grids"
FIG_DPI = 100

# ============================================================
# UTILS
# ============================================================

def list_sequences(root):
    return sorted(
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )


def filter_sequences(seqs, targets):
    if not targets:
        return seqs
    targets = [t.lower() for t in targets]
    return [
        s for s in seqs
        if any(t in os.path.basename(s).lower() for t in targets)
    ]


def load_sequence_images(seq_path, max_imgs=None):
    files = sorted(
        f for f in os.listdir(seq_path)
        if f.lower().endswith((".png", ".jpg", ".ppm"))
    )
    if max_imgs:
        files = files[:max_imgs]

    imgs = []
    for f in files:
        img = cv2.imread(os.path.join(seq_path, f))
        imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return imgs


# ============================================================
# MAIN
# ============================================================

def plot_hpatches_grids(root, seqs_per_fig, max_imgs_per_seq, targets):
    seqs = filter_sequences(list_sequences(root), targets)
    if not seqs:
        raise RuntimeError("No sequences matched filter")

    os.makedirs(OUT_DIR, exist_ok=True)
    n_figs = math.ceil(len(seqs) / seqs_per_fig)

    for fig_idx in range(n_figs):
        chunk = seqs[
            fig_idx * seqs_per_fig :
            (fig_idx + 1) * seqs_per_fig
        ]

        seq_imgs = []
        max_cols = 0
        for s in chunk:
            imgs = load_sequence_images(s, max_imgs_per_seq)
            seq_imgs.append(imgs)
            max_cols = max(max_cols, len(imgs))

        n_rows = len(seq_imgs)
        n_cols = max_cols

        fig = plt.figure(
            figsize=(n_cols * 2.2, n_rows * 2.2),
            dpi=FIG_DPI
        )

        gs = GridSpec(
            n_rows, n_cols,
            figure=fig,
            left=0, right=1, bottom=0, top=1,
            wspace=0, hspace=0
        )

        for r, imgs in enumerate(seq_imgs):
            for c in range(n_cols):
                ax = fig.add_subplot(gs[r, c])
                ax.set_axis_off()
                ax.set_aspect("auto")

                if c < len(imgs):
                    ax.imshow(imgs[c], aspect="equal")
                    ax.set_adjustable("box")


        out_path = os.path.join(
            OUT_DIR,
            f"hpatches_grid_{fig_idx:03d}.png"
        )

        if SAVE_FIGS:
            plt.savefig(out_path, dpi=FIG_DPI, pad_inches=0)

        plt.show()
        plt.close(fig)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    plot_hpatches_grids(
        HPATCHES_ROOT,
        SEQUENCES_PER_FIG,
        MAX_IMAGES_PER_SEQ,
        target_sequences
    )
