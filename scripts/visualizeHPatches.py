import os
import glob
from PIL import Image
import matplotlib.pyplot as plt

def visualize_ppm_grid(root_dir, output_path, prefix_filter='*', shuffle=False, limit=6):
    # Find matching subdirectories
    pattern = os.path.join(root_dir, prefix_filter)
    subdirs = sorted([
        d for d in glob.glob(pattern)
        if os.path.isdir(d)
    ])

    if shuffle:
        import random
        random.shuffle(subdirs)

    subdirs = subdirs[:limit]  # Limit the number of subdirectories to visualize
    num_subdirs = len(subdirs)

    if num_subdirs == 0:
        print("No matching subdirectories found.")
        return

    fig, axs = plt.subplots(nrows=num_subdirs, ncols=6, figsize=(6 * 2, num_subdirs * 2))

    for row, subdir in enumerate(subdirs):
        # Get sorted list of .ppm files
        ppm_files = sorted([
            f for f in os.listdir(subdir) if f.lower().endswith('.ppm')
        ])[:6]  # Only use the first 6 images

        for col in range(6):
            ax = axs[row, col] if num_subdirs > 1 else axs[col]
            if col < len(ppm_files):
                img_path = os.path.join(subdir, ppm_files[col])
                img = Image.open(img_path)
                ax.imshow(img)
            ax.axis('off')

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Image grid saved to {output_path}")

output_file = 'visualize_output/hpatches_grid_ll.png'
visualize_ppm_grid('datasets/HPatches_s21fe', output_path=output_file, prefix_filter='ll*', shuffle=True, limit=100)
