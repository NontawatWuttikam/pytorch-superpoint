import os
import shutil
from pathlib import Path

def process_hpatches_sequence(seq_path, out_seq_path):
    # Ensure the output sequence folder exists
    out_seq_path.mkdir(parents=True, exist_ok=True)

    # Read the swap sequence
    swap_file = seq_path / 'swap_seq.txt'
    if not swap_file.exists():
        print("WARNING: swap_seq.txt not found in", seq_path)
        raise Exception("swap_seq.txt is required for reordering sequences.")

    with open(swap_file, 'r') as f:
        swap = list(map(int, f.read().strip().split()))

    # Create a reverse map: old_index -> new_index
    new_to_old = {i + 1: swap[i] for i in range(6)}
    old_to_new = {v: k for k, v in new_to_old.items()}

    # Copy and rename ppm and dng
    for old_idx, new_idx in old_to_new.items():
        for ext in ['ppm', 'dng']:
            src = seq_path / f"{old_idx}.{ext}"
            dst = out_seq_path / f"{new_idx}.{ext}"
            print("copying", src, "to", dst)
            shutil.copyfile(src, dst)

    # Copy and rename homographies
    for old_idx in range(2, 7):
        old_h_file = seq_path / f"H_1_{old_idx}"
        if old_h_file.exists():
            new_src_idx = old_to_new[1]
            assert new_src_idx == 1, "Source index for H1 should always be 1"
            new_dst_idx = old_to_new[old_idx]
            new_h_file = out_seq_path / f"H_{new_src_idx}_{new_dst_idx}"
            print("copying", old_h_file, "to", new_h_file)
            shutil.copyfile(old_h_file, new_h_file)

    # Copy swap_seq.txt for reference
    shutil.copyfile(swap_file, out_seq_path / 'swap_seq.txt')

def process_hpatches_dataset(src_root, dst_root):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    for seq_folder in sorted(src_root.iterdir()):
        if seq_folder.is_dir():
            print(f"Processing: {seq_folder.name}")
            process_hpatches_sequence(seq_folder, dst_root / seq_folder.name)

# Example usage:
process_hpatches_dataset('/mnt/ssd2tb/boat/thesis/HPatches_s21fe_sl_coarseHomo', '/mnt/ssd2tb/boat/thesis/HPatches_s21fe_sl_coarseHomo_reordered')
