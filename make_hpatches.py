import os
import shutil
from pathlib import Path
import sys
import yaml
import cv2
import numpy as np
import sys
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.insert(0, "../ProxyOpt/")
sys.path.insert(0, "../fast-openISP/")
sys.path.insert(0, "../ProxyOpt/pytorch-msssim/")
from ISP_tools.ProxyISPDataset import ProxyISPDataset, EXPERIMENT_OUTPUT_PATH

def process_dng_file(dng_file_info):
    """Worker function to process a single DNG file"""
    dng_file, target_sequence, dataset, hyp_, mtx, dist = dng_file_info

    index = dng_file.stem
    print(f"Processing: {dng_file}")

    try:
        rgb_image = dataset.process_raw(str(dng_file), hyp_, False)

        # Apply undistortion if mtx and dist are provided
        if mtx is not None and dist is not None:
            h, w = rgb_image.shape[:2]
            new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
            rgb_image = cv2.undistort(rgb_image, mtx, dist, None, new_mtx)

        ppm_path = target_sequence / f"{index}.ppm"
        cv2.imwrite(str(ppm_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

        return f"Successfully processed: {dng_file.name}"
    except Exception as e:
        return f"Error processing {dng_file.name}: {str(e)}"

def process_sequence(sequence_info):
    """Worker function to process a single sequence"""
    sequence, target_path, dataset, hyp_, mtx, dist = sequence_info

    try:
        target_sequence = target_path / sequence.name
        target_sequence.mkdir(parents=True, exist_ok=True)

        # Collect all DNG files for sequential processing within this sequence
        dng_files = sorted(sequence.glob("*.dng"))

        success_count = 0
        error_count = 0

        if dng_files:
            print(f"Processing {len(dng_files)} DNG files in sequence: {sequence.name}")

            # Process DNG files sequentially within this sequence
            for dng_file in dng_files:
                dng_file_info = (dng_file, target_sequence, dataset, hyp_, mtx, dist)
                result = process_dng_file(dng_file_info)

                if "Successfully" in result:
                    success_count += 1
                else:
                    error_count += 1

            sequence_result = f"Sequence {sequence.name}: {success_count} files processed successfully"
            if error_count > 0:
                sequence_result += f", {error_count} files failed"

        # Copy H_1_* files
        homography_files = list(sequence.glob("H_1_*"))
        for homography_file in homography_files:
            shutil.copy(homography_file, target_sequence)

        if homography_files:
            sequence_result += f", {len(homography_files)} homography files copied"

        return sequence_result

    except Exception as e:
        return f"Error processing sequence {sequence.name}: {str(e)}"

def process_dataset_flat_parallel(source_dir, target_dir, train_config_path, hpatch_prefix, hyp=None, mtx=None, dist=None, num_processes=None):
    """Alternative approach: flatten all DNG files and process in single pool for maximum parallelization"""
    if num_processes is None:
        num_processes = cpu_count()

    with open(train_config_path, "r") as f:
        yaml_dict = yaml.safe_load(f)

    config = yaml_dict["config"]
    openisp_config = yaml_dict["openisp_config"]
    hyp_setting = yaml_dict["hyp_setting"]
    additional_conf = {
        "proxyopt_base_path" : Path(os.getcwd()).parent / "ProxyOpt"
    }

    dataset = ProxyISPDataset(config, openisp_config, hyp_setting, additional_conf)

    hyp_ = dataset.get_original_hyp(False, False)
    if hyp is not None:
        hyp = dataset.denormalize_hyp(hyp)
        hyp_ = hyp

    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # Collect all DNG files from all sequences
    all_dng_files = []
    sequence_dirs = {}

    for sequence in source_path.iterdir():
        if not sequence.is_dir():
            continue
        if sequence.name.split("_")[0] != hpatch_prefix and hpatch_prefix != "":
            continue

        target_sequence = target_path / sequence.name
        target_sequence.mkdir(parents=True, exist_ok=True)
        sequence_dirs[sequence.name] = target_sequence

        # Collect DNG files for this sequence
        for dng_file in sorted(sequence.glob("*.dng")):
            all_dng_files.append((dng_file, target_sequence, dataset, hyp_, mtx, dist))

    if not all_dng_files:
        print("No DNG files found to process")
        return

    print(f"Processing {len(all_dng_files)} DNG files across all sequences using {num_processes} processes")

    # Process all DNG files in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_dng_file, all_dng_files)

    # Count results by sequence
    sequence_results = {}
    for result, (dng_file, target_sequence, _, _, _, _) in zip(results, all_dng_files):
        seq_name = target_sequence.name
        if seq_name not in sequence_results:
            sequence_results[seq_name] = {'success': 0, 'error': 0}

        if "Successfully" in result:
            sequence_results[seq_name]['success'] += 1
        else:
            sequence_results[seq_name]['error'] += 1

    # Copy homography files for each sequence
    for sequence in source_path.iterdir():
        if sequence.name not in sequence_dirs:
            continue

        target_sequence = sequence_dirs[sequence.name]
        homography_files = list(sequence.glob("H_1_*"))
        for homography_file in homography_files:
            shutil.copy(homography_file, target_sequence)

    # Print results summary
    print("\n=== Processing Summary ===")
    for seq_name, counts in sequence_results.items():
        result_str = f"Sequence {seq_name}: {counts['success']} files processed successfully"
        if counts['error'] > 0:
            result_str += f", {counts['error']} files failed"
        print(result_str)

def process_dataset(source_dir, target_dir, train_config_path, hpatch_prefix, hyp=None, mtx=None, dist=None, num_processes=None):
    # Set default number of processes to CPU count
    if num_processes is None:
        num_processes = cpu_count()

    with open(train_config_path, "r") as f:
        yaml_dict = yaml.safe_load(f)

    config = yaml_dict["config"]
    openisp_config = yaml_dict["openisp_config"]
    hyp_setting = yaml_dict["hyp_setting"]
    additional_conf = {
        "proxyopt_base_path" : Path(os.getcwd()).parent / "ProxyOpt" # assume called in pytorch-superpoint
    }

    dataset = ProxyISPDataset(config, openisp_config, hyp_setting, additional_conf)

    hyp_ = dataset.get_original_hyp(False, False)
    if hyp is not None:
        hyp = dataset.denormalize_hyp(hyp)
        hyp_ = hyp

    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # Collect all valid sequences for parallel processing
    sequences = []
    for sequence in source_path.iterdir():
        if not sequence.is_dir():
            continue
        if sequence.name.split("_")[0] != hpatch_prefix and hpatch_prefix != "":
            continue
        sequences.append(sequence)

    if not sequences:
        print("No matching sequences found to process")
        return

    # Use available processes to parallelize at sequence level
    sequence_count = len(sequences)
    sequences_processes = min(num_processes, sequence_count)

    print(f"Processing {sequence_count} sequences using {sequences_processes} processes")
    print("Note: DNG files within each sequence are processed sequentially to avoid nested multiprocessing")

    # Prepare arguments for each sequence
    sequence_args = [(seq, target_path, dataset, hyp_, mtx, dist)
                    for seq in sequences]

    # Process sequences in parallel
    with Pool(processes=sequences_processes) as pool:
        results = pool.map(process_sequence, sequence_args)

    # Print results summary
    print("\n=== Processing Summary ===")
    for result in results:
        print(result)

if __name__ == "__main__":
    source_directory = sys.argv[4]
    # source_directory = "datasets/HPatches_s21fe_reordered"
    target_directory = "datasets/HPatches"
    train_config_path = sys.argv[1]

    hyp_path = sys.argv[2]

    hpatch_prefix = sys.argv[3]

    # Optional: number of processes (defaults to CPU count if not provided)
    num_processes = 32
    if len(sys.argv) > 5:
        try:
            num_processes = int(sys.argv[5])
            print(f"Using custom process count: {num_processes}")
        except ValueError:
            print("Invalid process count provided, using default (CPU count)")

    if hyp_path == "original":
        hyp = None
    else:
        import pickle
        with open(hyp_path, "rb") as f:
            hyp = pickle.load(f)["proxy_hype"]

    # Example camera matrix and distortion coefficients (should be replaced with actual values)
    cam_param = np.load("../ProxyOpt/camera_parameters/s21fe_main_3000x4000/CalibrationMatrix_college_cpt.npz")
    mtx = cam_param["Camera_matrix"]
    dist = cam_param["distCoeff"]  # Placeholder values

    # Use flat parallel approach for maximum DNG file parallelization (avoids nested multiprocessing)
    process_dataset_flat_parallel(source_directory, target_directory, train_config_path, hpatch_prefix, hyp, mtx, dist, num_processes)

    # Alternative: Use sequence-level parallelization (uncomment line below and comment line above)
    # process_dataset(source_directory, target_directory, train_config_path, hpatch_prefix, hyp, mtx, dist, num_processes)
