import os
import shutil
from pathlib import Path
import sys
import yaml
import cv2
import numpy as np
import sys

sys.path.insert(0, "../ProxyOpt/")
sys.path.insert(0, "../fast-openISP/")
sys.path.insert(0, "../ProxyOpt/pytorch-msssim/")
from ISP_tools.ProxyISPDataset import ProxyISPDataset, EXPERIMENT_OUTPUT_PATH

def process_dataset(source_dir, target_dir, train_config_path, hpatch_prefix, hyp=None, mtx=None, dist=None):
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

    for sequence in source_path.iterdir():
        if not sequence.is_dir():
            continue

        if sequence.name.split("_")[0] != hpatch_prefix and hpatch_prefix != "": continue
        target_sequence = target_path / sequence.name
        target_sequence.mkdir(parents=True, exist_ok=True)

        # Process .dng files and save as .ppm
        for dng_file in sorted(sequence.glob("*.dng")):
            index = dng_file.stem
            print("Processing:", dng_file)

            rgb_image = dataset.process_raw(str(dng_file), hyp_, False)

            # Apply undistortion if mtx and dist are provided
            if mtx is not None and dist is not None:
                h, w = rgb_image.shape[:2]
                new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
                rgb_image = cv2.undistort(rgb_image, mtx, dist, None, new_mtx)

            ppm_path = target_sequence / f"{index}.ppm"
            cv2.imwrite(str(ppm_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

        # Copy H_1_* files
        for homography_file in sequence.glob("H_1_*"):
            shutil.copy(homography_file, target_sequence)

if __name__ == "__main__":
    source_directory = "datasets/HPatches_s21fe_reordered"
    target_directory = "datasets/HPatches"
    train_config_path = sys.argv[1]

    hyp_path = sys.argv[2]

    hpatch_prefix = sys.argv[3]
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
    process_dataset(source_directory, target_directory, train_config_path, hpatch_prefix, hyp, mtx, dist)
