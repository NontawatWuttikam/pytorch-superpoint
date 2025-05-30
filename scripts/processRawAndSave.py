import os
import sys
import glob
import yaml
from tqdm import tqdm
import cv2

sys.path.insert(0, "../ProxyOpt/")
from ISP_tools.ProxyISPDataset import ProxyISPDataset

input_raw_path = "/mnt/ssd2tb/boat/thesis/s21fe_dataset/raw"
train_config_path = "/home/boat/proxyISP/ProxyOpt/train_configs/v16.1.yaml"

config_name = os.path.basename(train_config_path).replace(".yaml", "")
output_base_path = f"/mnt/ssd2tb/boat/thesis/s21fe_dataset/processed_{config_name}"

if not os.path.exists(output_base_path):
    os.makedirs(output_base_path)
    print(f"Created base output directory: {output_base_path}")
else:
    # prompt to keep or delete existing output directory
    user_input = input(f"Output directory {output_base_path} already exists. Do you want to delete it? (y/n): ")
    if user_input.lower() == 'y':
        import shutil
        shutil.rmtree(output_base_path)
        os.makedirs(output_base_path)
        print(f"Deleted and recreated output directory: {output_base_path}")
    else:
        print(f"Using existing output directory: {output_base_path}")

with open(train_config_path, "r") as f:
    yaml_dict = yaml.safe_load(f)

config = yaml_dict["config"]
openisp_config = yaml_dict["openisp_config"]
hyp_setting = yaml_dict["hyp_setting"]

additional_conf = {
    "proxyopt_base_path": "/home/boat/proxyISP/ProxyOpt/",
}
dataset = ProxyISPDataset(config, openisp_config, hyp_setting, additional_conf=additional_conf)
dataset.target_size = [1920, 1920]
dataset.is_dynamic_size = False

# Find all .dng files recursively
image_paths = glob.glob(os.path.join(input_raw_path, "**", "*.dng"), recursive=True)

for image_path in tqdm(image_paths, desc="Processing images"):
    # Compute relative path and destination
    rel_path = os.path.relpath(image_path, input_raw_path)
    rel_dir = os.path.dirname(rel_path)
    output_dir = os.path.join(output_base_path, rel_dir)
    os.makedirs(output_dir, exist_ok=True)

    image_name = os.path.basename(image_path).replace(".dng", ".jpg")
    output_image_path = os.path.join(output_dir, image_name)

    if not os.path.exists(output_image_path):
        processed_image = dataset.process_raw(str(image_path))
        if processed_image is not None:
            processed_image = processed_image[:, :, ::-1]  # Convert BGR to RGB
            cv2.imwrite(output_image_path, processed_image)
            print(f"Processed and saved: {output_image_path}")
        else:
            print(f"Failed to process: {image_path}")
    else:
        print(f"Skipping {output_image_path}, already exists.")
