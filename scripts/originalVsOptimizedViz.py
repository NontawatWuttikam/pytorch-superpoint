import sys
sys.path.insert(0, "../ProxyOpt/")
sys.path.insert(0, "../fast-openISP/")
import yaml
import rawpy
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import io
from contextlib import redirect_stdout
import os
from ISP_tools.ProxyISPDataset import ProxyISPDataset, EXPERIMENT_OUTPUT_PATH
from tqdm import tqdm

plot_original_hype = True
limit = 10
config_path = "../ProxyOpt/train_configs/v16.1.yaml"
images_path = "/mnt/ssd2tb/boat/thesis/s21fe_dataset/raw"
hype_checkpoint_paths = [
    "logs/PRETRAINED_train_v16.1_RawEdgeContrastIndexTop100_lr0.0005_descLossOnly_gradac8/proxyopt_checkpoints/checkpoint_158200.pkl"
]
fig_path = "visualize_output/originalVsOptimizedViz-.png"

isp_hyps = [np.load(p, allow_pickle = True)['proxy_hype'] for p in hype_checkpoint_paths]

with open(config_path, "r") as f:
    yaml_dict = yaml.safe_load(f)

config = yaml_dict["config"]
openisp_config = yaml_dict["openisp_config"]
hyp_setting = yaml_dict["hyp_setting"]

additional_conf = {
        # "target_image": ["/home/boat/proxyISP/data/s21fe_dataset/20240115_123915.dng"],
        # "target_image": ["/home/boat/proxyISP/data/s21fe_dataset/20240117_182706.dng"],
        "proxyopt_base_path": "/home/boat/proxyISP/ProxyOpt/"
}

raw_images = [str(p) for p in Path(images_path).rglob("*.dng")]
additional_conf["target_image"] = raw_images

dataset = ProxyISPDataset(config, openisp_config, hyp_setting, additional_conf)

total_figure_per_image = len(isp_hyps) + plot_original_hype
# plt.figure(figsize=(5 * total_figure_per_image, 5))


row_count = min(limit, len(dataset.raw_img))

plt.figure(figsize=(total_figure_per_image * 4,  row_count * 4))

with redirect_stdout(io.StringIO()):
    for idx, image_path in enumerate(tqdm(dataset.raw_img[:limit], desc="Processing images")):
        print("image : ", os.path.basename(image_path))
        bayer = rawpy.imread(image_path).raw_image
        if plot_original_hype:
            plt.subplot(row_count, total_figure_per_image, idx * total_figure_per_image + 1)
            processed = dataset.process_raw(bayer, original_hyp=True)
            plt.imshow(processed)
            if idx == 0:
                plt.title("Original hype")
            plt.axis("off")
        for k, isp_hyp in enumerate(isp_hyps):
            isp_hyp = dataset.denormalize_hyp(isp_hyp)
            processed = dataset.process_raw(bayer, isp_hyp, original_hyp=False)
            plt.subplot(row_count, total_figure_per_image, idx * total_figure_per_image + k + 2)
            plt.imshow(processed)
            if idx == 0:
                plt.title(hype_checkpoint_paths[k].split("/")[-1].replace(".pkl", ""))
            plt.axis("off")
    # plt.tight_layout()
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
