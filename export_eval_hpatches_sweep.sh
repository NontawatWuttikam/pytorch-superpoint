#!/bin/bash

# Usage Example:
# ./sweep_eval.sh 80000 200000 20000

start=75000      # e.g. 80000
end=90000        # e.g. 200000
step=200

# proxyopt config path
proxyoptConfig="../ProxyOpt/train_configs/v16.2-chroma-ISPDefaultInitialHype.yaml"

# common settings
extraSuffix="_HPatchesV4.1"
gpu_devices="0"
PreHPatchesPath="/mnt/ssd2tb/boat/thesis/s21fe_hpatches_v4.1"
hpatchesSeqPrefix="sl"  # ll, wl, sl

# base directory where checkpoints are located
checkpoint_base="/home/boat/proxyISP/pytorch-superpoint/logs/PRETRAINED_SAMEMODELHOMOADAPT_AGGRESSIVEHOMOADAPT_CFANORMALIZE_XHOMOWARP_DETERMHOMOADAPT_train_v16.2-chroma-HumanTunedInitialHype_sunlit_lr0.005_bothLoss_initialHypeHomoAdaptOnly_gradac187/proxyopt_checkpoints"

# Conda environments
env_gen="proxyopt"
env_eval="py36-sp"

echo "Sweeping checkpoints from ${start} to ${end} step ${step}..."
echo ""

for ((step_num=${start}; step_num<=${end}; step_num+=${step})); do

    stage2Checkpoint="${checkpoint_base}/checkpoint_${step_num}.pkl"

    if [[ ! -f "$stage2Checkpoint" ]]; then
        echo "❌ Checkpoint not found: $stage2Checkpoint — skipping"
        continue
    fi

    parent_dir=$(basename "$(dirname "$checkpoint_base")")
    dataName="eval_${hpatchesSeqPrefix}_${parent_dir}_${step_num}${extraSuffix}"

    echo "======================================================"
    echo "✨ Evaluating checkpoint: $stage2Checkpoint"
    echo "🔹 Using dataName: $dataName"
    echo "======================================================"

    rm -rf datasets/HPatches

    source ~/miniconda3/etc/profile.d/conda.sh

    # Generate HPatches
    conda activate $env_gen
    python make_hpatches.py "$proxyoptConfig" "$stage2Checkpoint" "$hpatchesSeqPrefix" "$PreHPatchesPath"
    conda deactivate

    # Export & Evaluate
    conda activate $env_eval
    CUDA_VISIBLE_DEVICES=$gpu_devices python export.py export_descriptor configs/magicpoint_repeatability_heatmap_proxyopt.yaml "$dataName"
    CUDA_VISIBLE_DEVICES=$gpu_devices python evaluation.py "logs/$dataName/predictions" --repeatibility --homography
    conda deactivate

    echo "🎉 Finished step $step_num"
    echo ""

done

echo "🧡 All done. Take a breath, you're doing great."
