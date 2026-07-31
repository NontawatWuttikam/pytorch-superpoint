#!/bin/bash

start=0
end=2500
step=30

# Skip checkpoints that have already been evaluated
skip_existing=true   # true / false

# proxyopt config path
proxyoptConfig="../ProxyOpt/train_configs/v16.2-chroma-HumanTunedInitialHype.yaml"

# common settings
extraSuffix="_HPatchesV4.1"
gpu_devices="0"
PreHPatchesPath="/mnt/ssd2tb/boat/thesis/s21fe_hpatches_v4.1"
hpatchesSeqPrefix="sl"

# base directory where checkpoints are located
checkpoint_base="/home/boat/proxyISP/pytorch-superpoint/logs/CMAES_train_sunlit_maxstd0.01/cma_es_checkpoints"

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

    # Skip if evaluation already exists
    if [[ "$skip_existing" == true && -d "logs/$dataName" ]]; then
        echo "⏭️  Already exists: logs/$dataName — skipping"
        continue
    fi

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

echo "🧡 All done."