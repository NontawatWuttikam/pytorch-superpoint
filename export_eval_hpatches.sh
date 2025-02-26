#!/bin/bash
dataName="superpoint_hpatches-superpoint_coco_proxyopt_allim-checkpoint_28500"
python export.py export_descriptor  configs/magicpoint_repeatability_heatmap.yaml $dataName
python evaluation.py logs/$dataName/predictions --repeatibility --outputImg --homography --plotMatching