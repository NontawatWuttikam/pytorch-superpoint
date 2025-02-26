#!/bin/bash
python export.py export_descriptor  configs/magicpoint_repeatability_heatmap_coco.yaml superpoint_coco_test
python evaluation.py logs/superpoint_coco_test/predictions --repeatibility --outputImg --homography --plotMatching