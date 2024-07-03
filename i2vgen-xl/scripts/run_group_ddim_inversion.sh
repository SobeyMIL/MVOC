#!/bin/bash
#source /home/YourName/miniconda3/etc/profile.d/conda.sh  #<-- change this to your own miniconda path

PYTHONPATH=.. python inverse.py \
--template_config "configs/group_inversion/template.yaml" \
--configs_json "configs/group_inversion/group_config.json"