#!/bin/bash
#source /home/YourName/miniconda3/etc/profile.d/conda.sh #<-- change this to your own miniconda path

PYTHONPATH=.. python composite.py \
--template_config "configs/group_composite/template.yaml" \
--configs_json "configs/group_composite/group_config.json"