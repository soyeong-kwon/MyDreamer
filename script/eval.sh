#!/bin/bash
cd evaluation

mode=""
num_prompts=8
num_objects=8

real_folder=""
sample_folder=""
sample_dm_folder=""

echo $mode
echo $num_prompts
echo $real_folder
echo $sample_folder
echo $sample_dm_folder

### image quality : FID 
python -m pytorch_fid "$real_folder" "$sample_folder/base"

# text alignment & object fidelity 
python evaluate.py --result_dir "$sample_dm_folder" \
                   --model "base" \
                   --mode "$mode" \
                   --num_prompts $num_prompts