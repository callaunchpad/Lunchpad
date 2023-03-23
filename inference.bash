#!/bin/bash -l

eval "$(conda shell.bash hook)"

conda activate /home/sebbyzhao/miniconda3/envs/invcook
python --version
PROMPT=`python src/image_gen.py --image "/home/sebbyzhao/Lunchpad/data/1.jpg" --save_images "/home/sebbyzhao/Lunchpad/data"`
echo ${PROMPT}

conda activate /home/sebbyzhao/miniconda3/envs/stable_diff_debug
python --version
IMG=`python diffusers/stable_diffusion.py $PROMPT`

conda activate /home/sebbyzhao/miniconda3/envs/invcook
python --version
python src/final_inv_cook.py