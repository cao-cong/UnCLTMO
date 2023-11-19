#!/bin/bash

#source /cs/labs/raananf/yael_vinker/my_venv/bin/activate

echo "path_real $1"
echo "path_fake $2"
echo "number_of_images $3"
echo "batch-size $4"

python3.6 fid_score_small_dset.py --path_real $1 \
  --path_fake $2 \
  --batch-size $4 --dims 768 --gpu "cuda" --number_of_images $3 --format "jpg"

