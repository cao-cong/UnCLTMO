#!/bin/csh

source /cs/labs/raananf/yael_vinker/python7-venv/bin/activate.csh

echo "input_images_path $1"
echo "mean_hist_path $2"
echo "output_path $3"
echo "output_name $4"
echo "input_names_path $5"

python3.7 -W ignore -u lum_est_test.py --input_images_path $1 --mean_hist_path $2 --output_path $3 --output_name $4 --inpue_names_path $5