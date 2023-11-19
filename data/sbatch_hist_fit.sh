#!/bin/bash

input_images_path="/cs/labs/raananf/yael_vinker/dng_collection/"
mean_hist_path="/cs/labs/raananf/yael_vinker/data/new_lum_est_hist/ldr_avg_hist_900_images_20_bins.npy"
output_path="/cs/labs/raananf/yael_vinker/data/new_lum_est_hist/zhang/"
inpue_names_path_dir="/cs/labs/raananf/yael_vinker/data/new_lum_est_hist/dng_name_split"
names=("dng[0_100]")
for ((i = 0; i < ${#names[@]}; ++i)); do
	cur_name="${names[i]}"
	inpue_names_path="$inpue_names_path_dir/$cur_name.npy"
	output_name="$cur_name"
	echo "$inpue_names_path"
	echo "$output_name"
	sbatch --mem=10000m -c2 --gres=gpu:1 --time=2-0 run_hist_fit.sh $input_images_path $mean_hist_path $output_path $output_name $inpue_names_path
done