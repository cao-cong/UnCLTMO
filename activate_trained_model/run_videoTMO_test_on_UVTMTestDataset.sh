model_path="model_weights_videoTMO"
model_name="bs8_pretrain0epoch"

input_images_path="../../../data/tone_mapping/test_HDRvideo"
f_factor_path="lambda_data/input_images_lambdas.npy"
output_path="output_testvideo"

# lambda calc params
mean_hist_path="lambda_data/ldr_avg_hist_900_images_20_bins.npy"
lambda_output_path="lambda_data"
bins=20

python test_videoTMO.py --model_name $model_name \
  --input_images_path $input_images_path --output_path $output_path --model_path $model_path \
  --f_factor_path $f_factor_path \
  --mean_hist_path $mean_hist_path --lambda_output_path $lambda_output_path --bins $bins
