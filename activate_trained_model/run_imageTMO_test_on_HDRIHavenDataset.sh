model_path="model_weights_imageTMO"
model_name="11_08_lr15D_size268_D_[1,1,1]_pad_0_G_ssr_doubleConvT__d1.0_struct_1.0[1,1,1]__trans2_replicate__noframe__min_log_0.1hist_fit_"

input_images_path="../../../data/tone_mapping/HDRIHaven_dataset"
f_factor_path="lambda_data/input_images_lambdas_HDRHaven.npy"
output_path="output_testimage_HDRIHaven"

# lambda calc params
mean_hist_path="lambda_data/ldr_avg_hist_900_images_20_bins.npy"
lambda_output_path="lambda_data"
bins=20

python test_imageTMO.py --model_name $model_name \
  --input_images_path $input_images_path --output_path $output_path --model_path $model_path \
  --f_factor_path $f_factor_path \
  --mean_hist_path $mean_hist_path --lambda_output_path $lambda_output_path --bins $bins
