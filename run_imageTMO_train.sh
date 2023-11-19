# ====== GENERAL SETTINGS ======
checkpoint=0
change_random_seed=0

# ====== TRAINING ======
#batch_size=16
batch_size=8
num_epochs=21
#num_epochs=100
G_lr=0.00001
D_lr=0.000015
lr_decay_step=50
#d_pretrain_epochs=50
#d_pretrain_epochs=1
d_pretrain_epochs=0
use_xaviar=1

# ====== SLIDER_MODE ======
manual_d_training=0
d_weight_mul_mode="none"
strong_details_D_weights="1,1,1"
basic_details_D_weights="0.8,0.5,0"

# ====== ARCHITECTURES ======
model="unet"
filters=32
unet_depth=4
con_operator="square_and_square_root"
unet_norm="none"
g_activation="relu"

d_down_dim=16
d_nlayers=3
d_norm="none"
last_layer="sigmoid"
#d_model="multiLayerD_simpleD"
d_model="simpleD"
num_D=3
#d_last_activation="sigmoid"
d_last_activation="none"
stretch_g="none"
g_doubleConvTranspose=1
d_fully_connected=0
simpleD_maxpool=0
bilinear=0
padding="replicate"
d_padding=0
convtranspose_kernel=2
final_shape_addition=0
up_mode=0
input_dim=1
output_dim=1

# ====== LOSS ======
train_with_D=1
loss_g_d_factor=0.1
#loss_g_d_factor=0.5
adv_weight_list="1,1,0"
#adv_weight_list="0,0,0"
#adv_weight_list="0.1,0.1,0"
struct_method="gamma_ssim"
ssim_loss_factor=1
ssim_window_size=5
pyramid_weight_list="0.2,0.4,0.6"

# ====== DATASET ======
data_root_npy="../../data/tone_mapping/HDRplus_patches512_npy/"
data_root_ldr="../../data/tone_mapping/DIV2K_patches512_npy2"
test_dataroot_npy="../../data/tone_mapping/HDRplus_patches256_npy_eval"
test_dataroot_original_hdr="activate_trained_model/input_images"
test_dataroot_ldr="../../data/tone_mapping/DIV2K_patches256_npy_eval"
use_factorise_data=1
factor_coeff=0.1
gamma_log=10
f_factor_path="data/input_images_lambdas_HDRplus256train.npy"
use_new_f=0
use_contrast_ratio_f=0
use_hist_fit=1
f_train_dict_path="data/input_images_lambdas_HDRplus256train.npy"
data_trc="min_log"
max_stretch=1
min_stretch=0
add_frame=0
normalization="bugy_max_normalization"

# ====== SAVE RESULTS ======
#epoch_to_save=10
epoch_to_save=1
result_dir_prefix="results_imageTMOTrain/bs8_pretrain0epoch"

final_epoch=20
#final_epoch=40
fid_real_path="other/fid_real/"
fid_res_path="other/fid_res/"


test_names=("lr_reg")

#pyramid_weight_list_lst=("0.3,0.4,0.8")
#"0.2,0.4,0.6" "0.2,0.2,0.6" "0.2,0.2,0.8" "0.2,0.4,0.6")
#adv_weight_list_lst=("0.8,0.5,0.1")
#"1,1,0.5" "1,1,0.2" "1,1,0" "1,1,0.3")
#"0.1,0.4,0.6" "0.2,0.2,0.6" "0.1,0.2,0.6" "0.2,0.2,0.4")
#pyramid_weight_list_lst=("1,1,1")
#adv_weight_list_lst=("0.1,0.1,0.1")
pyramid_weight_list_lst=("1,1,1")
#adv_weight_list_lst=("1,1,1")
adv_weight_list_lst=("0.2,0.2,0.2")

for ((i = 0; i < ${#pyramid_weight_list_lst[@]}; ++i)); do

  test_name="${test_names[0]}"
  pyramid_weight_list="${pyramid_weight_list_lst[i]}"
  adv_weight_list="${adv_weight_list_lst[i]}"

  echo "======================================================"
  echo "tests_name $test_name"
  echo "adv_weight_list $adv_weight_list"
  echo "pyramid_weight_list $pyramid_weight_list"

  bash train_imageTMO.sh \
    $checkpoint \
    $change_random_seed \
    $batch_size \
    $num_epochs \
    $G_lr \
    $D_lr \
    $lr_decay_step \
    $d_pretrain_epochs \
    $use_xaviar \
    $manual_d_training \
    $d_weight_mul_mode \
    $strong_details_D_weights \
    $basic_details_D_weights \
    $model \
    $filters \
    $unet_depth \
    $con_operator \
    $unet_norm \
    $g_activation \
    $d_down_dim \
    $d_nlayers \
    $d_norm \
    $last_layer \
    $d_model \
    $num_D \
    $d_last_activation \
    $stretch_g \
    $g_doubleConvTranspose \
    $d_fully_connected \
    $simpleD_maxpool \
    $bilinear \
    $padding \
    $d_padding \
    $convtranspose_kernel \
    $final_shape_addition \
    $up_mode \
    $input_dim \
    $output_dim \
    $train_with_D \
    $loss_g_d_factor \
    $adv_weight_list \
    $struct_method \
    $ssim_loss_factor \
    $ssim_window_size \
    $pyramid_weight_list \
    $data_root_npy \
    $data_root_ldr \
    $test_dataroot_npy \
    $test_dataroot_original_hdr \
    $test_dataroot_ldr \
    $use_factorise_data \
    $factor_coeff \
    $gamma_log \
    $f_factor_path \
    $use_new_f \
    $use_contrast_ratio_f \
    $use_hist_fit \
    $f_train_dict_path \
    $data_trc \
    $max_stretch \
    $min_stretch \
    $add_frame \
    $normalization \
    $epoch_to_save \
    $result_dir_prefix \
    $final_epoch \
    $fid_real_path \
    $fid_res_path
  echo "======================================================"
done
