#!/bin/csh

# ====== GENERAL SETTINGS ======
echo "checkpoint $1"
echo "change_random_seed $2"

# ====== TRAINING ======
echo "batch_size $3"
echo "num_epochs $4"
echo "G_lr $5"
echo "D_lr $6"
echo "lr_decay_step $7"
echo "d_pretrain_epochs $8"
echo "use_xaviar $9"

# ====== SLIDER_MODE ======
echo "manual_d_training ${10}"
echo "d_weight_mul_mode ${11}"
echo "strong_details_D_weights ${12}"
echo "basic_details_D_weights ${13}"

# ====== ARCHITECTURES ======
echo "model ${14}"
echo "filters ${15}"
echo "unet_depth ${16}"
echo "con_operator ${17}"
echo "unet_norm ${18}"
echo "g_activation ${19}"
echo "d_down_dim ${20}"
echo "d_nlayers ${21}"
echo "d_norm ${22}"
echo "last_layer ${23}"
echo "d_model ${24}"
echo "num_D ${25}"
echo "d_last_activation ${26}"
echo "stretch_g ${27}"
echo "g_doubleConvTranspose ${28}"
echo "d_fully_connected ${29}"
echo "simpleD_maxpool ${30}"
echo "bilinear ${31}"
echo "padding ${32}"
echo "d_padding ${33}"
echo "convtranspose_kernel ${34}"
echo "final_shape_addition ${35}"
echo "up_mode ${36}"
echo "input_dim ${37}"
echo "output_dim ${38}"

# ====== LOSS ======
echo "train_with_D ${39}"
echo "loss_g_d_factor ${40}"
echo "adv_weight_list ${41}"
echo "struct_method ${42}"
echo "ssim_loss_factor ${43}"
echo "ssim_window_size ${44}"
echo "pyramid_weight_list ${45}"

# ====== DATASET ======
echo "data_root_npy ${46}"
echo "data_root_ldr ${47}"
echo "test_dataroot_npy ${48}"
echo "test_dataroot_original_hdr ${49}"
echo "test_dataroot_ldr ${50}"
echo "use_factorise_data ${51}"
echo "factor_coeff ${52}"
echo "gamma_log ${53}"
echo "f_factor_path ${54}"
echo "use_new_f ${55}"
echo "use_contrast_ratio_f ${56}"
echo "use_hist_fit ${57}"
echo "f_train_dict_path ${58}"
echo "data_trc ${59}"
echo "max_stretch ${60}"
echo "min_stretch ${61}"
echo "add_frame ${62}"
echo "normalization ${63}"

# ====== SAVE RESULTS ======
echo "epoch_to_save ${64}"
echo "result_dir_prefix ${65}"
echo "final_epoch ${66}"
echo "fid_real_path ${67}"
echo "fid_res_path ${68}"


nohup python -W ignore -u main_train.py \
  --checkpoint $1 \
  --change_random_seed $2 \
  --batch_size $3 \
  --num_epochs $4 \
  --G_lr $5 \
  --D_lr $6 \
  --lr_decay_step $7 \
  --d_pretrain_epochs $8 \
  --use_xaviar $9 \
  --manual_d_training ${10} \
  --d_weight_mul_mode ${11} \
  --strong_details_D_weights ${12} \
  --basic_details_D_weights ${13} \
  --model ${14} \
  --filters ${15} \
  --unet_depth ${16} \
  --con_operator ${17} \
  --unet_norm ${18} \
  --g_activation ${19} \
  --d_down_dim ${20} \
  --d_nlayers ${21} \
  --d_norm ${22} \
  --last_layer ${23} \
  --d_model ${24} \
  --num_D ${25} \
  --d_last_activation ${26} \
  --stretch_g ${27} \
  --g_doubleConvTranspose ${28} \
  --d_fully_connected ${29} \
  --simpleD_maxpool ${30} \
  --bilinear ${31} \
  --padding ${32} \
  --d_padding ${33} \
  --convtranspose_kernel ${34} \
  --final_shape_addition ${35} \
  --up_mode ${36} \
  --input_dim ${37} \
  --output_dim ${38} \
  --train_with_D ${39} \
  --loss_g_d_factor ${40} \
  --adv_weight_list ${41} \
  --struct_method ${42} \
  --ssim_loss_factor ${43} \
  --ssim_window_size ${44} \
  --pyramid_weight_list ${45} \
  --data_root_npy ${46} \
  --data_root_ldr ${47} \
  --test_dataroot_npy ${48} \
  --test_dataroot_original_hdr ${49} \
  --test_dataroot_ldr ${50} \
  --use_factorise_data ${51} \
  --factor_coeff ${52} \
  --gamma_log ${53} \
  --f_factor_path ${54} \
  --use_new_f ${55} \
  --use_contrast_ratio_f ${56} \
  --use_hist_fit ${57} \
  --f_train_dict_path ${58} \
  --data_trc ${59} \
  --max_stretch ${60} \
  --min_stretch ${61} \
  --add_frame ${62} \
  --normalization ${63} \
  --epoch_to_save ${64} \
  --result_dir_prefix ${65} \
  --final_epoch ${66} \
  --fid_real_path ${67} \
  --fid_res_path ${68} &