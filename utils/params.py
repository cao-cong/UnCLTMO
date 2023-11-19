import os

train_dataroot_hdr = "data/hdr_data_with_f"
train_dataroot_ldr = "data/ldr_data"

test_dataroot_hdr = "data/hdr_data_with_f"
test_dataroot_ldr = "data/ldr_data"
test_dataroot_original_hdr = "data/temp_data"


# f_factor_path = "/Users/yaelvinker/PycharmProjects/lab/utils/test_factors.npy"
f_factor_path_hist="data/lambdas/train_hdr_20_bins.npy"
mean_hist_path="activate_trained_model/lambda_data/ldr_avg_hist_900_images_20_bins.npy"
lambdas_path="data/lambdas/"
image_key = "hdr_image"
window_image_key = "binary_windows"
gray_input_image_key = "input_im"
color_image_key = "color_im"
original_gray_key = "original_gray"
original_gray_norm_key = "original_gray_norm"
gamma_factor = "gamma_factor"

apply_windows_loss = False


models_save_path = os.path.join("models")
loss_path = os.path.join("loss_plot")
best_acc_images_path = os.path.join("best_acc_images")
results_path = os.path.join("result_images")
gradient_flow_path = os.path.join("gradient_flow")
models_images = os.path.join("model_results")

# shape_addition = 45
# shape_addition = 32 #78
# final_shape_addition = 32 * 2
#input_size = 268
input_size = 256
n_downsample = 1
n_downsamples_d = 3
dim = 64
dim_d = 6
# Number of workers for dataloader
workers = 1

# Batch size during training
batch_size = 28

epsilon = 1e-08
epsilon2 = 1e-05

# Spatial size of training images. All images will be resized to this
# size using a transformer.

# Number of training epochs
num_epochs = 10

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# # Number of GPUs available. Use 0 for CPU mode.
# ngpu = 1

manualSeed = 999

g_d_loss_factor = 0.2

unet_network = "unet"
torus_network = "torus"

layer_factor_2_operators = ["original_unet"]
layer_factor_3_operators = ["square", "square_root", "gamma"]
layer_factor_4_operators = ["square_and_square_root", "square_and_square_root_manual_d"]

# con operators
original_unet = "original_unet"
square = "square"
square_root = "square_root"
square_and_square_root = "square_and_square_root"
gamma = "gamma"
square_and_square_root_manual_d = "square_and_square_root_manual_d"

# con operator short
con_op_short = {"original_unet": "ou",
                "square": "s",
                "square_root": "sr",
                "square_and_square_root": "ssr",
                "gamma": "g",
                "square_and_square_root_manual_d": "ssrMD"}

filters = 32

ssim_tmqi = "ssim_tmqi"
ssim_custom = "ssim_custom"

patchD_map_dim = {3: 30 ** 2,
                  4: 14 ** 2,
                  5: 6 ** 2,
                  6: 2 ** 2}

def get_multiLayerD_map_dim(num_D, d_nlayers):
    num_dim = 0
    for i in range(num_D):
        num_dim += patchD_map_dim[d_nlayers + i]
    return num_dim


# patchD_map_dim = {3: 35 ** 2,
#                   4: 19 ** 2,
#                   5: 6 ** 2,
#                   6: 2 ** 2}