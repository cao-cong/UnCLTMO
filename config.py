import argparse
import os
import random

import numpy as np
import torch

from utils import params


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parser for gan network")
    # ====== GENERAL SETTINGS ======
    parser.add_argument("--checkpoint", type=int, default=0)
    parser.add_argument("--change_random_seed", type=int, default=10)

    # ====== TRAINING ======
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--G_lr", type=float, default=params.lr)
    parser.add_argument("--D_lr", type=float, default=params.lr)
    parser.add_argument("--lr_decay_step", type=float, default=1)
    parser.add_argument("--d_pretrain_epochs", type=int, default=5)
    parser.add_argument('--use_xaviar', type=int, default=1)

    # ====== SLIDER_MODE ======
    parser.add_argument("--manual_d_training", type=int, default=0)
    parser.add_argument("--d_weight_mul_mode", type=str, default="double")
    parser.add_argument("--strong_details_D_weights", type=str, default="1,1,1")
    parser.add_argument("--basic_details_D_weights", type=str, default="0.1,0.1,0.1")

    # ====== ARCHITECTURES ======
    parser.add_argument("--model", type=str, default=params.unet_network)
    parser.add_argument("--filters", type=int, default=params.filters)
    parser.add_argument("--unet_depth", type=int, default=4)
    parser.add_argument("--con_operator", type=str, default=params.square_and_square_root)
    parser.add_argument('--unet_norm', type=str, default='none', help="none/instance_norm/batch_norm")
    parser.add_argument('--g_activation', type=str, default='relu', help="none/relu/leakyrelu")
    parser.add_argument("--d_down_dim", type=int, default=16)
    parser.add_argument("--d_nlayers", type=int, default=3)
    parser.add_argument("--d_norm", type=str, default='none')
    parser.add_argument('--last_layer', type=str, default='sigmoid', help="none/tanh")
    #parser.add_argument('--d_model', type=str, default='multiLayerD_simpleD', help="original/patchD/multiLayerD")
    parser.add_argument('--d_model', type=str, default='simpleD', help="simpleD")

    parser.add_argument('--num_D', type=int, default=3, help="if d_model is multiLayerD then specify numD")
    #parser.add_argument('--d_last_activation', type=str, default='sigmoid', help="sigmoid/none")
    parser.add_argument('--d_last_activation', type=str, default='none', help="sigmoid/none")
    parser.add_argument('--stretch_g', type=str, default="none")
    parser.add_argument('--g_doubleConvTranspose', type=int, default=1)
    parser.add_argument('--d_fully_connected', type=int, default=0)
    parser.add_argument('--simpleD_maxpool', type=int, default=0)
    parser.add_argument('--bilinear', type=int, default=0)
    parser.add_argument('--padding', type=str, default="replicate")
    parser.add_argument('--d_padding', type=int, default=0)
    parser.add_argument('--convtranspose_kernel', type=int, default=2)
    parser.add_argument('--final_shape_addition', type=int, default=0)
    parser.add_argument('--up_mode', type=int, default=0)
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--output_dim", type=int, default=1)

    # ====== LOSS ======
    parser.add_argument('--train_with_D', type=int, default=1)
    parser.add_argument("--loss_g_d_factor", type=float, default=1)
    parser.add_argument('--adv_weight_list', help='delimited list input', type=str, default="1,1,1")
    parser.add_argument('--struct_method', type=str, default="gamma_struct")
    parser.add_argument("--ssim_loss_factor", type=float, default=1)
    parser.add_argument("--ssim_window_size", type=int, default=5)
    parser.add_argument('--pyramid_weight_list', help='delimited list input', type=str, default="1,1,1")

    # ====== DATASET ======
    parser.add_argument("--data_root_npy", type=str, default=params.train_dataroot_hdr)
    parser.add_argument("--data_root_ldr", type=str, default=params.train_dataroot_ldr)
    parser.add_argument("--test_dataroot_npy", type=str, default=params.test_dataroot_hdr)
    parser.add_argument("--test_dataroot_original_hdr", type=str, default=params.test_dataroot_original_hdr)
    parser.add_argument("--test_dataroot_ldr", type=str, default=params.test_dataroot_ldr)
    parser.add_argument("--input_images_mean", type=float, default=0)
    parser.add_argument('--use_factorise_data', type=int, default=1)
    parser.add_argument('--factor_coeff', type=float, default=0.1)
    parser.add_argument('--gamma_log', type=int, default=10)
    parser.add_argument('--f_factor_path', type=str, default=params.f_factor_path_hist)
    parser.add_argument('--use_new_f', type=int, default=0)
    parser.add_argument('--use_contrast_ratio_f', type=int, default=0)
    parser.add_argument('--use_hist_fit', type=int, default=1)
    parser.add_argument('--f_train_dict_path', type=str, default=params.f_factor_path_hist)
    parser.add_argument('--mean_hist_path', type=str, default=params.mean_hist_path)
    parser.add_argument('--lambdas_path', type=str, default=params.lambdas_path)
    parser.add_argument('--bins', type=int, default=20)
    parser.add_argument('--data_trc', type=str, default="min_log", help="gamma/log")
    parser.add_argument('--max_stretch', type=float, default=1)
    parser.add_argument('--min_stretch', type=float, default=0)
    parser.add_argument("--add_frame", type=int, default=0)
    parser.add_argument("--normalization", type=str, default='bugy_max_normalization', help='max/min_max')

    # ====== SAVE RESULTS ======
    parser.add_argument("--epoch_to_save", type=int, default=2)
    parser.add_argument("--result_dir_prefix", type=str, default="")
    parser.add_argument("--final_epoch", type=int, default=1)
    parser.add_argument("--fid_real_path", type=str,
                        default="/cs/snapless/raananf/yael_vinker/data/div2k_large/test_half2")  # default="/Users/yaelvinker/PycharmProjects/lab/fid/fake_jpg")#
    parser.add_argument("--fid_res_path", type=str,
                        default="/Users/yaelvinker/PycharmProjects/lab/results/")

    args = parser.parse_args()
    return args


def get_opt():
    opt = parse_arguments()
    if opt.change_random_seed > 1:
        manualSeed = opt.change_random_seed
    elif opt.change_random_seed == 1:
        manualSeed = random.randint(1, 10000)
    else:
        manualSeed = params.manualSeed
    opt.manual_seed = manualSeed
    torch.manual_seed(manualSeed)
    opt = define_dirs(opt)
    if opt.manual_d_training:
        opt.input_dim = 2
    opt.dataset_properties = get_dataset_properties(opt)
    np.save(os.path.join(opt.output_dir, "run_settings.npy"), vars(opt))

    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    opt.device = device
    opt.pyramid_weight_list = torch.FloatTensor([float(item) for item in opt.pyramid_weight_list.split(',')]).to(device)
    opt.adv_weight_list = torch.FloatTensor([float(item) for item in opt.adv_weight_list.split(',')]).to(
        device)

    opt.strong_details_D_weights = torch.FloatTensor(
        [float(item) for item in opt.strong_details_D_weights.split(',')]).to(device)
    opt.basic_details_D_weights = torch.FloatTensor(
        [float(item) for item in opt.basic_details_D_weights.split(',')]).to(device)
    return opt


def define_dirs(opt):
    opt.data_root_npy = os.path.join(opt.data_root_npy)
    opt.data_root_ldr = os.path.join(opt.data_root_ldr)
    opt.test_data_root_npy = os.path.join(opt.test_dataroot_npy)
    opt.test_data_root_ldr = os.path.join(opt.test_dataroot_ldr)
    opt.output_dir = create_dir(opt)
    return opt


def get_dataset_properties(opt):
    dataset_properties = {"train_root_npy": opt.data_root_npy,
                          "train_root_ldr": opt.data_root_ldr,
                          "test_dataroot_npy": opt.test_dataroot_npy,
                          "test_dataroot_original_hdr": opt.test_dataroot_original_hdr,
                          "test_dataroot_ldr": opt.test_dataroot_ldr,
                          "input_dim": opt.input_dim,
                          "output_dim": opt.output_dim,
                          "input_images_mean": opt.input_images_mean,
                          "use_factorise_data": opt.use_factorise_data,
                          "factor_coeff": opt.factor_coeff,
                          "add_frame": opt.add_frame,
                          "batch_size": opt.batch_size,
                          "normalization": opt.normalization,
                          "use_c3": False,
                          "max_stretch": opt.max_stretch,
                          "min_stretch": opt.min_stretch,
                          "data_trc": opt.data_trc,
                          "use_contrast_ratio_f": opt.use_contrast_ratio_f,
                          "use_hist_fit": opt.use_hist_fit,
                          "f_train_dict_path": opt.f_train_dict_path,
                          "final_shape_addition": opt.final_shape_addition}
    return dataset_properties


def get_G_params(opt):
    result_dir_pref = "G_%s" % params.con_op_short[opt.con_operator]
    # result_dir_pref += opt.model + "_" + params.con_op_short[opt.con_operator] + "_" + opt.g_activation
    if not opt.g_doubleConvTranspose:
        result_dir_pref += "_doubleConv_"
    else:
        result_dir_pref += "_doubleConvT_"
    if opt.up_mode:
        result_dir_pref += "_up_mode_"
    if opt.unet_norm != "none":
        result_dir_pref = result_dir_pref + "_g" + opt.unet_norm + "_"
    if opt.stretch_g != "none":
        result_dir_pref += opt.stretch_g + "_"
    return result_dir_pref


def get_D_params(opt):
    result_dir_pref = "D"
    # result_dir_pref = "D_%s_" % (opt.d_model)
    if "multiLayerD" in opt.d_model:
        # result_dir_pref = result_dir_pref + "_num_D" + str(opt.num_D)
        result_dir_pref += "_[%s]_" % (opt.adv_weight_list)
    # result_dir_pref += "ch%d_" % (opt.d_down_dim)
    # result_dir_pref += "%slayers_%s_" % (str(opt.d_nlayers), opt.d_last_activation)
    if opt.d_fully_connected:
        result_dir_pref += "fullyCon_"
    if "simpleD" in opt.d_model:
        if opt.simpleD_maxpool:
            result_dir_pref += "maxPool_"
    if opt.d_norm != "none":
        result_dir_pref += opt.d_norm + "_"
    # if opt.d_padding:
    result_dir_pref += "pad_" + str(opt.d_padding)
    return result_dir_pref


def get_training_params(opt):
    result_dir_pref = ""
    if opt.bilinear:
        result_dir_pref += "bilinear_"
    else:
        result_dir_pref += "trans" + str(opt.convtranspose_kernel) + "_"
    result_dir_pref += opt.padding + "_"
    if opt.change_random_seed:
        result_dir_pref = result_dir_pref + "rseed" + str(opt.manual_seed)
    if not opt.add_frame:
        result_dir_pref = result_dir_pref + "_noframe_"
    else:
        result_dir_pref = result_dir_pref + "_frame_" + str(opt.final_shape_addition)
    if opt.normalization == "stretch":
        result_dir_pref = result_dir_pref + "stretch_" + str(opt.max_stretch)
    return result_dir_pref


def get_data_params(opt):
    result_dir_pref = opt.data_trc + "_" + str(opt.factor_coeff)
    if opt.use_new_f:
        result_dir_pref = result_dir_pref + "new_f_"
    elif opt.use_contrast_ratio_f:
        result_dir_pref = result_dir_pref + "contrast_ratio_f_"
    elif opt.use_hist_fit:
        result_dir_pref = result_dir_pref + "hist_fit_"
    else:
        result_dir_pref = result_dir_pref + "data" + str(opt.gamma_log) + "_"
    return result_dir_pref


def get_losses_params(opt):
    result_dir_pref = ""
    result_dir_pref = result_dir_pref + "d" + str(opt.loss_g_d_factor)
    if opt.ssim_loss_factor:
        struct_loss = opt.struct_method
        if opt.manual_d_training:
            result_dir_pref += "_interp_" + opt.d_weight_mul_mode
            if opt.d_weight_mul_mode == "double":
                result_dir_pref += "_[(" + opt.strong_details_D_weights + ")_(" + opt.basic_details_D_weights + ")]_"
            else:
                result_dir_pref += "_%s_%s[%s]_" % (struct_loss, str(opt.ssim_loss_factor), opt.pyramid_weight_list)
        else:
            result_dir_pref += "_%s_%s[%s]_" % (struct_loss, str(opt.ssim_loss_factor), opt.pyramid_weight_list)
    return result_dir_pref


def create_dir(opt):
    result_dir_pref, model_name, con_operator, model_depth, filters, add_frame = opt.result_dir_prefix, opt.model, \
                                                                                 opt.con_operator, opt.unet_depth, \
                                                                                 opt.filters, opt.add_frame
    #if not opt.train_with_D:
    #    result_dir_pref = result_dir_pref + "no_D_"
    #else:
    #    result_dir_pref += get_D_params(opt)
    #result_dir_pref += "_" + get_G_params(opt) + "_" + get_losses_params(opt) + "_" + get_training_params(opt) \
    #                   + "_" + get_data_params(opt)
    output_dir = result_dir_pref

    model_path = params.models_save_path
    loss_graph_path = params.loss_path
    result_path = params.results_path

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Directory ", output_dir, " created")
    else:
        print("Directory ", output_dir, " exists")
    if not os.path.exists(opt.fid_res_path):
        os.makedirs(opt.fid_res_path)
    opt.fid_res_path = os.path.join(opt.fid_res_path, "fid_results.npy")

    models_images = os.path.join(output_dir, params.models_images)
    model_path = os.path.join(output_dir, model_path)
    best_model_save_path = os.path.join("best_model", "best_model.pth")
    best_model_path = os.path.join(output_dir, best_model_save_path)
    loss_graph_path = os.path.join(output_dir, loss_graph_path)
    result_path = os.path.join(output_dir, result_path)
    acc_path = os.path.join(output_dir, "accuracy")

    if not os.path.exists(models_images):
        os.mkdir(models_images)
        print("Directory ", models_images, " created")

    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
        print("Directory ", model_path, " created")

    if not os.path.exists(os.path.dirname(best_model_path)):
        os.makedirs(os.path.dirname(best_model_path))
        print("Directory ", best_model_path, " created")

    if not os.path.exists(loss_graph_path):
        os.mkdir(loss_graph_path)
        print("Directory ", loss_graph_path, " created")

    if not os.path.exists(result_path):
        os.mkdir(result_path)
        print("Directory ", result_path, " created")

    if not os.path.exists(acc_path):
        os.mkdir(acc_path)
        print("Directory ", acc_path, " created")
    return output_dir
