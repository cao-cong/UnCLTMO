import argparse
import os
import time
import sys
import inspect

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import models.unet_multi_filters.Unet_singleFrame as Generator
import tranforms
from utils import model_save_util, hdr_image_util, data_loader_util, params, adaptive_lambda

extensions = [".hdr", ".dng", ".exr", ".npy"]


def get_args():
    parser = argparse.ArgumentParser(description="Parser for gan network")
    parser.add_argument("--model_name", type=str, default=default_params["model_name"])
    parser.add_argument("--input_images_path", type=str, default=default_params["input_images_path"])
    parser.add_argument("--output_path", type=str, default=default_params["output_path"])
    parser.add_argument("--model_path", type=str, default=default_params["model_path"])
    parser.add_argument("--f_factor_path", type=str, default=default_params["f_factor_path"])

    # lambda calc params
    parser.add_argument("--mean_hist_path", type=str, default=default_params["mean_hist_path"])
    parser.add_argument("--lambda_output_path", type=str, default=default_params["lambda_output_path"])
    parser.add_argument("--bins", type=str, default=default_params["bins"])
    args = parser.parse_args()
    # print_args(args)
    return args


def run_trained_model(args):
    #args.f_factor_path = adaptive_lambda.calc_lambda(args.f_factor_path, extensions, args.input_images_path,
    #                                            args.mean_hist_path, args.lambda_output_path, args.bins)
    start = time.time()
    #net_path = os.path.join(args.model_path, "trained_weights.pth")
    net_path = os.path.join(args.model_path, "net_epoch5_iter62.pth")
    train_settings_path = os.path.join(args.model_path, "run_settings.npy")
    output_images_path = os.path.join(args.output_path)
    model_params = model_save_util.get_model_params(args.model_name, train_settings_path)
    # model_params["factor_coeff"] = 0.2
    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    run_model_on_path(model_params, device, net_path, args.input_images_path,
                      output_images_path, args.f_factor_path, None, model_params["final_shape_addition"])
    print("tone mapping took [%.2f] seconds" % (time.time() - start))


def run_model_on_path(model_params, device, net_path, input_images_path, output_images_path,
                      f_factor_path, net_G, final_shape_addition):
    if not net_G:
        net_G = load_g_model(model_params, device, net_path)
    print("\nModel [%s] was loaded successfully\n" % model_params["model"])
    names = os.listdir(input_images_path)
    for img_name in names:
        im_path = os.path.join(input_images_path, img_name)
        print("processing [%s]" % img_name)
        if os.path.splitext(img_name)[1] in extensions:
            model_save_util.run_model_on_single_image2(net_G, im_path, device, os.path.splitext(img_name)[0], output_images_path,
                                      model_params, f_factor_path, final_shape_addition)
        else:
            raise Exception('invalid hdr file format: %s' % os.path.splitext(img_name)[1])


def load_g_model(model_params, device, net_path):
    G_net = create_G_net(model_params, device, is_checkpoint=True, activation="relu", output_dim=1)
    checkpoint = torch.load(net_path, map_location=device)
    state_dict = checkpoint['modelG_state_dict']
    new_state_dict = state_dict
    G_net.load_state_dict(new_state_dict)
    G_net = set_parallel_net(G_net, device)
    G_net.to(device)
    G_net.eval()
    return G_net


def create_G_net(model_params, device_, is_checkpoint, activation, output_dim):
    layer_factor = get_layer_factor(model_params["con_operator"])
    if model_params["model"] != params.unet_network:
        assert 0, "Unsupported g model request: {}".format(model_params["model"])
    new_net = Generator.UNet(model_params["input_dim"], output_dim, model_params["last_layer"],
                             depth=model_params["depth"], layer_factor=layer_factor,
                             con_operator=model_params["con_operator"],
                             filters=model_params["filters"], bilinear=model_params["bilinear"],
                             network=model_params["model"], dilation=0,
                             to_crop=model_params["add_frame"], unet_norm=model_params["unet_norm"],
                             stretch_g=model_params["stretch_g"],
                             activation=activation, doubleConvTranspose=model_params["g_doubleConvTranspose"],
                             padding_mode=model_params["padding"],
                             convtranspose_kernel=model_params["convtranspose_kernel"],
                             up_mode=model_params["up_mode"]).to(device_)
    #return set_parallel_net(new_net, device_, is_checkpoint, "Generator")
    return new_net


def get_layer_factor(con_operator):
    if con_operator in params.layer_factor_2_operators:
        return 2
    elif con_operator in params.layer_factor_3_operators:
        return 3
    elif con_operator in params.layer_factor_4_operators:
        return 4
    else:
        assert 0, "Unsupported con_operator request: {}".format(con_operator)


#def set_parallel_net(net, device_, is_checkpoint, net_name):
def set_parallel_net(net, device_):
    # Handle multi-gpu if desired
    if (device_.type == 'cuda') and (torch.cuda.device_count() > 1):
        print("Using [%d] GPUs" % torch.cuda.device_count())
        net = nn.DataParallel(net, list(range(torch.cuda.device_count())))
    return net


default_params = {"model_path": "model_weights_retrain20220815",
                  "model_name": "11_08_lr15D_size268_D_[1,1,1]_pad_0_G_ssr_doubleConvT__d1.0_struct_1.0[1,1,1]__trans2_replicate__noframe__min_log_0.1hist_fit_",
                  "input_images_path": "input_images",
                  "f_factor_path": "lambda_data/input_images_lambdas_HDRSdataset.npy",
                  "output_path": "output",
                  "mean_hist_path": "lambda_data/ldr_avg_hist_900_images_20_bins.npy",
                  "lambda_output_path": "lambda_data",
                  "bins": 20}


def print_args(args):
    print("========== input arguments ==========")
    for arg in args.__dict__:
        print("%s [%s]" % (arg, args.__dict__[arg]))
    print()


if __name__ == '__main__':
    opt = get_args()
    run_trained_model(opt)
