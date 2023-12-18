import inspect
import os
import re
import sys

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.unet_multi_filters.Unet as Generator
import models.unet_multi_filters.Unet_singleFrame as GeneratorImg
import tranforms
from models import Discriminator
from utils import params, data_loader_util, hdr_image_util
import cv2


# ======================================
# =========== TRAIN RELATED ============
# ======================================
def weights_init(m):
    """custom weights initialization called on netG and netD"""
    classname = m.__class__.__name__
    if (classname.find('Conv2d') != -1 or classname.find('Linear') != -1) and hasattr(m, 'weight'):
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def weights_init_xavier(m):
    """custom weights initialization called on netG and netD"""
    classname = m.__class__.__name__
    if (classname.find('Conv2d') != -1 or classname.find('Linear') != -1) and hasattr(m, 'weight'):
        torch.nn.init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def set_parallel_net(net, device_, is_checkpoint, net_name, use_xaviar=False):
    # Handle multi-gpu if desired
    if (device_.type == 'cuda') and (torch.cuda.device_count() > 1):
        print("Using [%d] GPUs" % torch.cuda.device_count())
        net = nn.DataParallel(net, list(range(torch.cuda.device_count())))

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    if not is_checkpoint:
        if use_xaviar:
            net.apply(weights_init_xavier)
        else:
            net.apply(weights_init)
        print("Weights for " + net_name + " were initialized successfully")
    return net


def create_G_net(model, device_, is_checkpoint, input_dim_, last_layer, filters, con_operator, unet_depth_,
                 add_frame, unet_norm, stretch_g, activation, use_xaviar, output_dim,
                 g_doubleConvTranspose, bilinear, padding, convtranspose_kernel, up_mode):
    layer_factor = get_layer_factor(con_operator)
    if model != params.unet_network:
        assert 0, "Unsupported g model request: {}".format(model)
    new_net = Generator.UNet(input_dim_, output_dim, last_layer, depth=unet_depth_,
                             layer_factor=layer_factor, con_operator=con_operator, filters=filters,
                             bilinear=bilinear, network=model, dilation=0, to_crop=add_frame,
                             unet_norm=unet_norm, stretch_g=stretch_g,
                             activation=activation,
                             doubleConvTranspose=g_doubleConvTranspose, padding_mode=padding,
                             convtranspose_kernel=convtranspose_kernel, up_mode=up_mode).to(device_)
    return set_parallel_net(new_net, device_, is_checkpoint, "Generator", use_xaviar)
    
def create_G_net2(model, device_, is_checkpoint, input_dim_, last_layer, filters, con_operator, unet_depth_,
                 add_frame, unet_norm, stretch_g, activation, use_xaviar, output_dim,
                 g_doubleConvTranspose, bilinear, padding, convtranspose_kernel, up_mode):
    layer_factor = get_layer_factor(con_operator)
    if model != params.unet_network:
        assert 0, "Unsupported g model request: {}".format(model)
    new_net = GeneratorImg.UNet(input_dim_, output_dim, last_layer, depth=unet_depth_,
                             layer_factor=layer_factor, con_operator=con_operator, filters=filters,
                             bilinear=bilinear, network=model, dilation=0, to_crop=add_frame,
                             unet_norm=unet_norm, stretch_g=stretch_g,
                             activation=activation,
                             doubleConvTranspose=g_doubleConvTranspose, padding_mode=padding,
                             convtranspose_kernel=convtranspose_kernel, up_mode=up_mode).to(device_)
    return set_parallel_net(new_net, device_, is_checkpoint, "Generator", use_xaviar)


def create_D_net(input_dim_, down_dim, device_, is_checkpoint, norm, use_xaviar, d_model,
                 d_nlayers, last_activation, num_D, d_fully_connected, simpleD_maxpool, d_padding):
    if d_model == "original":
        new_net = Discriminator.Discriminator(params.input_size, input_dim_,
                                              down_dim, norm, last_activation,
                                              d_fully_connected, d_nlayers).to(device_)
    elif d_model == "patchD":
        new_net = Discriminator.NLayerDiscriminator(input_dim_, ndf=down_dim, n_layers=d_nlayers,
                                                    norm_layer=norm, last_activation=last_activation).to(device_)
    elif "multiLayerD" in d_model:
        new_net = Discriminator.MultiscaleDiscriminator(params.input_size, d_model, input_dim_, ndf=down_dim,
                                                        n_layers=d_nlayers,
                                                        norm_layer=norm, last_activation=last_activation,
                                                        num_D=num_D, d_fully_connected=d_fully_connected,
                                                        simpleD_maxpool=simpleD_maxpool, padding=d_padding).to(device_)
    elif d_model == "simpleD":
        new_net = Discriminator.SimpleDiscriminator(params.input_size, input_dim_,
                                                    down_dim, norm, last_activation, simpleD_maxpool, d_padding).to(
            device_)
    else:
        assert 0, "Unsupported d model request: {}".format(d_model)
    return set_parallel_net(new_net, device_, is_checkpoint, "Discriminator", use_xaviar)


def save_model(path, epoch, epoch_iter, output_dir, netG, optimizerG, netD, optimizerD):
    path = os.path.join(output_dir, path, "net_epoch" + str(epoch) + '_iter' + str(epoch_iter) + ".pth")
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save({
        'epoch': epoch,
        'modelD_state_dict': netD.state_dict(),
        'modelG_state_dict': netG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
    }, path)


def save_discriminator_model(path, epoch, output_dir, netD, optimizerD):
    path = os.path.join(output_dir, path, "net_epoch_" + str(epoch) + ".pth")
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save({
        'epoch': epoch,
        'modelD_state_dict': netD.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
    }, path)


def get_layer_factor(con_operator):
    if con_operator in params.layer_factor_2_operators:
        return 2
    elif con_operator in params.layer_factor_3_operators:
        return 3
    elif con_operator in params.layer_factor_4_operators:
        return 4
    else:
        assert 0, "Unsupported con_operator request: {}".format(con_operator)


# ======================================
# ============ TEST RELATED ============
# ========= USE TRAINED MODEL ==========
# ======================================
def run_model_on_path(model_params, device, cur_net_path, input_images_path, output_images_path,
                      f_factor_path, net_G, final_shape_addition):
    extensions = [".hdr", ".dng", ".exr", ".npy"]
    if not net_G:
        net_G = load_g_model(model_params, device, cur_net_path)
    print("model " + model_params["model"] + " was loaded successfully")
    names = os.listdir(input_images_path)
    for img_name in names:
        im_path = os.path.join(input_images_path, img_name)
        print("processing [%s]" % img_name)
        if os.path.splitext(img_name)[1] in extensions:
            run_model_on_single_image(net_G, im_path, device, os.path.splitext(img_name)[0], output_images_path,
                                      model_params, f_factor_path, final_shape_addition)
        else:
            print("%s in already exists" % (img_name))


def load_g_model(model_params, device, net_path):
    G_net = create_G_net(model_params["model"], device, True, model_params["input_dim"],
                         model_params["last_layer"],
                         model_params["filters"], model_params["con_operator"], model_params["depth"],
                         model_params["add_frame"], model_params["unet_norm"], model_params["stretch_g"],
                         "relu", use_xaviar=False, output_dim=1,
                         g_doubleConvTranspose=model_params["g_doubleConvTranspose"],
                         bilinear=model_params["bilinear"],
                         padding=model_params["padding"],
                         convtranspose_kernel=model_params["convtranspose_kernel"],
                         up_mode=model_params["up_mode"])
    checkpoint = torch.load(net_path, map_location=device)
    state_dict = checkpoint['modelG_state_dict']
    if "module" in list(state_dict.keys())[0]:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
    else:
        new_state_dict = state_dict
    G_net.load_state_dict(new_state_dict)
    G_net.to(device)
    G_net.eval()
    return G_net


def load_inference(im_path, f_factor_path, factor_coeff, device):
    data = np.load(f_factor_path, allow_pickle=True)[()]
    f_factor = data[os.path.splitext(os.path.basename(im_path))[0]] * 255 * factor_coeff
    rgb_img = hdr_image_util.read_hdr_image(im_path)
    rgb_img = hdr_image_util.reshape_image(rgb_img, train_reshape=False)
    rgb_img = tranforms.hdr_im_transform(rgb_img).to(device)
    # shift for exr format
    if rgb_img.min() < 0:
        rgb_img = rgb_img - rgb_img.min()
    gray_im = hdr_image_util.to_gray_tensor(rgb_img).to(device)
    gray_im = gray_im - gray_im.min()
    gray_im = torch.log10((gray_im / gray_im.max()) * f_factor + 1)
    gray_im = gray_im / gray_im.max()
    return rgb_img, gray_im, f_factor

def load_inference2(im_path, f_factor_path, factor_coeff, device):
    print(f_factor_path)
    data = np.load(f_factor_path, allow_pickle=True)[()]
    f_factor = data[os.path.splitext(os.path.basename(im_path))[0]] * 255 * factor_coeff
    rgb_img = hdr_image_util.read_hdr_image(im_path)
    #scale=1
    scale=4
    rgb_img = cv2.resize(rgb_img, (rgb_img.shape[1]//scale,rgb_img.shape[0]//scale))
    print('-------------------')
    print(rgb_img.shape)
    #rgb_img = hdr_image_util.reshape_image(rgb_img, train_reshape=False)
    #print(rgb_img.shape)
    rgb_img = tranforms.hdr_im_transform(rgb_img).to(device)
    print(rgb_img.shape)
    # shift for exr format
    if rgb_img.min() < 0:
        rgb_img = rgb_img - rgb_img.min()
    gray_im = hdr_image_util.to_gray_tensor(rgb_img).to(device)
    gray_im = gray_im - gray_im.min()
    gray_im = torch.log10((gray_im / gray_im.max()) * f_factor + 1)
    gray_im = gray_im / gray_im.max()
    return rgb_img, gray_im, f_factor

def load_inference_testvideo(im_path, f_factor_path, factor_coeff, device):
    print(f_factor_path)
    #f_factor_path = "lambda_data/input_images_lambdas_HDRSdataset.npy"
    #f_factor_path = "lambda_data/input_images_lambdas_Kalantari13_1.npy"
    data = np.load(f_factor_path, allow_pickle=True)[()]
    #f_factor = data[os.path.splitext(os.path.basename(im_path))[0]] * 255 * factor_coeff
    f_factor = data[im_path.split('/')[-2]] * 255 * factor_coeff
    rgb_img = hdr_image_util.read_hdr_image(im_path)
    print('-------------------')
    print(rgb_img.shape)
    #rgb_img = hdr_image_util.reshape_image(rgb_img, train_reshape=False)
    #print(rgb_img.shape)
    rgb_img = tranforms.hdr_im_transform(rgb_img).to(device)
    print(rgb_img.shape)
    # shift for exr format
    if rgb_img.min() < 0:
        rgb_img = rgb_img - rgb_img.min()
    gray_im = hdr_image_util.to_gray_tensor(rgb_img).to(device)
    gray_im = gray_im - gray_im.min()
    gray_im = torch.log10((gray_im / gray_im.max()) * f_factor + 1)
    gray_im = gray_im / gray_im.max()
    return rgb_img, gray_im, f_factor


def run_model_on_single_image(G_net, im_path, device, im_name, output_path, model_params,
                              f_factor_path, final_shape_addition):
    rgb_img, gray_im_log, f_factor = load_inference(im_path, f_factor_path, model_params["factor_coeff"], device)
    rgb_img, diffY, diffX = data_loader_util.resize_im(rgb_img, model_params["add_frame"], final_shape_addition)
    gray_im_log, diffY, diffX = data_loader_util.resize_im(gray_im_log, model_params["add_frame"], final_shape_addition)
    with torch.no_grad():
        fake = G_net(gray_im_log.unsqueeze(0), apply_crop=model_params["add_frame"], diffY=diffY, diffX=diffX)
    max_p = np.percentile(fake.cpu().numpy(), 99.5)
    min_p = np.percentile(fake.cpu().numpy(), 0.5)
    # max_p = np.percentile(fake.cpu().numpy(), 100)
    # min_p = np.percentile(fake.cpu().numpy(), 0.0001)
    fake2 = fake.clamp(min_p, max_p)
    fake_im_gray_stretch = (fake2 - fake2.min()) / (fake2.max() - fake2.min())
    fake_im_color2 = hdr_image_util.back_to_color_tensor(rgb_img, fake_im_gray_stretch[0], device)
    h, w = fake_im_color2.shape[1], fake_im_color2.shape[2]
    im_max = fake_im_color2.max()
    fake_im_color2 = F.interpolate(fake_im_color2.unsqueeze(dim=0), size=(h - diffY, w - diffX),
                                   mode='bicubic',
                                   align_corners=False).squeeze(dim=0).clamp(min=0, max=im_max)
    print(fake.shape)
    #hdr_image_util.save_gray_tensor_as_numpy_stretch(fake[0]*100, output_path,
    #                                                 im_name + "_fake_clamp_and_stretch")
    #hdr_image_util.save_gray_tensor_as_numpy_stretch(fake[0], output_path,
    #                                                 im_name + "_fake_clamp_and_stretch")
    hdr_image_util.save_gray_tensor_as_numpy_stretch(fake_im_color2, output_path,
                                                     im_name + "_fake_clamp_and_stretch")

def run_model_on_single_image2(G_net, im_path, device, im_name, output_path, model_params,
                              f_factor_path, final_shape_addition):
    rgb_img, gray_im_log, f_factor = load_inference2(im_path, f_factor_path, model_params["factor_coeff"], device)
    #print('-------------------')
    print(rgb_img.shape)
    print(gray_im_log.shape)
    rgb_img, diffY, diffX = data_loader_util.resize_im(rgb_img, model_params["add_frame"], final_shape_addition)
    gray_im_log, diffY, diffX = data_loader_util.resize_im(gray_im_log, model_params["add_frame"], final_shape_addition)
    print(gray_im_log.shape)
    c, h, w = gray_im_log.size()
    fake = test_big_size_image2(input_data=gray_im_log.unsqueeze(0), model=G_net, apply_crop=model_params["add_frame"], diffY=diffY, diffX=diffX, patch_h = 256, patch_w = 256, patch_h_overlap = 64, patch_w_overlap = 64) # For 1/4 resolution, i.e. scale=4
    #fake = test_big_size_image2(input_data=gray_im_log.unsqueeze(0), model=G_net, apply_crop=model_params["add_frame"], diffY=diffY, diffX=diffX, patch_h = 256, patch_w = 256, patch_h_overlap = 192, patch_w_overlap = 192) # For full resolution, i.e. scale=1
    #with torch.no_grad():
    #    fake = G_net(gray_im_log.unsqueeze(0), apply_crop=model_params["add_frame"], diffY=diffY, diffX=diffX)
    '''fake = torch.zeros((1,c,h,w)).cuda()
    #def inference(fake, gray_im_log):
    #    _, _, h, w = fake.size()
    #    fake[:,:,:h//2,:w//2] = G_net(gray_im_log[:,:h//2+256,:w//2+256].unsqueeze(0), apply_crop=model_params["add_frame"], diffY=diffY, diffX=diffX)[:,:,:-256,:-256]
    #    fake[:,:,h//2:,:w//2] = G_net(gray_im_log[:,h//2-256:,:w//2+256].unsqueeze(0), apply_crop=model_params["add_frame"], diffY=diffY, diffX=diffX)[:,:,256:,:-256]
    #    fake[:,:,:h//2,w//2:] = G_net(gray_im_log[:,:h//2+256,w//2-256:].unsqueeze(0), apply_crop=model_params["add_frame"], diffY=diffY, diffX=diffX)[:,:,:-256,256:]
    #    fake[:,:,h//2:,w//2:] = G_net(gray_im_log[:,h//2-256:,w//2-256:].unsqueeze(0), apply_crop=model_params["add_frame"], diffY=diffY, diffX=diffX)[:,:,256:,256:]
    with torch.no_grad():
        #fake = G_net(gray_im_log.unsqueeze(0), apply_crop=model_params["add_frame"], diffY=diffY, diffX=diffX)
        #fake[:,:,:h//2,:w//2] = G_net(gray_im_log[:,:h//2+256,:w//2+256].unsqueeze(0), apply_crop=model_params["add_frame"], diffY=diffY, diffX=diffX)[:,:,:-256,:-256]
        #fake[:,:,h//2:,:w//2] = G_net(gray_im_log[:,h//2-256:,:w//2+256].unsqueeze(0), apply_crop=model_params["add_frame"], diffY=diffY, diffX=diffX)[:,:,256:,:-256]
        #fake[:,:,:h//2,w//2:] = G_net(gray_im_log[:,:h//2+256,w//2-256:].unsqueeze(0), apply_crop=model_params["add_frame"], diffY=diffY, diffX=diffX)[:,:,:-256,256:]
        #fake[:,:,h//2:,w//2:] = G_net(gray_im_log[:,h//2-256:,w//2-256:].unsqueeze(0), apply_crop=model_params["add_frame"], diffY=diffY, diffX=diffX)[:,:,256:,256:]
        ps = 1024
        m = h//ps
        if h%ps!=0:
            m += 1
        n = w//ps
        if w%ps!=0:
            n += 1
        overlap = 128
        for i in range(m):
            for j in range(n):
                h_start = i*ps
                h_end = h_start+ps
                if h_end>h:
                    h_end = h
                    h_start = h_end-ps
                w_start = j*ps
                w_end = w_start+ps
                if w_end>w:
                    w_end = w
                    w_start = w_end-ps
                if i==0:
                    h_start_new = h_start
                    overlap_h_start = 0
                else:
                    h_start_new = h_start-overlap
                    overlap_h_start = overlap
                if i==(m-1):
                    h_end_new = h_end
                    overlap_h_end = 0
                else:
                    h_end_new = h_end+overlap
                    overlap_h_end = overlap
                if j==0:
                    w_start_new = w_start
                    overlap_w_start = 0
                else:
                    w_start_new = w_start-overlap
                    overlap_w_start = overlap
                if j==(n-1):
                    w_end_new = w_end
                    overlap_w_end = 0
                else:
                    w_end_new = w_end+overlap
                    overlap_w_end = overlap
                if h_end_new>h:
                    overlap_h_end = overlap-(h_end_new-h)
                    h_end_new = h
                if w_end_new>w:
                    overlap_w_end = overlap-(w_end_new-w)
                    w_end_new = w
                #print(fake[:,:,h_start:h_end,w_start:w_end].shape)
                #print(gray_im_log[:,h_start_new:h_end_new,w_start_new:w_end_new].shape)
                #print(overlap_h_start)
                #print(overlap_h_end)
                #print(overlap_w_start)
                #print(overlap_w_end)
                #print(j)
                #print(n)
                if i==(m-1):
                    if j==(n-1):
                        fake[:,:,h_start:h_end,w_start:w_end] = G_net(gray_im_log[:,h_start_new:h_end_new,w_start_new:w_end_new].unsqueeze(0), apply_crop=model_params["add_frame"], diffY=diffY, diffX=diffX)[:,:,overlap_h_start:,overlap_w_start:]
                    else:
                        fake[:,:,h_start:h_end,w_start:w_end] = G_net(gray_im_log[:,h_start_new:h_end_new,w_start_new:w_end_new].unsqueeze(0), apply_crop=model_params["add_frame"], diffY=diffY, diffX=diffX)[:,:,overlap_h_start:,overlap_w_start:-overlap_w_end]
                else:
                    if j==(n-1):
                        fake[:,:,h_start:h_end,w_start:w_end] = G_net(gray_im_log[:,h_start_new:h_end_new,w_start_new:w_end_new].unsqueeze(0), apply_crop=model_params["add_frame"], diffY=diffY, diffX=diffX)[:,:,overlap_h_start:-overlap_h_end,overlap_w_start:]
                    else:
                        fake[:,:,h_start:h_end,w_start:w_end] = G_net(gray_im_log[:,h_start_new:h_end_new,w_start_new:w_end_new].unsqueeze(0), apply_crop=model_params["add_frame"], diffY=diffY, diffX=diffX)[:,:,overlap_h_start:-overlap_h_end,overlap_w_start:-overlap_w_end]
    #fake = gray_im_log'''
    max_p = np.percentile(fake.cpu().numpy(), 99.5)
    min_p = np.percentile(fake.cpu().numpy(), 0.5)
    # max_p = np.percentile(fake.cpu().numpy(), 100)
    # min_p = np.percentile(fake.cpu().numpy(), 0.0001)
    fake2 = fake.clamp(min_p, max_p)
    fake_im_gray_stretch = (fake2 - fake2.min()) / (fake2.max() - fake2.min())
    fake_im_color2 = hdr_image_util.back_to_color_tensor(rgb_img, fake_im_gray_stretch[0], device)
    h, w = fake_im_color2.shape[1], fake_im_color2.shape[2]
    im_max = fake_im_color2.max()
    #fake_im_color2 = F.interpolate(fake_im_color2.unsqueeze(dim=0), size=(h - diffY, w - diffX),
    #                               mode='bicubic',
    #                               align_corners=False).squeeze(dim=0).clamp(min=0, max=im_max)
    fake_im_color2 = fake_im_color2[:,diffY // 2:-(diffY - diffY // 2),diffX // 2:-(diffX - diffX // 2)]
    fake_im_color2 = fake_im_color2.clamp(min=0, max=im_max)
    print(fake_im_color2.shape)
    hdr_image_util.save_gray_tensor_as_numpy_stretch(fake_im_color2, output_path,
                                                     im_name + "_UnCLTMO")
    #hdr_image_util.save_gray_tensor_as_numpy_stretch(gray_im_log, output_path,
    #                                                 im_name + "_UnCLTMO")

def test_big_size_image2(input_data, model, apply_crop, diffY, diffX, patch_h = 256, patch_w = 256, patch_h_overlap = 64, patch_w_overlap = 64):

    H = input_data.shape[2]
    W = input_data.shape[3]
    
    test_result = torch.zeros_like(input_data).cuda()
    #t0 = time.clock()
    h_index = 1
    while (patch_h*h_index-patch_h_overlap*(h_index-1)) < H:
        test_horizontal_result = torch.zeros((input_data.shape[0],1,patch_h,W)).cuda()
        h_begin = patch_h*(h_index-1)-patch_h_overlap*(h_index-1)
        h_end = patch_h*h_index-patch_h_overlap*(h_index-1) 
        w_index = 1
        while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
            w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
            w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
            test_patch = input_data[:,:,h_begin:h_end,w_begin:w_end]               
            with torch.no_grad():
                output_patch, _ = model(test_patch, apply_crop=apply_crop, diffY=diffY, diffX=diffX)
            if w_index == 1:
                #print(output_patch.shape)
                #print(test_horizontal_result[:,:,w_begin:w_end,:].shape)
                test_horizontal_result[:,:,:,w_begin:w_end] = output_patch
            else:
                for i in range(patch_w_overlap):
                    test_horizontal_result[:,:,:,w_begin+i] = test_horizontal_result[:,:,:,w_begin+i]*(patch_w_overlap-1-i)/(patch_w_overlap-1)+output_patch[:,:,:,i]*i/(patch_w_overlap-1)
                test_horizontal_result[:,:,:,w_begin+patch_w_overlap:w_end] = output_patch[:,:,:,patch_w_overlap:]
            w_index += 1
    
        test_patch = input_data[:,:,h_begin:h_end,-patch_w:]
        with torch.no_grad():
            output_patch, _ = model(test_patch, apply_crop=apply_crop, diffY=diffY, diffX=diffX)
        last_range = w_end-(W-patch_w)
        for i in range(last_range):
            test_horizontal_result[:,:,:,W-patch_w+i] = test_horizontal_result[:,:,:,W-patch_w+i]*(last_range-1-i)/(last_range-1)+output_patch[:,:,:,i]*i/(last_range-1)
        test_horizontal_result[:,:,:,w_end:] = output_patch[:,:,:,last_range:]

        if h_index == 1:
            test_result[:,:,h_begin:h_end,:] = test_horizontal_result
        else:
            for i in range(patch_h_overlap):
                test_result[:,:,h_begin+i,:] = test_result[:,:,h_begin+i,:]*(patch_h_overlap-1-i)/(patch_h_overlap-1)+test_horizontal_result[:,:,i,:]*i/(patch_h_overlap-1)
            test_result[:,:,h_begin+patch_h_overlap:h_end,:] = test_horizontal_result[:,:,patch_h_overlap:,:] 
        h_index += 1

    test_horizontal_result = torch.zeros((input_data.shape[0],1,patch_h,W)).cuda()
    w_index = 1
    while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
        w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
        w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
        test_patch = input_data[:,:,-patch_h:,w_begin:w_end]
        with torch.no_grad():
            output_patch, _ = model(test_patch, apply_crop=apply_crop, diffY=diffY, diffX=diffX)
        if w_index == 1:
            test_horizontal_result[:,:,:,w_begin:w_end] = output_patch
        else:
            for i in range(patch_w_overlap):
                test_horizontal_result[:,:,:,w_begin+i] = test_horizontal_result[:,:,:,w_begin+i]*(patch_w_overlap-1-i)/(patch_w_overlap-1)+output_patch[:,:,:,i]*i/(patch_w_overlap-1)
            test_horizontal_result[:,:,:,w_begin+patch_w_overlap:w_end] = output_patch[:,:,:,patch_w_overlap:]
        w_index += 1

    test_patch = input_data[:,:,-patch_h:,-patch_w:]
    with torch.no_grad():
        output_patch, _ = model(test_patch, apply_crop=apply_crop, diffY=diffY, diffX=diffX)
    last_range = w_end-(W-patch_w)
    for i in range(last_range):
        test_horizontal_result[:,:,:,W-patch_w+i] = test_horizontal_result[:,:,:,W-patch_w+i]*(last_range-1-i)/(last_range-1)+output_patch[:,:,:,i]*i/(last_range-1) 
    test_horizontal_result[:,:,:,w_end:] = output_patch[:,:,:,last_range:] 

    last_last_range = h_end-(H-patch_h)
    for i in range(last_last_range):
        test_result[:,:,H-patch_w+i,:] = test_result[:,:,H-patch_w+i,:]*(last_last_range-1-i)/(last_last_range-1)+test_horizontal_result[:,:,i,:]*i/(last_last_range-1)
    test_result[:,:,h_end:,:] = test_horizontal_result[:,:,last_last_range:,:]
   
    #t1 = time.clock()
    #print('Total running time: %s s' % (str(t1 - t0)))

    return test_result

def test_big_size_image(input_data, model, apply_crop, diffY, diffX, patch_h = 256, patch_w = 256, patch_h_overlap = 64, patch_w_overlap = 64):

    H = input_data.shape[3]
    W = input_data.shape[4]
    
    test_result = torch.zeros_like(input_data).cuda()
    #t0 = time.clock()
    h_index = 1
    while (patch_h*h_index-patch_h_overlap*(h_index-1)) < H:
        test_horizontal_result = torch.zeros((input_data.shape[0],input_data.shape[1],1,patch_h,W)).cuda()
        h_begin = patch_h*(h_index-1)-patch_h_overlap*(h_index-1)
        h_end = patch_h*h_index-patch_h_overlap*(h_index-1) 
        w_index = 1
        while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
            w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
            w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
            test_patch = input_data[:,:,:,h_begin:h_end,w_begin:w_end]               
            with torch.no_grad():
                output_patch, _ = model(test_patch, apply_crop=apply_crop, diffY=diffY, diffX=diffX)
            if w_index == 1:
                #print(output_patch.shape)
                #print(test_horizontal_result[:,:,w_begin:w_end,:].shape)
                test_horizontal_result[:,:,:,:,w_begin:w_end] = output_patch
            else:
                for i in range(patch_w_overlap):
                    test_horizontal_result[:,:,:,:,w_begin+i] = test_horizontal_result[:,:,:,:,w_begin+i]*(patch_w_overlap-1-i)/(patch_w_overlap-1)+output_patch[:,:,:,:,i]*i/(patch_w_overlap-1)
                test_horizontal_result[:,:,:,:,w_begin+patch_w_overlap:w_end] = output_patch[:,:,:,:,patch_w_overlap:]
            w_index += 1
    
        test_patch = input_data[:,:,:,h_begin:h_end,-patch_w:]
        with torch.no_grad():
            output_patch, _ = model(test_patch, apply_crop=apply_crop, diffY=diffY, diffX=diffX)
        last_range = w_end-(W-patch_w)
        for i in range(last_range):
            test_horizontal_result[:,:,:,:,W-patch_w+i] = test_horizontal_result[:,:,:,:,W-patch_w+i]*(last_range-1-i)/(last_range-1)+output_patch[:,:,:,:,i]*i/(last_range-1)
        test_horizontal_result[:,:,:,:,w_end:] = output_patch[:,:,:,:,last_range:]

        if h_index == 1:
            test_result[:,:,:,h_begin:h_end,:] = test_horizontal_result
        else:
            for i in range(patch_h_overlap):
                test_result[:,:,:,h_begin+i,:] = test_result[:,:,:,h_begin+i,:]*(patch_h_overlap-1-i)/(patch_h_overlap-1)+test_horizontal_result[:,:,:,i,:]*i/(patch_h_overlap-1)
            test_result[:,:,:,h_begin+patch_h_overlap:h_end,:] = test_horizontal_result[:,:,:,patch_h_overlap:,:] 
        h_index += 1

    test_horizontal_result = torch.zeros((input_data.shape[0],input_data.shape[1],1,patch_h,W)).cuda()
    w_index = 1
    while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
        w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
        w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
        test_patch = input_data[:,:,:,-patch_h:,w_begin:w_end]
        with torch.no_grad():
            output_patch, _ = model(test_patch, apply_crop=apply_crop, diffY=diffY, diffX=diffX)
        if w_index == 1:
            test_horizontal_result[:,:,:,:,w_begin:w_end] = output_patch
        else:
            for i in range(patch_w_overlap):
                test_horizontal_result[:,:,:,:,w_begin+i] = test_horizontal_result[:,:,:,:,w_begin+i]*(patch_w_overlap-1-i)/(patch_w_overlap-1)+output_patch[:,:,:,:,i]*i/(patch_w_overlap-1)
            test_horizontal_result[:,:,:,:,w_begin+patch_w_overlap:w_end] = output_patch[:,:,:,:,patch_w_overlap:]
        w_index += 1

    test_patch = input_data[:,:,:,-patch_h:,-patch_w:]
    with torch.no_grad():
        output_patch, _ = model(test_patch, apply_crop=apply_crop, diffY=diffY, diffX=diffX)
    last_range = w_end-(W-patch_w)
    for i in range(last_range):
        test_horizontal_result[:,:,:,:,W-patch_w+i] = test_horizontal_result[:,:,:,:,W-patch_w+i]*(last_range-1-i)/(last_range-1)+output_patch[:,:,:,:,i]*i/(last_range-1) 
    test_horizontal_result[:,:,:,:,w_end:] = output_patch[:,:,:,:,last_range:] 

    last_last_range = h_end-(H-patch_h)
    for i in range(last_last_range):
        test_result[:,:,:,H-patch_w+i,:] = test_result[:,:,:,H-patch_w+i,:]*(last_last_range-1-i)/(last_last_range-1)+test_horizontal_result[:,:,:,i,:]*i/(last_last_range-1)
    test_result[:,:,:,h_end:,:] = test_horizontal_result[:,:,:,last_last_range:,:]
   
    #t1 = time.clock()
    #print('Total running time: %s s' % (str(t1 - t0)))

    return test_result

def run_model_on_video(G_net, im_paths, device, im_names, output_path, model_params,
                              f_factor_path, final_shape_addition):
    gray_im_logs = []
    rgb_imgs = []
    #gray_ims_log = []
    for im_path in im_paths:
        rgb_img, gray_im_log, f_factor = load_inference_testvideo(im_path, f_factor_path, model_params["factor_coeff"], device)
        #print('-------------------')
        print(rgb_img.shape)
        print(gray_im_log.shape)
        rgb_img, diffY, diffX = data_loader_util.resize_im(rgb_img, model_params["add_frame"], final_shape_addition)
        gray_im_log, diffY, diffX = data_loader_util.resize_im(gray_im_log, model_params["add_frame"], final_shape_addition)
        print(gray_im_log.shape)
        gray_im_logs.append(gray_im_log.unsqueeze(0).unsqueeze(0))
        rgb_imgs.append(rgb_img)
        #gray_ims_log.append(gray_im_log)
    gray_im_logs = torch.cat(gray_im_logs, 1)
    
    c, h, w = gray_im_log.size()
    fakes = test_big_size_image(input_data=gray_im_logs, model=G_net, apply_crop=model_params["add_frame"], diffY=diffY, diffX=diffX, patch_h = 256, patch_w = 256, patch_h_overlap = 64, patch_w_overlap = 64)
    #fakes = test_big_size_image(input_data=gray_im_logs, model=G_net, apply_crop=model_params["add_frame"], diffY=diffY, diffX=diffX, patch_h = 256, patch_w = 256, patch_h_overlap = 128, patch_w_overlap = 128)

    for i in range(len(im_paths)):
        fake = fakes[:,i,:,:,:]
        max_p = np.percentile(fake.cpu().numpy(), 99.5)
        min_p = np.percentile(fake.cpu().numpy(), 0.5)
        # max_p = np.percentile(fake.cpu().numpy(), 100)
        # min_p = np.percentile(fake.cpu().numpy(), 0.0001)
        fake2 = fake.clamp(min_p, max_p)
        fake_im_gray_stretch = (fake2 - fake2.min()) / (fake2.max() - fake2.min())
        fake_im_color2 = hdr_image_util.back_to_color_tensor(rgb_imgs[i], fake_im_gray_stretch[0], device)
        h, w = fake_im_color2.shape[1], fake_im_color2.shape[2]
        im_max = fake_im_color2.max()
        #fake_im_color2 = F.interpolate(fake_im_color2.unsqueeze(dim=0), size=(h - diffY, w - diffX),
        #                               mode='bicubic',
        #                               align_corners=False).squeeze(dim=0).clamp(min=0, max=im_max)
        fake_im_color2 = fake_im_color2[:,diffY // 2:-(diffY - diffY // 2),diffX // 2:-(diffX - diffX // 2)]
        fake_im_color2 = fake_im_color2.clamp(min=0, max=im_max)
        print(fake_im_color2.shape)
        save_path = os.path.join(output_path, im_paths[i].split('/')[-2])
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        hdr_image_util.save_gray_tensor_as_numpy_stretch(fake_im_color2, save_path,
                                                         im_names[i] + "_UnCLTMO")
        #gray = gray_ims_log[i].permute(1,2,0).cpu().numpy()
        #imageio.imwrite(os.path.join(save_path, im_names[i] + "_hdr_y_gaamma.png"), np.uint8((gray/gray.max())**(1/2.2)*255), format='PNG-FI')
    #hdr_image_util.save_gray_tensor_as_numpy_stretch(gray_im_log, output_path,
    #                                                 im_name + "_UnpairedHDRTMO")
    
    
# ========================
# ====== GET PARAMS ======
# ========================
def get_model_params(model_name, train_settings_path="none"):
    model_params = {"model_name": model_name, "model": params.unet_network,
                    "filters": 32, "depth": 4,
                    "factorised_data": True,
                    "input_loader": None,
                    "gamma_log": 10, "unet_norm": 'none', "input_dim": 1, "clip": False}

    params_dict = {"add_frame": get_frame,
                   "last_layer": get_last_layer,
                   "stretch_g": get_stretch_g,
                   "con_operator": get_con_operator,
                   "g_doubleConvTranspose": get_g_doubleConvTranspose,
                   "factor_coeff": get_factor_coeff,
                   "use_new_f": get_use_new_f,
                   "data_trc": get_data_trc,
                   "d_weight_mul_mode": get_manualD,
                   "manual_d_training": get_manual_d_training,
                   "use_contrast_ratio_f": get_use_contrast_ratio_f,
                   "final_shape_addition": get_use_contrast_ratio_f,
                   "bilinear": get_use_contrast_ratio_f,
                   "padding": get_use_contrast_ratio_f,
                   "up_mode": get_use_contrast_ratio_f,
                   "convtranspose_kernel": get_use_contrast_ratio_f}

    print(train_settings_path)
    if os.path.exists(train_settings_path):
        train_settings = np.load(train_settings_path, allow_pickle=True)[()]
        for param in params_dict.keys():
            model_params[param] = train_settings[param]

    if model_params["manual_d_training"]:
        model_params["input_dim"] = 2
    return model_params

# TODO: change to abs path
def get_f_factor_path(name, data_gamma_log, use_new_f, use_hist_fit):
    path_dict_prev_f = {
        "test_source": {10: "/Users/yaelvinker/PycharmProjects/lab/utils/test_factors.npy"},
        "open_exr_exr_format": {1: "/cs/snapless/raananf/yael_vinker/data/open_exr_source/exr_factors_mean150.npy",
                                2: "/cs/snapless/raananf/yael_vinker/data/open_exr_source/exr_factors_mean90.npy",
                                10: "/cs/snapless/raananf/yael_vinker/data/open_exr_source/exr_factors_mean160.npy"},
        "hdr_test": {10: "/Users/yaelvinker/PycharmProjects/lab/utils/test_factors.npy"},
        "npy_pth": {1: "/cs/snapless/raananf/yael_vinker/data/dng_data_fid.npy",
                    2: "/cs/snapless/raananf/yael_vinker/data/dng_data_fid.npy",
                    10: "/cs/snapless/raananf/yael_vinker/data/dng_data_fid.npy"}}
    # TODO : update these dirs when apply the new f on the test data
    path_dict_new = {"test_source": "/Users/yaelvinker/PycharmProjects/lab/utils/test_factors.npy",
                     "open_exr_exr_format": "/cs/snapless/raananf/yael_vinker/data/open_exr_source/exr_newf.npy",
                     "hdr_test": "/Users/yaelvinker/PycharmProjects/lab/utils/test_factors.npy",
                     "npy_pth": "/cs/snapless/raananf/yael_vinker/data/dng_fid_newf.npy"}
    path_dict_hist_fit = {
        "test_source": "/Users/yaelvinker/Documents/university/lab/lum_hist_re/valid_hist_dict_20_bins.npy",
        "open_exr_exr_format": "/cs/labs/raananf/yael_vinker/data/new_lum_est_hist/train_valid/exr_hist_dict_20_bins.npy",
        "hdr_test": "/cs/labs/raananf/yael_vinker/data/new_lum_est_hist/train_valid/valid_hist_dict_20_bins.npy",
        "npy_pth": "/cs/labs/raananf/yael_vinker/data/new_lum_est_hist/fix_lum_hist/dng_hist_20_bins_all_fix.npy"}

    if use_hist_fit:
        return path_dict_hist_fit[name]
    if use_new_f:
        return path_dict_new[name]
    if not data_gamma_log and name == "npy_pth":
        return path_dict_prev_f[name][1]
    return path_dict_prev_f[name][data_gamma_log]


def get_hdr_source_path(name):
    path_dict = {
        "test_source": "/Users/yaelvinker/PycharmProjects/lab/utils/folders/temp_data",
        "new_source": "/Users/yaelvinker/PycharmProjects/lab/utils/temp_data",
        "open_exr_hdr_format": "/cs/snapless/raananf/yael_vinker/data/open_exr_source/open_exr_fixed_size",
        "open_exr_exr_format": "/cs/snapless/raananf/yael_vinker/data/open_exr_source/exr_format_fixed_size",
        "open_exr_exr_format_original_size": "/cs/labs/raananf/yael_vinker/data/quality_assesment/from_openEXR_data",
        "hdr_test": "/cs/labs/raananf/yael_vinker/data/test/tmqi_test_hdr",
        "npy_pth": "/cs/snapless/raananf/yael_vinker/data/oct_fid_npy"}
    return path_dict[name]


def get_con_operator(model_name):
    if params.con_op_short["square_and_square_root_manual_d"] in model_name:
        return "square_and_square_root_manual_d"
    if params.con_op_short["square_and_square_root"] in model_name:
        return "square_and_square_root"
    if params.con_op_short["square_root"] in model_name:
        return "square_root"
    if params.con_op_short["original_unet"] in model_name:
        return "original_unet"
    if params.con_op_short["square"] in model_name:
        return "square"


def get_stretch_g(model_name):
    if "batchMax" in model_name:
        return "batchMax"
    elif "instanceMinMax" in model_name:
        return "instanceMinMax"
    return "none"


def get_last_layer(model_name):
    if "msig" in model_name:
        return "msig"
    else:
        return "sigmoid"


def get_frame(model_name):
    if "noframe" in model_name:
        return False
    return True


def get_clip(model_name):
    return "clip" in model_name


def get_factor_coeff(model_name):
    items = re.findall("min_log_(\d+\.*\d+)", model_name)
    return float(items[0])


def get_data_trc(model_name):
    items = re.findall("DATA_(\w*)_\d+\.*\d+", model_name)
    return items[0]


def get_manualD(model_name):
    if "manualD_single" in model_name:
        return "single"
    if "manualD_double" in model_name:
        return "double"
    return "none"


def get_manual_d_training(model_name):
    if "manualD" in model_name:
        return True
    return False


def get_use_new_f(model_name):
    if "new_f" in model_name:
        return True
    return False


def get_use_contrast_ratio_f(model_name):
    if "contrast_ratio_f_" in model_name:
        return True
    return False


def get_g_doubleConvTranspose(model_name):
    if "doubleConvT" in model_name:
        return True
    return False


def get_apply_wind_norm(model_name):
    if "apply_wind_norm" in model_name:
        return True
    else:
        return False


def get_gamma_log(model_name):
    if "data_10" in model_name:
        return 10
    if "data_2" in model_name:
        return 2
    if "data_1" in model_name:
        return 1
    return 10


def get_std_norm_factor(model_name):
    if "apply_wind_norm" in model_name:
        if "factor_0.8" in model_name:
            return 0.8
        if "factor_0.9" in model_name:
            return 0.9
        if "factor_1" in model_name:
            return 1
    else:
        return None


def get_clip_from_name(model_name):
    if "clip_False" in model_name:
        return False
    if "clip_True" in model_name:
        return True
    else:
        assert 0, "Unsupported clip def"


def get_normalise_from_name(model_name):
    if "normalise_False" in model_name:
        return False
    if "normalise_True" in model_name:
        return True
    else:
        assert 0, "Unsupported normalise def"


def is_factorised_data(model_name):
    if "use_f_False" in model_name:
        return False
    if "use_f_True" in model_name:
        return True
    else:
        assert 0, "Unsupported factorised def"
