import os
import pathlib

import imageio
import matplotlib.pyplot as plt
import numpy as np
import skimage
import torch
from utils import params


# ===========================
# ====== IMAGE PRINTER ======
# ===========================
def print_image_details(im, title):
    print(title)
    print("shape : ", im.shape)
    print("max : ", np.max(im), "  min : ", np.min(im), "mean : ", np.mean(im))
    print("type : ", im.dtype)
    print("unique values : ", np.unique(im).shape[0])
    print()


def print_tensor_details(im, title):
    print(title)
    print("shape : ", im.shape)
    print("max : ", im.max(), "  min : ", im.min(), "mean : ", im.mean())
    print("type : ", im.dtype)
    print()


# ==========================
# ====== IMAGE READER ======
# ==========================
def read_hdr_image(path):
    path_lib_path = pathlib.Path(path)
    file_extension = os.path.splitext(path)[1]
    if file_extension == ".hdr":
        im = imageio.imread(path_lib_path, format="HDR-FI").astype('float32')
        #im = imageio.imread(path_lib_path).astype('float32')
    elif file_extension == ".dng":
        im = imageio.imread(path_lib_path, format="RAW-FI").astype('float32')
    elif file_extension == ".exr":
        im = imageio.imread(path_lib_path, format="EXR-FI").astype('float32')
    elif file_extension == ".npy":
        im = np.load(path, allow_pickle=True).astype('float32')
        #im = np.load(path, allow_pickle=True)[()]
        #if not isinstance(im, np.ndarray):
        #    im = im["display_image"].permute(1, 2, 0).detach().cpu().numpy()
    else:
        print(path)
        raise Exception('invalid hdr file format: {}'.format(file_extension))
    return im


def read_ldr_image(path):
    path = pathlib.Path(path)
    im_origin = imageio.imread(path)
    im = im_origin / 255
    return im


def read_ldr_image_original_range(path):
    path = pathlib.Path(path)
    im_origin = imageio.imread(path)
    return im_origin


# ==============================
# ====== IMAGE MANIPULATE ======
# ==============================
def to_gray(im):
    return np.dot(im[..., :3], [0.299, 0.587, 0.114]).astype('float32')


def to_gray_tensor(rgb_tensor):
    r_image = rgb_tensor[0]
    g_image = rgb_tensor[1]
    b_image = rgb_tensor[2]
    grayscale_image = (0.299 * r_image + 0.587 * g_image + 0.114 * b_image)
    grayscale_image = grayscale_image[None, :, :]
    return grayscale_image


def to_0_1_range(im):
    if np.max(im) - np.min(im) == 0:
        im = (im - np.min(im)) / (np.max(im) - np.min(im) + params.epsilon)
    else:
        im = (im - np.min(im)) / (np.max(im) - np.min(im))
    return im


def to_0_1_range_outlier(im):
    im_max = np.percentile(im, 99.0)
    # TODO: check this parameter
    # im_min = np.percentile(im, 1.0)
    im_min = np.percentile(im, 0.1)
    if np.max(im) - np.min(im) == 0:
        im = (im - im_min) / (im_max - im_min + params.epsilon)
    else:
        im = (im - im_min) / (im_max - im_min)
    return np.clip(im, 0, 1)


def to_minus1_1_range(im):
    return 2 * im - 1


def back_to_color2(im_hdr, fake):
    if np.min(im_hdr) < 0:
        im_hdr = im_hdr + np.abs(np.min(im_hdr))
    im_gray_ = to_gray(im_hdr)#np.sum(im_hdr, axis=2)
    norm_im = np.zeros(im_hdr.shape)
    norm_im[:, :, 0] = im_hdr[:, :, 0] / (im_gray_ + params.epsilon)
    norm_im[:, :, 1] = im_hdr[:, :, 1] / (im_gray_ + params.epsilon)
    norm_im[:, :, 2] = im_hdr[:, :, 2] / (im_gray_ + params.epsilon)
    norm_im = np.power(norm_im, 0.5)
    output_im = norm_im * fake
    return output_im


def back_to_color_tensor(im_hdr, fake, device):
    if im_hdr.min() < 0:
        im_hdr = im_hdr - im_hdr.min()
    im_gray_ = to_gray_tensor(im_hdr)#np.sum(im_hdr, axis=2)
    norm_im = torch.zeros(im_hdr.shape).to(device)
    norm_im[0, :, :] = im_hdr[0, :, :] / (im_gray_ + params.epsilon)
    norm_im[1, :, :] = im_hdr[1, :, :] / (im_gray_ + params.epsilon)
    norm_im[2, :, :] = im_hdr[2, :, :] / (im_gray_ + params.epsilon)
    norm_im = torch.pow(norm_im, 0.5)
    output_im = norm_im * fake
    return output_im


def reshape_im(im, new_y, new_x):
    return skimage.transform.resize(im, (new_y, new_x),
                                    mode='reflect', preserve_range=False,
                                    anti_aliasing=True, order=3).astype("float32")


def reshape_image(rgb_im, train_reshape):
    h, w = rgb_im.shape[0], rgb_im.shape[1]
    if train_reshape:
        rgb_im = skimage.transform.resize(rgb_im, (params.input_size, params.input_size),
                                          mode='reflect', preserve_range=False,
                                          anti_aliasing=True, order=3).astype("float32")
    else:
        if min(h, w) > 3000:
            rgb_im = skimage.transform.resize(rgb_im, (int(rgb_im.shape[0] / 4),
                                                       int(rgb_im.shape[1] / 4)),
                                              mode='reflect', preserve_range=False,
                                              anti_aliasing=True, order=3).astype("float32")
        elif min(h, w) > 2000:
            rgb_im = skimage.transform.resize(rgb_im, (int(rgb_im.shape[0] / 3),
                                                       int(rgb_im.shape[1] / 3)),
                                              mode='reflect', preserve_range=False,
                                              anti_aliasing=True, order=3).astype("float32")
    return rgb_im


def reshape_image_fixed_size(rgb_im):
    rgb_im = skimage.transform.resize(rgb_im, (1024,
                                               2048),
                                      mode='reflect', preserve_range=False,
                                      order=3).astype("float32")
    return rgb_im


def back_to_color_batch2(im_hdr_batch, fake_batch):
    b_size = im_hdr_batch.shape[0]
    output = []
    import time
    a = time.time()
    for i in range(b_size):
        im_hdr = im_hdr_batch[i].clone().permute(1, 2, 0).detach().cpu().numpy()
        fake = fake_batch[i].clone().permute(1, 2, 0).detach().cpu().numpy()
        norm_im = back_to_color2(im_hdr, fake)
        output.append(torch.from_numpy(norm_im.transpose((2, 0, 1))).float())
    print("time took back to color", time.time() - a)
    return torch.stack(output)


def back_to_color_batch_tensor(im_hdr_batch, fake_batch):
    b_size = im_hdr_batch.shape[0]
    output = []
    import time
    a = time.time()
    for i in range(b_size):
        im_hdr = im_hdr_batch[i]
        fake = fake_batch[i]
        norm_im = back_to_color_tensor(im_hdr, fake)
        output.append(norm_im)
    print("time took back to color", time.time() - a)
    return torch.stack(output)


def display_tensor(tensor, cmap):
    im = tensor.clone().permute(1, 2, 0).detach().cpu().numpy()
    if cmap == "gray":
        im = np.squeeze(im)
    plt.imshow(im, cmap=cmap, vmin=im.min(), vmax=im.max())


def closest_power(x, final_shape_addition):
    divider = 0
    if (256 + final_shape_addition - 12) % 16 == 0:
        divider = 12
    closest_power_ = int(16. * round((x + final_shape_addition - divider) / 16.)) + divider
    return closest_power_


# =========================
# ====== SAVE IMAGES ======
# =========================
def save_color_tensor_batch_as_numpy(batch, output_path, batch_num):
    b_size = batch.shape[0]
    for i in range(b_size):
        im_hdr = batch[i]
        tensor_0_1 = im_hdr.squeeze()
        im = tensor_0_1.clamp(0,1).permute(1, 2, 0).detach().cpu().numpy()
        im = (im * 255).astype('uint8')
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        cur_output_path = os.path.join(output_path, str(batch_num) + "_" + str(i) + ".jpg")
        imageio.imwrite(cur_output_path, im)


def save_gray_tensor_as_numpy(tensor, output_path, im_name):
    tensor = tensor.clamp(0, 1).clone().permute(1, 2, 0).detach().cpu().numpy()
    tensor_0_1 = np.squeeze(tensor)
    im = (tensor_0_1 * 255).astype('uint8')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    imageio.imwrite(os.path.join(output_path, im_name + ".png"), im, format='PNG-FI')


def save_gray_tensor_as_numpy_stretch(tensor, output_path, im_name):
    tensor = tensor.clamp(0,1).clone().permute(1, 2, 0).detach().cpu().numpy()
    tensor_0_1 = np.squeeze(tensor)
    tensor_0_1 = to_0_1_range_outlier(tensor_0_1)
    im = (tensor_0_1 * 255).astype('uint8')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    imageio.imwrite(os.path.join(output_path, im_name + ".png"), im, format='PNG-FI')
    print("result was saved to [%s]" % (os.path.join(output_path, im_name + ".png")))


def save_gray_tensor_as_numpy_stretch_entire_range(tensor, output_path, im_name):
    tensor = tensor.clone().permute(1, 2, 0).detach().cpu().numpy()
    tensor_0_1 = np.squeeze(tensor)
    tensor_0_1 = to_0_1_range(tensor_0_1)
    im = (tensor_0_1 * 255).astype('uint8')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    imageio.imwrite(os.path.join(output_path, im_name + ".png"), im, format='PNG-FI')


def save_color_tensor_as_numpy(tensor, output_path, im_name):
    print(tensor.max(), tensor.min())
    tensor = tensor.clamp(0, 1).clone().permute(1, 2, 0).detach().cpu().numpy()
    tensor_0_1 = np.squeeze(tensor)
    im = (tensor_0_1 * 255).astype('uint8')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    imageio.imwrite(os.path.join(output_path, im_name + ".png"), im, format='PNG-FI')

