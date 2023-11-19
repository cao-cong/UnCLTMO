import os
import shutil
import numpy as np
import sys
import inspect
import os.path as path
two_up = path.abspath(path.join(__file__ ,"../.."))
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = two_up
sys.path.insert(0, parent_dir)
from utils import hdr_image_util
import matplotlib.pyplot as plt

def read_txt_file_with_bad_fid_images(input_dng_images_path, output_path):
    with open('bad_f_images_train.txt') as f:
        lines = [line.rstrip() for line in f]
    for im in lines:
        im_name = os.path.splitext(im)[0]
        print(im_name)
        cur_im = im_name + ".dng"
        print(cur_im)
        if cur_im in os.listdir(input_dng_images_path):

            old_im_path = os.path.join(input_dng_images_path, cur_im)
            new_im_path = os.path.join(output_path, cur_im)
            shutil.copy(old_im_path, new_im_path)
            # print(im)
        else:
            print(cur_im, " not in dir")


def print_f_from_dict(dict_path, input_im_path):
    f_res = np.load(dict_path, allow_pickle=True)[()]
    for im in os.listdir(input_im_path):
        im_name = os.path.splitext(im)[0]
        im_path = os.path.join(input_im_path, im)
        print("[%s] [%.4f]" % (im_name, f_res[im_name]))
        rgb_img = hdr_image_util.read_hdr_image(im_path)
        gray_im = hdr_image_util.to_gray(rgb_img)

        plt.subplot(2,1,1)
        plt.imshow(gray_im, cmap='gray')
        gray_im_ = gray_im - gray_im.min()
        gray_im_ = gray_im_ / gray_im_.max()
        gray_im_log = np.log10(gray_im_ * f_res[im_name] * 255 * 0.1 + 1)
        # gray_im_log = np.log10(gray_im_ * sol.x + 1)
        gray_im_log = gray_im_log / gray_im_log.max()
        plt.subplot(2, 1, 2)
        plt.imshow(gray_im_log, cmap='gray')
        plt.show()


print_f_from_dict("lum_hist_re/hdr_gallery20_bins.npy",
                  "../../train_data/HDRplus_patches256_npy")
# read_txt_file_with_bad_fid_images("","")