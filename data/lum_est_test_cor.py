import os
import sys
import inspect
import os.path as path
two_up = path.abspath(path.join(__file__ ,"../.."))
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = two_up
sys.path.insert(0, parent_dir)
import os
import numpy as np
from utils import hdr_image_util
import pandas as pd
import ntpath
import matplotlib.pyplot as plt
import imageio
import scipy.optimize as optimize
from scipy.optimize import minimize
from scipy.optimize import least_squares


def add_missing_data(dynamic_range_dict, lum_multiplier_dict):
    # dynamic_range_dict["BigfootPass"] = 1
    lum_multiplier_dict["BigfootPass"] = 4900

    # dynamic_range_dict["BigfootPass"] =
    lum_multiplier_dict["UpheavalDome"] = 4420

    # dynamic_range_dict["BigfootPass"] =
    lum_multiplier_dict["GeneralSherman"] = 611

    # dynamic_range_dict["BigfootPass"] =
    lum_multiplier_dict["KingsCanyon"] = 8420

    dynamic_range_dict["LuxoDoubleChecker"] = 800 * 1000
    lum_multiplier_dict["LuxoDoubleChecker"] = 7.68

    # dynamic_range_dict["BigfootPass"] =
    lum_multiplier_dict["DevilsGolfCourse"] = 13500

    # dynamic_range_dict["BigfootPass"] =
    lum_multiplier_dict["HooverDam"] = 3900

    # dynamic_range_dict["BigfootPass"] =
    lum_multiplier_dict["RoadsEndFireDamage"] = 1310

    # dynamic_range_dict["BigfootPass"] =
    lum_multiplier_dict["HalfDomeSunset"] = 40

    # dynamic_range_dict["BigfootPass"] =
    lum_multiplier_dict["ElCapitan"] = 5750

    return dynamic_range_dict, lum_multiplier_dict


def read_xls_data_and_save_to_dict(input_dir, output_path):
    dynamic_range_dict = {}
    lum_multiplier_dict = {}
    no_data_lst = []
    for xls_name in os.listdir(input_dir):
        im_path = os.path.join(input_dir, xls_name)
        xl = pd.ExcelFile(im_path)
        df1 = xl.parse('Sheet1')
        lum_mul = df1['Date:'][40]
        dr = df1['Unnamed: 4'][40]
        if "K" in str(dr):
            str_dr = str(dr)[:-3]
            dr_final = float(str_dr) * 1000
            print("%s lum_mul[%.4f] dr[%.4f]" % (os.path.splitext(xls_name)[0][:-4], lum_mul, dr_final))
            dynamic_range_dict[os.path.splitext(xls_name)[0][:-4]] = dr_final
            lum_multiplier_dict[os.path.splitext(xls_name)[0][:-4]] = lum_mul
        else:
            no_data_lst.append(os.path.splitext(xls_name)[0][:-4])

    dynamic_range_dict, lum_multiplier_dict = add_missing_data(dynamic_range_dict, lum_multiplier_dict)

    output_dynamic_range_path = os.path.join(output_path, "dynamic_range_dict.npy")
    output_lum_multiplier_path = os.path.join(output_path, "lum_multiplier_dict.npy")

    np.save(output_dynamic_range_path, dynamic_range_dict)
    np.save(output_lum_multiplier_path, lum_multiplier_dict)
    print(no_data_lst)


def save_f_factors_dict(input_dir, output_path):
    res_dict_no_mul = {}
    res_dict_mul = {}
    for img_name in os.listdir(input_dir):
        if img_name == "BarHarborSunrise.exr":
            im_path = os.path.join(input_dir, img_name)
            rgb_img = hdr_image_util.read_hdr_image(im_path)
            # if np.min(rgb_img) < 0:
            #     gray_im = hdr_image_util.to_gray(rgb_img)
            #     print(img_name, np.min(rgb_img), np.min(gray_im), np.percentile(gray_im, 1.0), np.percentile(gray_im, 99.0))
            #     rgb_img = rgb_img + np.abs(np.min(rgb_img))
            gray_im = hdr_image_util.to_gray(rgb_img)
            gamma_factor = hdr_image_util.get_new_brightness_factor(rgb_img)
            # gamma_factor = 1
            print("%s [%.4f]" % (img_name, gamma_factor))
            res_dict_no_mul[os.path.splitext(img_name)[0]] = gamma_factor
            res_dict_mul[os.path.splitext(img_name)[0]] = gamma_factor * 255 * 0.1
    # output_path_res_dict_no_mul = os.path.join(output_path, "f_factors_no_mul.npy")
    # output_path_res_dict_mul = os.path.join(output_path, "f_factors_mul.npy")
    # np.save(output_path_res_dict_no_mul, res_dict_no_mul)
    # np.save(output_path_res_dict_mul, res_dict_mul)


def fix_im_avg(gray_im, brightness_factor):
    new_gray_im = np.log10((gray_im / np.max(gray_im)) * brightness_factor + 1)
    new_gray_im = new_gray_im / new_gray_im.max()
    if new_gray_im.mean() < 0.5:
        dist = 0.6 - new_gray_im.mean()
        print("dist", dist)
        gray_im2 = gray_im
        gray_im2 = (gray_im2 / gray_im2.max()) * brightness_factor * (brightness_factor ** dist)
        gray_im2 = np.log10(gray_im2 + 1)
        gray_im2 = gray_im2 / gray_im2.max()
        return gray_im2
    return new_gray_im


def save_percentile_dict(input_dir, output_path, output_name, top_per, bot_per):
    res_dict_contrast_ratio = {}
    for img_name in os.listdir(input_dir):
        im_path = os.path.join(input_dir, img_name)
        rgb_img = hdr_image_util.read_hdr_image(im_path)
        # if np.min(rgb_img) < 0:
        #     rgb_img = rgb_img + np.abs(np.min(rgb_img))
        gray_im = hdr_image_util.to_gray(rgb_img)
        im_max = np.percentile(gray_im, top_per)
        im_min = np.percentile(gray_im, bot_per)
        if im_min == 0:
            im_min += 0.0001
        contrast_ratio = im_max / im_min
        if contrast_ratio < 0:
            print("====================")
        print("%s [%.4f]" % (img_name, contrast_ratio))
        res_dict_contrast_ratio[os.path.splitext(img_name)[0]] = contrast_ratio
    output_path_contrast_ratio = os.path.join(output_path, output_name)
    # output_path_stop = os.path.join(output_path, "stop.npy")
    np.save(output_path_contrast_ratio, res_dict_contrast_ratio)
    # np.save(output_path_stop, res_dict_stop)


def save_gaussian_filter_dict(input_dir, output_path, output_name, top_per, bot_per):
    from scipy import misc, ndimage
    res_dict_contrast_ratio = {}
    for img_name in os.listdir(input_dir):
        im_path = os.path.join(input_dir, img_name)
        rgb_img = hdr_image_util.read_hdr_image(im_path)
        # if np.min(rgb_img) < 0:
        #     rgb_img = rgb_img + np.abs(np.min(rgb_img))
        gray_im = hdr_image_util.to_gray(rgb_img)
        s = max(gray_im.shape[0], gray_im.shape[1])
        s = s * 0.002
        print(s, gray_im.max(), gray_im.mean(), gray_im.min())
        gray_im = ndimage.gaussian_filter(gray_im, sigma=s)
        print(s, gray_im.max(), gray_im.mean(), gray_im.min())
        im_max = np.percentile(gray_im, top_per)
        im_min = np.percentile(gray_im, bot_per)
        if im_min == 0:
            im_min += 0.0001
        contrast_ratio = im_max / im_min
        if contrast_ratio < 0:
            print("====================")
        print("%s [%.4f]" % (img_name, contrast_ratio))
        res_dict_contrast_ratio[os.path.splitext(img_name)[0]] = contrast_ratio
    output_path_contrast_ratio = os.path.join(output_path, output_name)
    # output_path_stop = os.path.join(output_path, "stop.npy")
    np.save(output_path_contrast_ratio, res_dict_contrast_ratio)
    # np.save(output_path_stop, res_dict_stop)


def calculate_corr(path_a_, path_b_, path_c_):
    dict1 = np.load(path_a_, allow_pickle=True)[()]
    # print(dict1)
    dict2 = np.load(path_b_, allow_pickle=True)[()]
    dict3 = np.load(path_c_, allow_pickle=True)[()]

    # keys = list(dict1.keys())
    keys = list(dict3.keys())
    # if "WaffleHouse" in keys:
    #     keys.remove("WaffleHouse")
    if "LuxoDoubleChecker" in keys:
        keys.remove("LuxoDoubleChecker")
    # if "Flamingo" in keys:
    #     keys.remove("Flamingo")
    # if "Zentrum" in keys:
    #     keys.remove("Zentrum")
    # if "LasVegasStore" in keys:
    #     keys.remove("LasVegasStore")
    # dict1.pop("Zentrum")
    # dict1.pop("LasVegasStore")
    # keys2 = list(dict2.keys())
    # if len(keys2) < len(keys):
    #     keys = keys2
    # print(keys[16], dict1[keys[16]])
    # print(keys[17])
    # print(keys[18])
    # print(keys[16], dict1[keys[16]], dict2[keys[16]])
    print(keys[3], dict1[keys[3]], dict2[keys[3]])
    # print(keys[3], dict1[keys[3]], dict2[keys[3]])
    # im = imageio.imread("/Users/yaelvinker/Documents/university/lab/open_exr_fixed_size/exr_format_fixed_size/" + keys[21] + ".exr", format='EXR-FI')
    # im = imageio.imread(
    #     "/Users/yaelvinker/Documents/university/lab/open_exr_fixed_size/exr_format_fixed_size/" + keys[3] + ".exr",
    #     format='EXR-FI')
    # plt.figure()
    # plt.imshow(im)
    # plt.show()
    vec1 = np.array([dict1.get(x, 0) for x in keys])
    vec1 = np.delete(vec1, 18)
    # vec1 = np.delete(vec1, 7)
    vec1 = np.delete(vec1, 3)
    vec1 = np.delete(vec1, 33)

    # vec1 = np.delete(vec1, 16)
    # vec1 = np.delete(vec1, 16)
    # vec1 = np.delete(vec1, 16)


    vec2 = np.array([dict2.get(x, 0) for x in keys])
    vec2 = np.delete(vec2, 18)
    # vec2 = np.delete(vec2, 7)
    vec2 = np.delete(vec2, 3)
    vec2 = np.delete(vec2, 33)


    # vec2 = np.delete(vec2, 16)
    # vec2 = np.delete(vec2, 16)
    # vec2 = np.delete(vec2, 16)

    #
    # vec2 = np.delete(vec2, 22)
    # print(keys[19], dict1[keys[19]], dict2[keys[19]])
    # print(keys[20], dict1[keys[20]], dict2[keys[20]])
    # print("contrast_ratio",vec1)
    # print("our",vec2)
    import operator
    # dict1.pop("Flamingo")
    # dict1.pop("Zentrum")
    # dict1.pop("LasVegasStore")
    m1 = max(dict1.items(), key=operator.itemgetter(1))
    # print(m1)
    # dict2.pop("Flamingo")
    # dict2.pop("Zentrum")
    # dict2.pop("LasVegasStore")
    m2 = max(dict2.items(), key=operator.itemgetter(1))
    # print(m2)
    vec1 = vec1 / np.max(vec1)
    vec2 = vec2 / np.max(vec2)
    # print(keys[32], vec1[32], vec2[32])
    a = np.max(vec2) / np.max(vec1)
    # print(a)
    # vec1 = vec1 * 26
    # print("contrast_ratio after", vec1)
    # print(vec1.max(), vec2.max())

    # print(len(vec1), len(vec2))
    # print(min(vec2))

    # print(keys[12], vec2[12])
    # print(list(map(int, vec1)))
    # print(list(map(int, vec2)))
    # print(np.correlate(vec1, vec2))
    print("corr of [%s][%s] on [%d] images" % (os.path.splitext(ntpath.basename(path_a_))[0],
                                   os.path.splitext(ntpath.basename(path_b_))[0], len(vec1)))
    print(np.corrcoef(vec1, vec2))
    title = "[%s][%s] on [%d] images [%.4f]" % (os.path.splitext(ntpath.basename(path_a_))[0],
                                   os.path.splitext(ntpath.basename(path_b_))[0], len(vec1), np.corrcoef(vec1, vec2)[0,1])
    plt.figure()
    plt.plot(vec1, "-b", label=os.path.splitext(ntpath.basename(path_a_))[0])
    plt.plot(vec2, "-r", label=os.path.splitext(ntpath.basename(path_b_))[0])
    plt.title(title)
    plt.legend()
    # import numpy
    # numpy.corrcoef(
    #     [dict1.get(x, 0) for x in keys],
    #     [dict2.get(x, 0) for x in keys])[0, 1]


def run_corr():
    # path_a = "/Users/yaelvinker/Documents/university/lab/open_exr_fixed_size/contrast_ratio.npy"

    path_b = "/Users/yaelvinker/Documents/university/lab/open_exr_fixed_size/dynamic_range_dict.npy"
    path_c = "/Users/yaelvinker/Documents/university/lab/open_exr_fixed_size/dynamic_range_dict.npy"

    path_a = "/Users/yaelvinker/Documents/university/lab/open_exr_fixed_size/original_size/exr_original_size20_bins.npy"
    calculate_corr(path_a, path_b, path_c)
    path_a = "/Users/yaelvinker/Documents/university/lab/open_exr_fixed_size/original_size/lowfilter_percentile_100_0.npy"
    calculate_corr(path_a, path_b, path_c)
    path_a = "/Users/yaelvinker/Documents/university/lab/open_exr_fixed_size/original_size/lowfilter_percentile_100_0.1.npy"
    calculate_corr(path_a, path_b, path_c)
    # path_a = "/Users/yaelvinker/Documents/university/lab/open_exr_fixed_size/original_size/percentile_99_1.npy"
    # calculate_corr(path_a, path_b, path_c)
    # path_a = "/Users/yaelvinker/PycharmProjects/lab/data_generator/hist_fit/results/exr_hist_dict_20_bins.npy"
    # calculate_corr(path_a, path_b, path_c)
    # path_a = "/Users/yaelvinker/Documents/university/lab/open_exr_fixed_size/original_size/f_factors_no_mul.npy"
    # calculate_corr(path_a, path_b, path_c)
    # path_a = "/Users/yaelvinker/Documents/university/lab/open_exr_fixed_size/f_factors_no_mul.npy"
    # calculate_corr(path_a, path_b, path_c)



def get_hist_from_path(im_path):
    rgb_img = imageio.imread(im_path).astype('float32')
    gray_im = hdr_image_util.to_gray(rgb_img)
    gray_im = gray_im / gray_im.max()
    gray_im_flat = np.reshape(gray_im, (-1,))
    val, all_bins = np.histogram(gray_im_flat, bins=20, density=True, range=(0, 1))
    return val


def fit_avg_hist(ldr_images_path, output_path):
    img_name = os.listdir(ldr_images_path)[0]
    im_path = os.path.join(ldr_images_path, img_name)
    rgb_img = imageio.imread(im_path).astype('float32')
    gray_im = hdr_image_util.to_gray(rgb_img)
    gray_im = gray_im / gray_im.max()
    gray_im_flat = np.reshape(gray_im, (-1,))
    sum_hists, all_bins = np.histogram(gray_im_flat, bins=20, density=True, range=(0, 1))
    # plt.figure()
    # plt.bar(all_bins[:-1] + np.diff(all_bins) / 2, sum_hists, np.diff(all_bins))
    # print(sum_hists.shape)
    print(img_name)
    for img_name in os.listdir(ldr_images_path)[1:]:
        im_path = os.path.join(ldr_images_path, img_name)
        rgb_img = imageio.imread(im_path).astype('float32')
        gray_im = hdr_image_util.to_gray(rgb_img)
        gray_im = hdr_image_util.reshape_image(gray_im, train_reshape=False)
        gray_im = gray_im / gray_im.max()
        gray_im_flat = np.reshape(gray_im, (-1,))
        val, bins = np.histogram(gray_im_flat, bins=20, density=True, range=(0, 1))
        sum_hists += val
        print(img_name)
    plt.figure()
    mean_vals = sum_hists / len(os.listdir(ldr_images_path))
    plt.bar(all_bins[:-1] + np.diff(all_bins) / 2, mean_vals, np.diff(all_bins))
    # plt.hist(gray_im_flat, rwidth=0.9, color='#607c8e', density=False, bins=20)
    mean_val_dict = {"mean_vals": mean_vals, "all_bins": all_bins}
    avg_output_path = os.path.join(output_path, "ldr_avg_hist_900_images.npy")
    np.save(avg_output_path, mean_val_dict)
    # return mean_vals, all_bins


def cross_entropy(factor, gray_im, targets, bins_):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    # print(factor)
    # epsilon = 1e-12
    # return (factor - 2) ** 2

    # gray_im_log = np.log10(gray_im*factor + 1)
    gray_im_log = np.log10(gray_im * factor + 1)
    # gray_im_log = (gray_im_log - gray_im_log.min()) / (gray_im_log.max() - gray_im_log.min())
    gray_im_log = gray_im_log / gray_im_log.max()
    gray_im_flat = np.reshape(gray_im_log, (-1,))
    predictions, all_bins = np.histogram(gray_im_flat, bins=bins_, density=True, range=(0, 1))

    # predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
    # print(ce)
    return ce


def plot_im_hist(sol, gray_im):
    # gray_im_ = gray_im - gray_im.min()
    max_a = np.percentile(gray_im, 99.)
    gray_im_ = gray_im / gray_im.max()
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(gray_im_ ** 0.01, cmap='gray')
    title = "max[%.1f] mean[%.f] min[%.1f]" % (gray_im_.max(), gray_im_.mean(), gray_im_.min())
    plt.title(title)
    gray_im_flat = np.reshape(gray_im_, (-1,))
    predictions, all_bins = np.histogram(gray_im_flat, bins=bins_, density=True, range=(0, 1))
    plt.subplot(2, 1, 2)
    plt.bar(all_bins[:-1] + np.diff(all_bins) / 2, predictions, np.diff(all_bins))


def plot_res(sol, gray_im):
    gray_im_ = gray_im - gray_im.min()
    gray_im_ = gray_im_ / gray_im_.max()
    gray_im_log = np.log10(gray_im_ * sol.x * 255 * 0.1 + 1)
    gray_im_log = gray_im_log / gray_im_log.max()
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title(sol.x)
    plt.imshow(gray_im_log, cmap='gray')
    gray_im_flat = np.reshape(gray_im_log, (-1,))
    predictions, all_bins = np.histogram(gray_im_flat, bins=bins_, density=True, range=(0, 1))
    plt.subplot(2, 1, 2)
    plt.bar(all_bins[:-1] + np.diff(all_bins) / 2, predictions, np.diff(all_bins))


def test_loss_for_belguim(im_path, mean_hist_path):
    #6539.5077]
    mean_data = np.load(mean_hist_path, allow_pickle=True)[()]
    targets, all_bins = mean_data["mean_vals"], mean_data["all_bins"]
    bins=20
    rgb_img = hdr_image_util.read_hdr_image(im_path)
    gray_im = hdr_image_util.to_gray(rgb_img)
    # gray_im = hdr_image_util.reshape_image(gray_im, train_reshape=False)
    gray_im = gray_im / gray_im.max()
    # factor = 6539.5077
    factor = 6539.5077 * 2
    print(cross_entropy(factor, gray_im, targets, bins))


def save_lumest_hist_dict(input_images_path, mean_hist_path, output_path, output_name, input_names_path, bins_):#, old_dict, fix_dict, fix_name):
    mean_data = np.load(mean_hist_path, allow_pickle=True)[()]
    targets, all_bins = mean_data["mean_vals"], mean_data["all_bins"]
    # plt.bar(all_bins[:-1] + np.diff(all_bins) / 2, targets, np.diff(all_bins))

    res_dict = {}
    hist_factors_path = os.path.join(output_path, output_name + str(bins_) + "_bins.npy")
    if os.path.isfile(hist_factors_path):
        res_dict = np.load(hist_factors_path, allow_pickle=True)[()]
        print(res_dict)
    if os.path.isdir(input_names_path):
        input_names = os.listdir(input_names_path)
    else:
        input_names = np.load(input_names_path, allow_pickle=True)[()]
    for img_name in input_names:
        if os.path.splitext(img_name)[0] not in res_dict and os.path.splitext(img_name)[1] in [".hdr", ".exr", ".dng", ".npy"]:
            im_path = os.path.join(input_images_path, img_name)
            rgb_img = hdr_image_util.read_hdr_image(im_path)
            gray_im = hdr_image_util.to_gray(rgb_img)
            if gray_im.min() < 0:
                gray_im = gray_im - gray_im.min()
                print("min")
            gray_im_ = hdr_image_util.reshape_image(gray_im, train_reshape=False)
            gray_im_ = gray_im_ / gray_im_.max()
            sol = optimize.differential_evolution(cross_entropy, args=(gray_im_, targets, bins_), bounds=[(1, 1000000000)], maxiter=1000)
            # print(sol)
            # print("[%s] [%.4f] [%.4f]" % (img_name, sol.x, old_dict[os.path.splitext(img_name)[0]]))
            if sol.x[0]>50:
                sol.x[0]=50
            print("[%s] [%.4f] [%.4f]" % (img_name, sol.x, sol.fun))
            # print(sol)
            # fix_dict[os.path.splitext(img_name)[0]] = sol.x[0]
            # np.save(fix_name, fix_dict)
            # plot_im_hist(sol, gray_im)
            # plot_res(sol, gray_im)
            res_dict[os.path.splitext(img_name)[0]] = sol.x[0]
            # plt.show()
            np.save(hist_factors_path, res_dict)


def get_names_to_fix(input_names_path):
    if os.path.isdir(input_names_path):
        input_names = os.listdir(input_names_path)
        return input_names
    ext = os.path.splitext(os.path.basename(input_names_path))[1]
    if ext == ".txt":
        with open(input_names_path) as f:
            input_names = [line.rstrip() for line in f]
    elif ext == ".npy":
        input_names = np.load(input_names_path, allow_pickle=True)[()]
    return input_names


def fix_outliers(input_images_path, mean_hist_path, output_path, output_name, input_names_path, bins_,
                 old_dict_path, fix_dict_path):
    mean_data = np.load(mean_hist_path, allow_pickle=True)[()]
    targets, all_bins = mean_data["mean_vals"], mean_data["all_bins"]
    old_dict = np.load(old_dict_path, allow_pickle=True)[()]
    fix_dict = np.load(fix_dict_path, allow_pickle=True)[()]

    res_dict = {}
    hist_factors_path = os.path.join(output_path, output_name + str(bins_) + "_bins.npy")

    input_names = get_names_to_fix(input_names_path)
    for img_name in input_names:
        im_path = os.path.join(input_images_path, img_name)
        rgb_img = hdr_image_util.read_hdr_image(im_path)
        gray_im = hdr_image_util.to_gray(rgb_img)
        if gray_im.min() < 0:
            gray_im = gray_im - gray_im.min()
            print("min")
        gray_im_ = hdr_image_util.reshape_image(gray_im, train_reshape=False)
        gray_im_ = gray_im_ / gray_im_.max()
        sol = optimize.differential_evolution(cross_entropy, args=(gray_im_, targets, bins_), bounds=[(1, 1000000000)], maxiter=1000)
        print("[%s] [%.4f] [%.4f] [%.4f]" % (img_name, old_dict[os.path.splitext(img_name)[0]], sol.x, sol.fun))
        fix_dict[os.path.splitext(img_name)[0]] = sol.x[0]
        np.save(fix_dict_path, fix_dict)
        res_dict[os.path.splitext(img_name)[0]] = sol.x[0]
        np.save(hist_factors_path, res_dict)


def split_data_names_to_dicts(input_names_path, output_path):
    total_image = len(os.listdir(input_names_path))
    counter = 0
    start, end = 0, 100
    while counter < total_image:
        output_list = []
        output_name = "dng[%d_%d].npy" % (start, end)
        print(output_name)
        for img_name in os.listdir(input_names_path)[start: end]:
            output_list.append(img_name)
            counter += 1
        cur_output_path = os.path.join(output_path, output_name)
        np.save(cur_output_path, output_list)
        start, end = end, min(end + 100, total_image)



def unite_dng_dicts(input_split_path, output_path):
    all_dict = {}
    for dict_name in os.listdir(input_split_path):
        print("========= %s ==========" % (dict_name))
        cur_path = os.path.join(input_split_path, dict_name)
        cur_dict = np.load(cur_path, allow_pickle=True)[()]
        for k in cur_dict.keys():
            print("[%s] %.4f" % (k, cur_dict[k]))
            all_dict[k] = cur_dict[k]
    output_name = "dng_hist_20_bins_all.npy"
    cur_output_path = os.path.join(output_path, output_name)
    np.save(cur_output_path, all_dict)


def save_outliers(input_path, output_path):
    outlier_dict = []
    a = np.load(input_path, allow_pickle=True)[()]
    vals = {k: v for k, v in sorted(a.items(), key=lambda item: item[1], reverse=True)}
    print(vals)
    print(len(vals.keys()))
    counter = 0
    for k in vals.keys():
        if vals[k] > 800:
            print(k, vals[k])
            outlier_dict.append(k)
            counter += 1
    print(counter)
    np.save(output_path, outlier_dict)


if __name__ == '__main__':
    # mean_hist_path = "/Users/yaelvinker/PycharmProjects/lab/tests/ldr_avg_hist_900_images_20_bins.npy"
    # im_path = "/Users/yaelvinker/PycharmProjects/lab/utils/folders/temp_data/belgium.hdr"
    # test_loss_for_belguim(im_path, mean_hist_path)
    # im = imageio.imread("/Users/yaelvinker/Documents/university/lab/Oct/10_31/"
    #                     "D_[0.8,0.8,0.8]_pad_0_G_ssr_doubleConvT__d1.0_struct_1.0[1,1,1]__trans2_replicate_lr_g1e-05_d1e-05_decay50__noframe__"
    #                     "min_log_0.1hist_fit_/our_merged_0006_20160722_183131_669_fake_clamp_and_stretch.png")
    # plt.subplot(2,2,1)
    # plt.imshow(im)
    # plt.subplot(2, 2, 2)
    # plt.imshow((im / im.max()) ** (1/1.5))
    # im_2 = imageio.imread("/Users/yaelvinker/Documents/university/lab/Oct/10_31/"
    #                     "D_[0.8,0.8,0.8]_pad_0_G_ssr_doubleConvT__d1.0_struct_1.0[1,1,1]__trans2_replicate_lr_g1e-05_d1e-05_decay50__noframe__"
    #                     "min_log_0.1hist_fit_/merged_0006_20160722_183131_669.png")
    # plt.subplot(2, 2, 3)
    # plt.imshow(im_2)
    # plt.show()
    # run_corr()
    # plt.show()
    # save_percentile_dict(input_dir, output_path, output_name, top_per, bot_per)
    # dict4 = np.load("/Users/yaelvinker/Documents/university/lab/open_exr_fixed_size/original_size/exr_original_size20_bins.npy", allow_pickle=True)[()]
    # dict4["TaughannockFalls"] = 39.8829
    # np.save("/Users/yaelvinker/Documents/university/lab/open_exr_fixed_size/original_size/exr_original_size20_bins.npy", dict4)
    # for k in dict4.keys():
    #     print(k, dict4[k])
    #Zentrum AhwahneeGreatLounge HooverGarage WillyDesk
    # print(dict4["TaughannockFalls"])

    # save_outliers("/cs/labs/raananf/yael_vinker/data/new_lum_est_hist/fix_lum_hist/dng_hist_20_bins_all_fix.npy",
    #               "/cs/labs/raananf/yael_vinker/data/new_lum_est_hist/fix_lum_hist/outliers.npy")

    # input_images_path = "/cs/labs/raananf/yael_vinker/data/zhang2019_data/testset/newdata-hdr/"
    # mean_hist_path = "/cs/labs/raananf/yael_vinker/data/new_lum_est_hist/ldr_avg_hist_900_images_20_bins.npy"
    # output_path = "//cs/labs/raananf/yael_vinker/data/new_lum_est_hist/zhang/"
    # output_name = "fix_ouliers"
    # input_names_path = "/cs/labs/raananf/yael_vinker/data/new_lum_est_hist/zhang/outliers.npy"
    # bins_ = 20
    # old_dict_path = "/cs/labs/raananf/yael_vinker/data/new_lum_est_hist/zhang/zhang20_bins.npy"
    # fix_dict_path = "/cs/labs/raananf/yael_vinker/data/new_lum_est_hist/zhang/zhang20_bins.npy"
    # fix_outliers(input_images_path, mean_hist_path, output_path, output_name, input_names_path, bins_,
    #                  old_dict_path, fix_dict_path)

    # unite_dng_dicts("/cs/labs/raananf/yael_vinker/data/new_lum_est_hist/dng_hist_fit_factor_dict_split" "/cs/labs/raananf/yael_vinker/data/new_lum_est_hist")
    # input_names = np.load("/Users/yaelvinker/Documents/university/lab/Oct/10_13//exr_hist_dict_20_bins.npy", allow_pickle=True)[()]
    # input_names["DelicateArch"] = 4.2439
    # input_names["DelicateFlowers"] = 7.2100
    # np.save("/Users/yaelvinker/Documents/university/lab/Oct/10_13//exr_hist_dict_20_bins.npy", input_names)
    # print(input_names)
    # print(len(input_names.keys()))
    # split_data_names_to_dicts("/Users/yaelvinker/PycharmProjects/lab/utils/folders/temp_data", "/Users/yaelvinker/PycharmProjects/lab/tests/hist_fit")
    import argparse
    parser = argparse.ArgumentParser(description="Parser for gan network")
    parser.add_argument("--input_images_path", type=str)
    parser.add_argument("--mean_hist_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--output_name", type=str)
    parser.add_argument("--inpue_names_path", type=str)
    args = parser.parse_args()
    # input_images_path_ = args.input_images_path
    # mean_hist_path_ = args.mean_hist_path
    # output_path_ = args.output_path
    # output_name_ = args.output_name
    # input_names_path_ = args.inpue_names_path
    mean_hist_path_ = "../activate_trained_model/lambda_data/ldr_avg_hist_900_images_20_bins.npy"
    output_path_ = "lum_hist_re/"
    input_images_path_="../../train_data/HDRplus_patches256_npy"
    input_names_path_=input_images_path_
    output_name_="hdr_gallery"
    bins_ = 20
    # # # output_name = "test_.npy"
    # # f_res = np.load("/Users/yaelvinker/Documents/university/lab/hist_fit_bad_train_im/dng_hist_20_bins_all.npy", allow_pickle=True)[()]
    # # fix_dict = np.load("/Users/yaelvinker/Documents/university/lab/hist_fit_bad_train_im/dng_hist_20_bins_all_fix.npy",
    # #                 allow_pickle=True)[()]
    save_lumest_hist_dict(input_images_path_, mean_hist_path_, output_path_, output_name_, input_names_path_, bins_)#, f_res, fix_dict,"/Users/yaelvinker/Documents/university/lab/hist_fit_bad_train_im/dng_hist_20_bins_all_fix.npy")
    # plt.show()
