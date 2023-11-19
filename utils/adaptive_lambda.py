import scipy.optimize as optimize
import numpy as np
import os
import utils.hdr_image_util as hdr_image_util


def cross_entropy(factor, gray_im, targets, bins_):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    gray_im_log = np.log10(gray_im * factor + 1)
    gray_im_log = gray_im_log / gray_im_log.max()
    gray_im_flat = np.reshape(gray_im_log, (-1,))
    predictions, all_bins = np.histogram(gray_im_flat, bins=bins_, density=True, range=(0, 1))
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
    return ce


def verify_lambda_dict(f_factor_path, input_images_path):
    """
    check if lambda values were precomputed
    """
    f_factor_path = os.path.join(f_factor_path)
    if not os.path.isfile(f_factor_path):
        return False
    data = np.load(f_factor_path, allow_pickle=True)[()]
    for im_name in os.listdir(input_images_path):
        if os.path.splitext(im_name)[0] not in data:
            return False
    return True


def calc_lambda(f_factor_path, extensions, input_images_path, mean_hist_path, lambda_output_path, bins):
    if verify_lambda_dict(f_factor_path, input_images_path):
        return
    print("Calculating lambdas for input data...")
    input_names = os.listdir(input_images_path)
    mean_data = np.load(mean_hist_path, allow_pickle=True)[()]
    targets, all_bins = mean_data["mean_vals"], mean_data["all_bins"]
    res_dict = {}
    lambdas_output_path = os.path.join(lambda_output_path, "input_images_lambdas.npy")
    if os.path.isfile(lambdas_output_path):
        res_dict = np.load(lambdas_output_path, allow_pickle=True)[()]

    for img_name in input_names:
        if os.path.splitext(img_name)[0] not in res_dict and os.path.splitext(img_name)[1] in extensions:
            im_path = os.path.join(input_images_path, img_name)
            rgb_img = hdr_image_util.read_hdr_image(im_path)
            gray_im = hdr_image_util.to_gray(rgb_img)
            if gray_im.min() < 0:
                gray_im = gray_im - gray_im.min()
            gray_im_ = hdr_image_util.reshape_image(gray_im, train_reshape=False)
            gray_im_ = gray_im_ / gray_im_.max()
            sol = optimize.differential_evolution(cross_entropy, args=(gray_im_, targets, bins),
                                                  bounds=[(1, 1000000000)], maxiter=1000)
            print("[%s] [%.4f] [%.4f]" % (img_name, sol.x, sol.fun))
            #if sol.x[0]>50:
            #    sol.x[0]=50
            res_dict[os.path.splitext(img_name)[0]] = sol.x[0]
            np.save(lambdas_output_path, res_dict)
    print("Lambdas data saved successfully")
    return lambdas_output_path
