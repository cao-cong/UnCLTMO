import math
import os

import matplotlib.pyplot as plt
import numpy as np

import utils.hdr_image_util as hdr_image_util


def plot_general_losses(G_loss_d, G_loss_ssim, G_loss_sigma, loss_D_fake, loss_D_real, title, iters_n, path, use_g_d_loss,
                        use_g_ssim_loss):
    if use_g_ssim_loss or use_g_d_loss:
        plt.figure()
        if len(loss_D_fake) != 0 and len(loss_D_real) != 0:
            plt.plot(range(len(loss_D_fake)), loss_D_fake, '-r', label='loss D fake')
            plt.plot(range(len(loss_D_fake)), loss_D_real, '-b', label='loss D real')
        plt.savefig(os.path.join(path, title + "D.png"))  # should before show method
        plt.close()
        plt.figure()
        if use_g_d_loss:
            plt.plot(range(iters_n), G_loss_d, '-g', label='loss G')
        if use_g_ssim_loss:
            plt.plot(range(iters_n), G_loss_ssim, '-y', label='loss G SSIM')
        if len(G_loss_sigma) != 0:
            plt.plot(range(iters_n), G_loss_sigma, '-k', label='loss G intensity')
        plt.xlabel("n iteration")
        plt.legend(loc='upper left')
        plt.title(title)

        # save image
        plt.savefig(os.path.join(path, title + "all.png"))  # should before show method
        plt.close()


def plot_discriminator_losses(loss_D_fake, loss_D_real, title, iters_n, path):
    plt.figure()
    plt.plot(range(iters_n), loss_D_fake, '-r', label='loss D fake')
    plt.plot(range(iters_n), loss_D_real, '-b', label='loss D real')
    plt.xlabel("n iteration")
    plt.legend(loc='upper left')
    plt.title(title)
    # save image
    plt.savefig(os.path.join(path, title + "all.png"))  # should before show method
    plt.close()


def plot_general_accuracy(acc_G, acc_D_fake, acc_D_real, title, iters_n, path):
    plt.figure()
    iters_n = max(iters_n, len(acc_D_real))
    plt.plot(range(iters_n), acc_D_fake, '-r', label='acc D fake')
    plt.plot(range(iters_n), acc_D_real, '-b', label='acc D real')
    # plt.plot(range(iters_n), acc_G, '-g', label='acc G')

    plt.xlabel("n iteration")
    plt.legend(loc='upper left')
    plt.title(title)

    # save image
    plt.savefig(os.path.join(path, title + ".png"))  # should before show method
    plt.close()


def display_batch_as_grid(batch, ncols_to_display, normalization, nrow=8, pad_value=0.0, isHDR=False,
                          batch_start_index=0, toPrint=False):
    batch = batch[batch_start_index:ncols_to_display]
    b_size = batch.shape[0]
    output = []
    for i in range(b_size):
        cur_im = batch[i].clone().permute(1, 2, 0).detach().cpu().numpy()
        if normalization == "0_1":
            norm_im = hdr_image_util.to_0_1_range(cur_im)
        elif normalization == "none":
            norm_im = cur_im
        else:
            raise Exception('ERROR: Not valid normalization for display')
        if i == 0 and toPrint:
            print("fake display --- max[%.4f]  min[%.4f]  dtype[%s]  shape[%s]" %
                  (float(np.max(norm_im)), float(np.min(norm_im)),
                   norm_im.dtype, str(norm_im.shape)))
        output.append(norm_im)
    norm_batch = np.asarray(output)
    nmaps = norm_batch.shape[0]
    xmaps = min(ncols_to_display, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(norm_batch.shape[1]), int(norm_batch.shape[2])
    if norm_im.shape[2] == 1:
        grid = np.full((height * ymaps, width * xmaps), pad_value)
    else:
        grid = np.full((height * ymaps, width * xmaps, norm_batch.shape[3]), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            if norm_im.shape[2] == 1:
                im_for_grid = norm_batch[k][:, :, 0]
            else:
                im_for_grid = norm_batch[k]
            grid[(y * height):(y * height + height), x * width: x * width + width] = im_for_grid
            k = k + 1
    return grid


def save_groups_images(test_hdr_batch, test_real_batch, fake1, fake0, new_out_dir, batch_size, epoch, image_mean):
    test_ldr_batch = test_real_batch["input_im"]
    test_hdr_image = test_hdr_batch["input_im"]
    test_ldr_batch = test_ldr_batch.reshape(-1,test_ldr_batch.shape[2],test_ldr_batch.shape[3],test_ldr_batch.shape[4])
    test_hdr_image = test_hdr_image.reshape(-1,test_hdr_image.shape[2],test_hdr_image.shape[3],test_hdr_image.shape[4])
    #output_len = int(batch_size / 4)
    output_len = 1
    display_group = [test_ldr_batch, test_hdr_image, fake1, fake0]
    titles = ["Real (LDR) Images", "Input (HDR) Images", "fake1", "fake0"]
    normalization_string_arr = ["0_1", "0_1", "0_1", "0_1"]
    for i in range(output_len):
        plt.figure(figsize=(15, 15))
        for j in range(4):
            display_im = display_batch_as_grid(display_group[j], ncols_to_display=(i + 1) * 4,
                                               normalization=normalization_string_arr[j],
                                               isHDR=False, batch_start_index=i * 4)
            plt.subplot(4, 1, j + 1)
            plt.axis("off")
            plt.title(titles[j])
            if display_im.ndim == 2:
                plt.imshow(display_im, cmap='gray', vmin=0, vmax=1)
            else:
                plt.imshow(display_im, vmin=0, vmax=1)
        plt.savefig(os.path.join(new_out_dir, "set " + str(i)))
        plt.close()


def plot_grad_flow(named_parameters, out_dir, epoch):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
