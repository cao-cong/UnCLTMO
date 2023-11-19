import os
import imageio
import numpy as np
import torch
from PIL import Image
from torchsummary import summary


def print_loader(im, title):
    print(title + " --- max[%.4f]  min[%.4f]  dtype[%s]  shape[%s]  unique[%s]" %
          (float(np.max(im)), float(np.min(im)),
           im.dtype, str(im.shape),
           str(np.unique(im).shape[0])))


def print_tensor_loader(im, title):
    print(title + " --- max[%.4f]  min[%.4f]  dtype[%s]  shape[%s]  unique[%s]" %
          (float(im.max()), float(im.min()),
           im.dtype, str(im.shape),
           str(np.unique(im.numpy()).shape[0])))


def load_data_dict_mode(test_hdr_dataloader, test_ldr_dataloader, title, images_number=1):
    test_hdr_loader = next(iter(test_hdr_dataloader))
    test_ldr_loader = next(iter(test_ldr_dataloader))
    for i in range(images_number):
        print()
        input_test_hdr_loader_single = np.asarray(test_hdr_loader["input_im"][i])
        print_loader(input_test_hdr_loader_single, title + "_gray_hdr_loader_single")
        color_test_hdr_loader_single = np.asarray(test_hdr_loader["color_im"][i])
        print_loader(color_test_hdr_loader_single, title + "_color_hdr_loader_single")

        input_test_ldr_loader_single = np.asarray(test_ldr_loader["input_im"][i])
        print_loader(input_test_ldr_loader_single, title + "_gray_ldr_loader_single")
        color_test_ldr_loader_single = np.asarray(test_ldr_loader["color_im"][i])
        print_loader(color_test_ldr_loader_single, title + "_color_ldr_loader_single")


def load_data_train_mode(test_hdr_dataloader, test_ldr_dataloader, images_number=1):
    test_hdr_loader = next(iter(test_hdr_dataloader))[0]
    test_ldr_loader = next(iter(test_ldr_dataloader))[0]
    for i in range(images_number):
        print()
        input_test_hdr_loader_single = np.asarray(test_hdr_loader[i])
        print("input image train_hdr_dataloader --- max[%.4f]  min[%.4f]  dtype[%s]  shape[%s]  unique[%s]" %
              (float(np.max(input_test_hdr_loader_single)), float(np.min(input_test_hdr_loader_single)),
               input_test_hdr_loader_single.dtype, str(input_test_hdr_loader_single.shape),
               str(np.unique(input_test_hdr_loader_single).shape[0])))

        test_ldr_loader_single = np.asarray(test_ldr_loader[i])
        print("test_ldr_dataloader --- max[%.4f]  min[%.4f]  dtype[%s]  shape[%s]  unique[%s]" %
              (float(np.max(test_ldr_loader_single)), float(np.min(test_ldr_loader_single)),
               test_ldr_loader_single.dtype, str(test_ldr_loader_single.shape),
               str(np.unique(test_ldr_loader_single).shape[0])))


def print_dataset_details(data_loaders, data_roots, titles, isHdr, testMode):
    print("-------------------------------- loader details --------------------------------")
    for i in range(len(data_loaders)):
        print(titles[i] + " [%d] images" % (len(data_loaders[i].dataset)))
        if testMode[i]:
            input_im, display_image = get_single_im(data_roots[i], isHdr[i], testMode[i])
            print_tensor_loader(input_im, "input_im " + titles[i])
            print_tensor_loader(display_image, "display_image " + titles[i])
        else:
            input_im = get_single_im(data_roots[i], isHdr[i], testMode[i])
            print_loader(input_im, "input_im " + titles[i])
        print()
    print("---------------------------------------------------------------------------------")


def get_single_ldr_im(im_path, file_extension, testMode):
    if file_extension == ".npy":
        data = np.load(im_path, allow_pickle=True)
        if testMode:
            input_im = data[()]["input_image"]
            display_image = data[()]["display_image"]
            return input_im, display_image
        return data
    elif file_extension == ".bmp":
        with open(im_path, 'rb') as f:
            img = Image.open(f)
            img = np.asarray(img.convert('RGB'))
    return img


def get_single_im(data_root, isHdr, testMode):
    x = next(os.walk(data_root))[1][0]
    dir_path = os.path.join(data_root, x)
    im_name = os.listdir(dir_path)[0]
    im_path = os.path.join(dir_path, im_name)
    file_extension = os.path.splitext(im_name)[1]
    if isHdr:
        if testMode:
            input_im, display_image = get_single_hdr_im(im_path, file_extension, testMode)
            return input_im, display_image
        else:
            input_im = get_single_hdr_im(im_path, file_extension, testMode)
    else:
        if testMode:
            input_im, display_image = get_single_ldr_im(im_path, file_extension, testMode)
            return input_im, display_image
        else:
            input_im = get_single_ldr_im(im_path, file_extension, testMode)
    return input_im


def get_single_hdr_im(im_path, file_extension, testMode):
    if file_extension == ".hdr":
        input_im = imageio.imread(im_path, format="HDR-FI").astype('float32')
    elif file_extension == ".dng":
        input_im = imageio.imread(im_path, format="RAW-FI").astype('float32')
    elif file_extension == ".npy":
        data = np.load(im_path, allow_pickle=True)
        if testMode:
            input_im = data[()]["input_image"]
            display_image = data[()]["display_image"]
            return input_im, display_image
        else:
            input_im = data
    return input_im


def print_cuda_details(device):
    if (device == 'cuda') and (torch.cuda.device_count() >= 1):
        print("Using [%d] GPUs" % torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print("GPU [%d] device name = %s" % (i, torch.cuda.get_device_name(i)))
        print(torch.cuda.current_device())


def print_test_epoch_losses_summary(num_epochs, epoch, test_loss_D, test_errGd, accDreal_test, accDfake_test,
                                    accG_test):
    print("===== Test results =====")
    print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
          % (epoch, num_epochs, test_loss_D, test_errGd))
    print('[%d/%d]\taccuracy_D_real: %.4f \taccuracy_D_fake: %.4f \taccuracy_G: %.4f'
          % (epoch, num_epochs, accDreal_test, accDfake_test, accG_test))


def print_TMQI_summary(q, s, n, num_epochs, epoch):
    print('[%d/%d]\tQ: %.8f\tS: %.8f\tN: %.8f\t'
          % (epoch, num_epochs, q, s, n))


def print_g_progress(fake, title=""):
    with torch.no_grad():
        fake_single = np.asarray(fake[0].clone().cpu().detach())
        print("fake [%s] --- max[%.4f]  mean[%.4f] min[%.4f]  dtype[%s]  shape[%s]" %
              (title, float(np.max(fake_single)), float(np.mean(fake_single)), float(np.min(fake_single)),
               fake_single.dtype, str(fake_single.shape)))

def print_g_progress_tensor(fake, title=""):
    with torch.no_grad():
        print("fake [%s] --- max[%.4f]  mean[%.4f] min[%.4f]  dtype[%s]  shape[%s]" %
              (title, float(fake.max().item()), float(fake.mean().item()), float(fake.min().item()),
               fake.dtype, str(fake.shape)))


def print_epoch_losses_summary(epoch, num_epochs, errD, errD_real, errD_fake, loss_g_d_factor, errG_d,
                               ssim_loss_g_factor, errG_ssim, errG_intensity, errG_mu):
    output_str = '[%d/%d]\tLoss_D: %.4f \tLoss_D_real: %.4f \tLoss_D_fake: %.4f'
    format_str = (epoch, num_epochs, errD, errD_real, errD_fake)
    if loss_g_d_factor != 0:
        output_str = output_str + ' \tLoss_G: %.4f'
        format_str = format_str + (errG_d.item(),)

    if ssim_loss_g_factor != 0:
        output_str = output_str + ' \tLoss_G_SSIM: %.4f'
        format_str = format_str + (errG_ssim.item(),)
    if errG_intensity:
        output_str = output_str + ' \tLoss_G_intensity: %.4f'
        format_str = format_str + (errG_intensity.item(),)
    if errG_mu:
        output_str = output_str + ' \tLoss_G_mu: %.4f'
        format_str = format_str + (errG_mu.item(),)
    print(output_str % format_str)


def print_epoch_acc_summary(epoch, num_epochs, accDfake, accDreal, accG):
    output_str = '[%d/%d]\tAcc_D_fake: %.4f \tAcc_D_real: %.4f \tAcc_G: %.4f'
    format_str = (epoch, num_epochs, accDfake, accDreal, accG)
    print(output_str % format_str)


def print_tmqi_update(Q, color):
    text = "=============== TMQI " + color + " ===============\n" + "Ours = " + str(Q)
    print(text)


def print_best_acc_error(best_errG, epoch):
    print("================ EPOCH [%d] BEST ACC G [%.4f] ===================\n" % (epoch, best_errG))


def print_net(net_name, net, opt, input_size):
    print("=================  NET %s  ==================" % net_name)
    print(net)
    dev = "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
    if net_name == "D":
        summary(net, (opt.output_dim, input_size, input_size), device=dev)
    if net_name == "G":
        summary(net, (opt.input_dim, input_size[0], input_size[1]), device=dev)
    print()


def print_opt(opt):
    print("=====================")
    print("RANDOM SEED:", opt.manual_seed)
    print("DEVICE:", opt.device)
    print("BATCH SIZE:", opt.batch_size)
    print("EPOCHS:", opt.num_epochs)
    print("MODEL:", opt.model, opt.con_operator)
    print("G LR: ", opt.G_lr)
    print("D LR: ", opt.D_lr)
    print("LOSS G D FACTOR:", opt.loss_g_d_factor)
    print("STRUCT LOSS FACTOR:", opt.ssim_loss_factor)
    print("STRUCT LOSS PYRAMID WEIGHTS:", opt.pyramid_weight_list)
    print("FACTORISED DATA:", opt.use_factorise_data)
    print("=====================\n")

def print_D_weights(weights, mul):
    print("==== D mul[%.1f] weights [%.4f, %.4f, %.4f] =====" % (mul, weights[0], weights[1], weights[2]))
