import os
import cv2
import numpy as np

import torch
import torch.nn.functional as F

import tranforms
import utils.data_loader_util as data_loader_util
import utils.hdr_image_util as hdr_image_util
import utils.plot_util as plot_util
from utils import printer, adaptive_lambda
#import time
from TMQI import TMQI, TMQIr
import imageio


class Tester:
    def __init__(self, device, loss_g_d_factor_, ssim_loss_g_factor_, args):
        self.args = args
        self.to_crop = args.add_frame
        self.data_trc = args.data_trc
        self.final_shape_addition = args.final_shape_addition
        self.test_data_loader_npy, self.test_data_loader_ldr = \
            data_loader_util.load_test_data(args.dataset_properties, title="test")
        self.accG_counter, self.accDreal_counter, self.accDfake_counter = 0, 0, 0
        self.G_accuracy_test, self.D_accuracy_real_test, self.D_accuracy_fake_test = [], [], []
        self.real_label, self.fake_label = 1, 0
        self.device = device
        self.test_D_losses, self.test_D_loss_fake, self.test_D_loss_real = [], [], []
        self.test_G_losses_d, self.test_G_loss_ssim = [], []
        self.ssim_loss_g_factor = ssim_loss_g_factor_
        self.loss_g_d_factor = loss_g_d_factor_
        self.test_num_iter = 0
        self.use_contrast_ratio_f = args.use_contrast_ratio_f
        self.extensions = [".hdr", ".dng", ".exr", ".npy"]
        self.test_original_hdr_images = self.load_original_test_hdr_images(args.test_dataroot_original_hdr)


    def load_original_test_hdr_images(self, root):
        f_factor_path = adaptive_lambda.calc_lambda(self.args.f_factor_path, self.extensions, root,
                                    self.args.mean_hist_path, self.args.lambdas_path, self.args.bins)
        original_hdr_images = []
        counter = 1
        for img_name in os.listdir(root):
            im_path = os.path.join(root, img_name)
            print(img_name)
            rgb_img, gray_im_log, f_factor = \
                data_loader_util.hdr_preprocess(im_path,
                                               self.args.factor_coeff, train_reshape=False,
                                               f_factor_path=f_factor_path,
                                               data_trc=self.data_trc)
            rgb_img, gray_im_log = tranforms.hdr_im_transform(rgb_img), tranforms.hdr_im_transform(gray_im_log)
            rgb_img, diffY, diffX = data_loader_util.resize_im(rgb_img, self.to_crop, self.final_shape_addition)
            gray_im_log, diffY, diffX = data_loader_util.resize_im(gray_im_log, self.to_crop, self.final_shape_addition)
            original_hdr_images.append({'im_name': os.path.splitext(img_name)[0],
                                        'im_hdr_original': rgb_img,
                                        'im_log_normalize_tensor': gray_im_log,
                                        'epoch': 0, 'diffX': diffX, 'diffY': diffY})
            counter += 1
        return original_hdr_images

    def update_test_loss(self, netD, criterion, ssim_loss, b_size, num_epochs, first_b_tonemap, fake, hdr_input, epoch):
        self.accG_counter, self.accDreal_counter, self.accDfake_counter = 0, 0, 0
        with torch.no_grad():
            test_D_output_on_real = netD(first_b_tonemap.detach()).view(-1)
            self.accDreal_counter += (test_D_output_on_real > 0.5).sum().item()

            real_label = torch.full(test_D_output_on_real.shape, self.real_label, device=self.device)
            test_errD_real = criterion(test_D_output_on_real, real_label)
            self.test_D_loss_real.append(test_errD_real.item())

            fake_label = torch.full(test_D_output_on_real.shape, self.fake_label, device=self.device)
            output_on_fake = netD(fake.detach()).view(-1)
            self.accDfake_counter += (output_on_fake <= 0.5).sum().item()

            test_errD_fake = criterion(output_on_fake, fake_label)
            test_loss_D = test_errD_real + test_errD_fake
            self.test_D_loss_fake.append(test_errD_fake.item())
            self.test_D_losses.append(test_loss_D.item())

            # output_on_fakake = self.netD(fake.detach()).view(-1)
            self.accG_counter += (output_on_fake > 0.5).sum().item()
            # if self.loss_g_d_factor != 0:
            test_errGd = criterion(output_on_fake, real_label)
            self.test_G_losses_d.append(test_errGd.item())
            if self.ssim_loss_g_factor != 0:
                if self.to_crop:
                    hdr_input = data_loader_util.crop_input_hdr_batch(hdr_input, self.final_shape_addition,
                                                                      self.final_shape_addition)
                fake_rgb_n = fake + 1
                hdr_input_rgb_n = hdr_input + 1
                test_errGssim = self.ssim_loss_g_factor * (1 - ssim_loss(fake_rgb_n, hdr_input_rgb_n))
                self.test_G_loss_ssim.append(test_errGssim.item())
            self.update_accuracy()
            printer.print_test_epoch_losses_summary(num_epochs, epoch, test_loss_D, test_errGd, self.accDreal_test,
                                                    self.accDfake_test, self.accG_test)

    @staticmethod
    def get_fake_test_images(first_b_hdr, netG):
        with torch.no_grad():
            fake, _ = netG(first_b_hdr.detach())
            return fake

    def update_accuracy(self):
        len_hdr_test_dset = len(self.test_data_loader_npy.dataset)
        len_ldr_test_dset = len(self.test_data_loader_ldr.dataset)
        self.accG_test = self.accG_counter / len_hdr_test_dset
        self.accDreal_test = self.accDreal_counter / len_ldr_test_dset
        self.accDfake_test = self.accDfake_counter / len_ldr_test_dset
        self.G_accuracy_test.append(self.accG_test)
        self.D_accuracy_real_test.append(self.accDreal_test)
        self.D_accuracy_fake_test.append(self.accDfake_test)

    def save_test_loss(self, epoch, out_dir):
        acc_path = os.path.join(out_dir, "accuracy")
        loss_path = os.path.join(out_dir, "loss_plot")
        plot_util.plot_general_losses(self.test_G_losses_d, self.test_G_loss_ssim,
                                      self.test_D_loss_fake, self.test_D_loss_real,
                                      "TEST epoch loss = " + str(epoch), self.test_num_iter, loss_path,
                                      (self.loss_g_d_factor != 0), (self.ssim_loss_g_factor != 0))

        plot_util.plot_general_accuracy(self.G_accuracy_test, self.D_accuracy_fake_test, self.D_accuracy_real_test,
                                        "TEST epoch acc = " + str(epoch), epoch, acc_path)

    def save_test_images(self, epoch, epoch_iter, out_dir, input_images_mean, netD, netG,
                         criterion, ssim_loss, num_epochs, add_frame):
        out_dir = os.path.join(out_dir, "result_images")
        new_out_dir = os.path.join(out_dir, "images_epoch" + str(epoch) + "_iter" + str(epoch_iter))

        if not os.path.exists(new_out_dir):
            os.mkdir(new_out_dir)

        self.test_num_iter += 1
        test_real_batch = next(iter(self.test_data_loader_ldr))
        test_real_first_b = test_real_batch["input_im"].to(self.device)

        test_hdr_batch = next(iter(self.test_data_loader_npy))
        # test_hdr_batch_image = test_hdr_batch[params.image_key].to(self.device)
        test_hdr_batch_image = test_hdr_batch["input_im"].to(self.device)
        fake1 = self.get_fake_test_images(test_hdr_batch_image, netG)
        
        #test_hdr_batch = test_hdr_batch.reshape(-1,test_hdr_batch.shape[2],test_hdr_batch.shape[3],test_hdr_batch.shape[4])
        #test_real_batch = test_real_batch.reshape(-1,test_real_batch.shape[2],test_real_batch.shape[3],test_real_batch.shape[4])
        fake1 = fake1.reshape(-1,fake1.shape[2],fake1.shape[3],fake1.shape[4])
        plot_util.save_groups_images(test_hdr_batch, test_real_batch, fake1, fake1,
                                     new_out_dir, len(self.test_data_loader_npy.dataset), epoch,
                                     input_images_mean)

    def test_big_size_image(self, input_data, model, apply_crop, diffY, diffX, patch_h = 256, patch_w = 256, patch_h_overlap = 64, patch_w_overlap = 64):

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

    def load_inference(self, im_path, f_factor_path, factor_coeff, device):
        #print(f_factor_path)
        #f_factor_path = "lambda_data/input_images_lambdas_HDRSdataset.npy"
        #f_factor_path = "lambda_data/input_images_lambdas_Kalantari13_1.npy"
        data = np.load(f_factor_path, allow_pickle=True)[()]
        #f_factor = data[os.path.splitext(os.path.basename(im_path))[0]] * 255 * factor_coeff
        f_factor = data[im_path.split('/')[-2]] * 255 * factor_coeff
        rgb_img = hdr_image_util.read_hdr_image(im_path)
        rgb_original = rgb_img.copy()
        #print('-------------------')
        #print(rgb_img.shape)
        #rgb_img = hdr_image_util.reshape_image(rgb_img, train_reshape=False)
        #print(rgb_img.shape)
        rgb_img = tranforms.hdr_im_transform(rgb_img).to(device)
        #print(rgb_img.shape)
        # shift for exr format
        if rgb_img.min() < 0:
            rgb_img = rgb_img - rgb_img.min()
        gray_im = hdr_image_util.to_gray_tensor(rgb_img).to(device)
        gray_im = gray_im - gray_im.min()
        gray_im = torch.log10((gray_im / gray_im.max()) * f_factor + 1)
        gray_im = gray_im / gray_im.max()
        return rgb_original, rgb_img, gray_im, f_factor

    def save_images_for_model(self, netG, out_dir, epoch, epoch_iter):

        f_factor_path = "activate_trained_model/lambda_data/input_images_lambdas.npy"
        input_images_path = "../../data/tone_mapping/test_HDRvideo"
        scenes = os.listdir(input_images_path)
        tmqi_final = 0
        warp_error_final = 0
        warp_error_final2 = 0
        for scene in scenes:
            names = os.listdir(os.path.join(input_images_path, scene))
            names = sorted(names)[:6]
            #print('-------')
            #print(names)
            #print('-------')
            im_paths = []
            im_names = []
            for img_name in names:
                im_path = os.path.join(input_images_path, scene, img_name)
                print("processing [%s]" % img_name)
                im_paths.append(im_path)
                im_names.append(os.path.splitext(img_name)[0])
            tmqi_scene, warp_error_scene, warp_error_scene2 = self.eval_on_video(netG, im_paths, self.device, im_names, f_factor_path, self.args.final_shape_addition, epoch, epoch_iter)
            tmqi_final += tmqi_scene
            warp_error_final += warp_error_scene
            warp_error_final2 += warp_error_scene2
        tmqi_final = tmqi_final/len(scenes)
        warp_error_final = warp_error_final/len(scenes)
        warp_error_final2 = warp_error_final2/len(scenes)

        out_dir = os.path.join(out_dir, "model_results", "epoch"+str(epoch)+"_iter"+str(epoch_iter)+"_m1st"+str(tmqi_final)+"_m2nd"+str(warp_error_final)+"_m3rd"+str(warp_error_final2))
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
            print("Directory ", out_dir, " created")
        with torch.no_grad():
            for im_and_q in self.test_original_hdr_images:
                print(im_and_q["im_name"])
                im_hdr_original = im_and_q['im_hdr_original']
                im_log_normalize_tensor = im_and_q['im_log_normalize_tensor'].unsqueeze(0).unsqueeze(0).to(self.device)
                im_log_normalize_frames_tensor = []
                for i in range(4):
                    im_log_normalize_frames_tensor.append(im_log_normalize_tensor)
                im_log_normalize_frames_tensor = torch.cat(im_log_normalize_frames_tensor, 1)
                #printer.print_g_progress(im_log_normalize_tensor, "input_tester")
                #with torch.no_grad():
                #    fake = netG(im_log_normalize_tensor, apply_crop=self.to_crop, diffY=im_and_q['diffY'], diffX=im_and_q['diffX'])
                #    print("fake", fake.max(), fake.mean(), fake.min())
                fake = self.test_big_size_image(im_log_normalize_frames_tensor, model=netG, apply_crop=self.to_crop, diffY=im_and_q['diffY'], diffX=im_and_q['diffX'], patch_h = 256, patch_w = 256, patch_h_overlap = 64, patch_w_overlap = 64)
                fake = fake.reshape(-1,fake.shape[2],fake.shape[3],fake.shape[4])[-1]
                fake2 = fake.clamp(0.005, 0.995)
                fake_im_gray_stretch = (fake2 - fake2.min()) / (fake2.max() - fake2.min())
                fake_im_color2 = hdr_image_util.back_to_color_tensor(im_hdr_original, fake_im_gray_stretch[0],
                                                                     self.device)
                h, w = fake_im_color2.shape[1], fake_im_color2.shape[2]
                im_max = fake_im_color2.max()
                fake_im_color2 = F.interpolate(fake_im_color2.unsqueeze(dim=0), size=(h - im_and_q['diffY'],
                                                                                      w - im_and_q['diffX']),
                                               mode='bicubic',
                                               align_corners=False).squeeze(dim=0).clamp(min=0, max=im_max)
                hdr_image_util.save_gray_tensor_as_numpy_stretch(fake_im_color2, out_dir + "/color_stretch",
                                                                 im_and_q["im_name"] + "_color_stretch")

    def eval_on_video(self, G_net, im_paths, device, im_names, f_factor_path, final_shape_addition, epoch, epoch_iter):
        gray_im_logs = []
        rgb_imgs = []
        rgb_imgs_original = []
        #gray_ims_log = []
        for im_path in im_paths:
            rgb_img_original, rgb_img, gray_im_log, f_factor = self.load_inference(im_path, f_factor_path, self.args.factor_coeff, device)
            rgb_imgs_original.append(rgb_img_original)
            #gray_ims_log.append(gray_im_log)
            #print('-------------------')
            #print(rgb_img.shape)
            #print(gray_im_log.shape)
            rgb_img, diffY, diffX = data_loader_util.resize_im(rgb_img, self.args.add_frame, final_shape_addition)
            gray_im_log, diffY, diffX = data_loader_util.resize_im(gray_im_log, self.args.add_frame, final_shape_addition)
            #diffY=0
            #diffX=0
            #print(gray_im_log.shape)
            gray_im_logs.append(gray_im_log.unsqueeze(0).unsqueeze(0))
            rgb_imgs.append(rgb_img)
        gray_im_logs = torch.cat(gray_im_logs, 1)
        
        c, h, w = gray_im_log.size()
        fakes = self.test_big_size_image(input_data=gray_im_logs, model=G_net, apply_crop=False, diffY=diffY, diffX=diffX, patch_h = 256, patch_w = 256, patch_h_overlap = 64, patch_w_overlap = 64)

        tmqi_scene = 0
        tmqi = TMQI()
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
            ldr_result = self.tensor_to_numpy(fake_im_color2)
            output_path = 'output_epoch{}_iter{}'.format(epoch, epoch_iter)
            #save_path = os.path.join(output_path, im_paths[i].split('/')[-2])
            #if not os.path.isdir(save_path):
            #    os.makedirs(save_path)
            #imageio.imwrite(os.path.join(save_path, im_names[i] + ".png"), ldr_result, format='PNG-FI')
            #imageio.imwrite(os.path.join(save_path, im_names[i] + "_hdr_rgb_gaamma.png"), np.uint8((rgb_imgs_original[i]/rgb_imgs_original[i].max())**(1/2.2)*255), format='PNG-FI')
            #gray = gray_ims_log[i].permute(1,2,0).cpu().numpy()
            #imageio.imwrite(os.path.join(save_path, im_names[i] + "_hdr_y_gaamma.png"), np.uint8((gray/gray.max())**(1/2.2)*255), format='PNG-FI')
            #print(ldr_result.astype(np.float32).shape)
            #print(rgb_imgs_original[i].shape)
            #print(ldr_result.astype(np.float32).dtype)
            #print(rgb_imgs_original[i].dtype)
            if i == 0:
                img0_target = ldr_result
            if i == 1:
                img1_to_align = ldr_result
            score, s_score, n_score, _, _ = tmqi(rgb_imgs_original[i], ldr_result.astype(np.float32))
            print(score)
            tmqi_scene += score
        tmqi_scene = tmqi_scene/len(im_paths)

        img0_name = os.path.basename(im_paths[0]).replace('.npy','_L1L0TM.png')
        img1_name = os.path.basename(im_paths[1]).replace('.npy','_L1L0TM.png')
        scene = im_paths[0].split('/')[-2]
        img0_path = os.path.join('../output_testvideoall_L1L0', scene, img0_name)
        img1_path = os.path.join('../output_testvideoall_L1L0', scene, img1_name)
        img0 = cv2.imread(img0_path)
        img1 = cv2.imread(img1_path)
        flow = compute_flow(img1, img0)
        img1_aligned = align_frames(img1_to_align, flow)
        img0_target = img0_target.astype(np.float32)/255.0
        img1_aligned = img1_aligned.astype(np.float32)/255.0
        warp_error_scene = np.mean(np.power(img1_aligned[32:-32,32:-32,:]-img0_target[32:-32,32:-32,:], 2))
        warp_error_scene2 = np.mean(np.abs(img1_aligned[32:-32,32:-32,:]-img0_target[32:-32,32:-32,:])/(1e-8+img1_aligned[32:-32,32:-32,:]+img0_target[32:-32,32:-32,:]))

        return tmqi_scene, warp_error_scene, warp_error_scene2
        
    def tensor_to_numpy(self, tensor):
        tensor = tensor.clamp(0,1).clone().permute(1, 2, 0).detach().cpu().numpy()
        tensor_0_1 = np.squeeze(tensor)
        tensor_0_1 = self.to_0_1_range_outlier(tensor_0_1)
        im = (tensor_0_1 * 255).astype('uint8')
        return im
        
    def to_0_1_range_outlier(self, im):
        im_max = np.percentile(im, 99.0)
        # TODO: check this parameter
        # im_min = np.percentile(im, 1.0)
        im_min = np.percentile(im, 0.1)
        if np.max(im) - np.min(im) == 0:
            im = (im - im_min) / (im_max - im_min + params.epsilon)
        else:
            im = (im - im_min) / (im_max - im_min)
        return np.clip(im, 0, 1)


# Parameters of the motion estimation algorithms
def warp_flow(img, flow):
    '''
        Applies to img the transformation described by flow.
    '''
    assert len(flow.shape) == 3 and flow.shape[-1] == 2

    hf, wf = flow.shape[:2]
    # flow 		= -flow
    flow[:, :, 0] += np.arange(wf)
    flow[:, :, 1] += np.arange(hf)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

def estimate_invflow(img0, img1, me_algo):
    '''
        Estimates inverse optical flow by using the me_algo algorithm.
    '''
    # # # img0, img1 have to be uint8 grayscale
    assert img0.dtype == 'uint8' and img1.dtype == 'uint8'

    # Create estimator object
    if me_algo == "DeepFlow":
        of_estim = cv2.optflow.createOptFlow_DeepFlow()
    elif me_algo == "SimpleFlow":
        of_estim = cv2.optflow.createOptFlow_SimpleFlow()
    elif me_algo == "TVL1":
        of_estim = cv2.DualTVL1OpticalFlow_create()
    else:
        raise Exception("Incorrect motion estimation algorithm")

    # Run flow estimation (inverse flow)
    flow = of_estim.calc(img1, img0, None)
#	flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    return flow

def compute_flow(img_to_align, img_source, mc_alg='DeepFlow'):
    '''
        Applies to img_to_align a transformation which converts it into img_source.
        Args:
            img_to_align: HxWxC image
            img_source: HxWxC image
            mc_alg: selects between DeepFlow, SimpleFlow, and TVL1. DeepFlow runs by default.
        Returns:
            HxWxC aligned image
    '''

    # make sure images are uint8 in the [0, 255] range
    if img_source.max() <= 1.0:
        img_source = (img_source*255).clip(0, 255)
    img_source = img_source.astype(np.uint8)
    if img_to_align.max() <= 1.0:
        img_to_align = (img_to_align*255).clip(0, 255)
    img_to_align = img_to_align.astype(np.uint8)

    img0 = img_to_align[:, :, 0]
    img1 = img_source[:, :, 0]
    out_img = None

    # Align frames according to selection in mc_alg
    flow = estimate_invflow(img0, img1, mc_alg)

    return flow

def align_frames(img_to_align, flow):
    '''
        Applies to img_to_align a transformation which converts it into img_source.
        Args:
            img_to_align: HxWxC image
            img_source: HxWxC image
            mc_alg: selects between DeepFlow, SimpleFlow, and TVL1. DeepFlow runs by default.
        Returns:
            HxWxC aligned image
    '''

    # make sure images are uint8 in the [0, 255] range
    if img_to_align.max() <= 1.0:
        img_to_align = (img_to_align*255).clip(0, 255)
    img_to_align = img_to_align.astype(np.uint8)

    # rectifier
    out_img = warp_flow(img_to_align, flow)

    return out_img