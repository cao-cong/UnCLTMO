from __future__ import print_function

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch import autograd

import Tester
import utils.data_loader_util as data_loader_util
import utils.model_save_util as model_save_util
import utils.plot_util as plot_util
from fid import fid_score
from models import struct_loss
from utils import printer, params
from einops import rearrange, repeat
from TMQI import TMQI, TMQIr
import cv2

def fspecial_gauss(size, sigma, channels):
    # Function to mimic the 'fspecial' gaussian MATLAB function
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    g = torch.from_numpy(g/g.sum()).float().unsqueeze(0).unsqueeze(0)
    return g.repeat(channels,1,1,1)

def gaussian_filter(input, win):
    out = F.conv2d(input, win, stride=1, padding=0, groups=input.shape[1])
    return out

def compute_contrast(X, win):
    b, c, h, w = X.shape
    X_reshape = X.reshape(b*c, 1, h, w)
    win = win.to(X.device)

    mu1 = gaussian_filter(X_reshape, win)
    mu1_sq = mu1.pow(2)
    sigma1_sq = gaussian_filter(X * X, win) - mu1_sq
    sigma1_sq = sigma1_sq.reshape(b, c, sigma1_sq.shape[2], sigma1_sq.shape[3])
        
    return sigma1_sq


class ContrastExtracter(torch.nn.Module):
    def __init__(self, channels=1):

        super(ContrastExtracter, self).__init__()
        self.win = fspecial_gauss(11, 1.5, 1)

    def forward(self, X):
        contrast_map = compute_contrast(X, win=self.win)
        return contrast_map

class GanTrainer:
    def __init__(self, opt, t_netG, t_netD, t_optimizerG, t_optimizerD, lr_scheduler_G, lr_scheduler_D):
        # ====== GENERAL SETTINGS ======
        self.device = opt.device
        self.isCheckpoint = opt.checkpoint
        self.checkpoint = None

        # ====== TRAINING ======
        self.batch_size = opt.batch_size
        self.num_epochs = opt.num_epochs
        self.netG = t_netG
        self.netD = t_netD
        self.optimizerD = t_optimizerD
        self.optimizerG = t_optimizerG
        self.lr_scheduler_G = lr_scheduler_G
        self.lr_scheduler_D = lr_scheduler_D
        self.real_label, self.fake_label = 1, 0
        self.epoch, self.num_iter, self.test_num_iter = 0, 0, 0
        self.d_model = opt.d_model
        self.num_D = opt.num_D
        self.d_pretrain_epochs = opt.d_pretrain_epochs
        self.pre_train_mode = False
        self.manual_d_training = opt.manual_d_training

        # ====== LOSS ======
        self.train_with_D = opt.train_with_D
        self.d_nlayers = opt.d_nlayers
        self.pyramid_weight_list = opt.pyramid_weight_list
        self.mse_loss = torch.nn.MSELoss()
        self.wind_size = opt.ssim_window_size
        self.struct_method = opt.struct_method
        if opt.ssim_loss_factor:
            self.struct_loss = struct_loss.StructLoss(window_size=opt.ssim_window_size,
                                                      pyramid_weight_list=opt.pyramid_weight_list,
                                                      pyramid_pow=False, use_c3=False,
                                                      struct_method=opt.struct_method,
                                                      crop_input=opt.add_frame,
                                                      final_shape_addition=opt.final_shape_addition)

        self.loss_g_d_factor = opt.loss_g_d_factor
        self.struct_loss_factor = opt.ssim_loss_factor
        self.errG_d, self.errG_struct, self.errG_intensity, self.errG_mu = None, None, None, None
        self.errD_real, self.errD_fake, self.errD = None, None, None
        self.accG, self.accD, self.accDreal, self.accDfake = None, None, None, None
        self.accG_counter, self.accDreal_counter, self.accDfake_counter = 0, 0, 0
        self.G_accuracy, self.D_accuracy_real, self.D_accuracy_fake = [], [], []
        self.G_loss_struct, self.G_loss_d, self.G_loss_intensity = [], [], []
        self.D_losses, self.D_loss_fake, self.D_loss_real = [], [], []
        self.adv_weight_list = opt.adv_weight_list
        self.strong_details_D_weights = opt.strong_details_D_weights
        self.basic_details_D_weights = opt.basic_details_D_weights
        self.d_weight_mul = 1.0
        self.d_weight_mul_mode = opt.d_weight_mul_mode
        #self.epoch_step1 = 50
        self.epoch_step1 = 6
        self.epoch_step2 = 9

        # ====== DATASET ======
        #self.train_data_loader_npy, self.train_data_loader_ldr = \
        #    data_loader_util.load_train_data(opt.dataset_properties, title="train")
        self.train_data_loader_npy, self.train_data_loader_ldr_pos, self.train_data_loader_ldr_neg = \
            data_loader_util.load_train_data(opt.dataset_properties, title="train")
        self.input_dim = opt.input_dim
        self.input_images_mean = opt.input_images_mean
        self.gamma_log = opt.gamma_log
        self.use_new_f = opt.use_new_f
        self.use_hist_fit = opt.use_hist_fit
        self.final_shape_addition = opt.final_shape_addition

        # ====== POST PROCESS ======
        self.to_crop = opt.add_frame
        self.normalization = opt.normalization

        # ====== SAVE RESULTS ======
        self.output_dir = opt.output_dir
        self.epoch_to_save = opt.epoch_to_save
        self.best_accG = 0
        self.tester = Tester.Tester(self.device, self.loss_g_d_factor, self.struct_loss_factor,
                                    opt)
        self.final_epoch = opt.final_epoch
        #self.fid_real_path = opt.fid_real_path
        #self.fid_res_path = opt.fid_res_path

    def train(self):
        printer.print_cuda_details(self.device.type)
        self.verify_checkpoint()
        start_epoch = self.epoch
        if self.d_pretrain_epochs:
            self.pre_train_mode = True
            print("Starting Discriminator Pre-training Loop...")
            for epoch in range(self.d_pretrain_epochs):
                self.train_epoch()
                printer.print_epoch_acc_summary(epoch, self.d_pretrain_epochs, self.accDfake, self.accDreal, self.accG)
        self.save_loss_plot(self.d_pretrain_epochs, self.output_dir)
        self.D_losses, self.D_loss_fake, self.D_loss_real = [], [], []
        self.G_accuracy, self.D_accuracy_real, self.D_accuracy_fake = [], [], []
        self.pre_train_mode = False
        self.num_iter = 0

        print("\nStarting Training Loop...")
        for epoch in range(start_epoch, self.num_epochs):
            print('epoch:{},iter:{}'.format(epoch,self.num_iter))
            #start = time.time()
            self.epoch += 1
            self.train_epoch(epoch)
            self.lr_scheduler_G.step()
            if self.train_with_D:
                self.lr_scheduler_D.step()

    def train_epoch(self, epoch):
        #self.tester.save_images_for_model(self.netG, self.output_dir, epoch, 0)
        self.accG_counter, self.accDreal_counter, self.accDfake_counter = 0, 0, 0
        epoch_iter=0
        for (h, data_hdr), (l, data_ldr_pos), (l, data_ldr_neg) in zip(enumerate(self.train_data_loader_npy, 0),
                                                                       enumerate(self.train_data_loader_ldr_pos, 0),
                                                                       enumerate(self.train_data_loader_ldr_neg, 0)):
            self.num_iter += 1
            epoch_iter += 1
            if not self.d_weight_mul_mode == "single":
                self.d_weight_mul = torch.rand(1).to(self.device)
            with autograd.detect_anomaly():
                real_ldr_pos = data_ldr_pos[params.gray_input_image_key].to(self.device)
                real_ldr_neg = data_ldr_neg[params.gray_input_image_key].to(self.device)
                hdr_input = self.get_hdr_input(data_hdr)
                hdr_original_gray_norm = data_hdr[params.original_gray_norm_key].to(self.device)
                if self.train_with_D:
                    self.train_D(hdr_input, real_ldr_pos, real_ldr_neg, epoch)
                    #if epoch<=4:
                    #    self.train_D(hdr_input, real_ldr, epoch)
                if not self.pre_train_mode:
                    self.train_G(hdr_input, hdr_original_gray_norm, real_ldr_pos, real_ldr_neg, epoch)
            #print('dataset length:{}'.format(len(self.train_data_loader_npy)))
            #print(epoch_iter)
            #if epoch_iter % ((len(self.train_data_loader_npy))//4) == 0:
            #    self.print_epoch_summary(epoch, epoch_iter)
            if epoch<4 or epoch>7:
                if epoch_iter % ((len(self.train_data_loader_npy))//4) == 0:
                    self.print_epoch_summary(epoch, epoch_iter)
            else:
                if epoch_iter % ((len(self.train_data_loader_npy))//8) == 0:
                    self.print_epoch_summary(epoch, epoch_iter)
        #self.update_accuracy()

    def train_D(self, hdr_input, real_ldr_pos, real_ldr_neg, epoch):
        """
        Update D network
        :param hdr_input: HDR images as input to G to generate fake data
        :param real_ldr: "real" LDR images as input to D
        """
        # Train with all-real batch
        self.netD.zero_grad()
        #self.D_real_pass(real_ldr.float())
        #self.D_fake_pass(hdr_input.float())
        self.D_real_fake_pass(real_ldr_pos.float(), real_ldr_neg.float(), hdr_input.float(), epoch)
        # Add the gradients from the all-real and all-fake batches
        #self.errD = self.errD_real + self.errD_fake
        # Update D
        self.optimizerD.step()
        self.D_losses.append(self.errD.item())
        #self.D_loss_fake.append(self.errD_fake.item())
        #self.D_loss_real.append(self.errD_real.item())

    def contrastive_D_loss(self, real_logits, fake_logits):
        device = real_logits.device
        real_logits, fake_logits = map(lambda t: rearrange(t, '... -> (...)'), (real_logits, fake_logits))

        def loss_half(t1, t2):
            t1 = rearrange(t1, 'i -> i ()')
            t2 = repeat(t2, 'j -> i j', i=t1.shape[0])
            t = torch.cat((t1, t2), dim=-1)
            return F.cross_entropy(t, torch.zeros(t1.shape[0], device=device, dtype=torch.long))

        return loss_half(real_logits, fake_logits) + loss_half(-fake_logits, -real_logits)

    def D_real_fake_pass(self, real_ldr_pos, real_ldr_neg, hdr_input, epoch):
        """
        Forward pass real batch through D
        """
        #print(real_ldr.shape)
        d_real_pos, _ = self.netD(real_ldr_pos.reshape(-1,real_ldr_pos.shape[2],real_ldr_pos.shape[3],real_ldr_pos.shape[4]))
        d_real_neg, _ = self.netD(real_ldr_neg.reshape(-1,real_ldr_neg.shape[2],real_ldr_neg.shape[3],real_ldr_neg.shape[4]))
        # Generate fake image batch with G
        if not self.pre_train_mode:
            fake, _ = self.netG(hdr_input, diffY=self.final_shape_addition, diffX=self.final_shape_addition)
            fake = fake.reshape(-1,fake.shape[2],fake.shape[3],fake.shape[4])
        else:
            fake = hdr_input
            fake = fake.reshape(-1,fake.shape[2],fake.shape[3],fake.shape[4])
            if self.to_crop:
                fake = data_loader_util.crop_input_hdr_batch(hdr_input, self.final_shape_addition,
                                                             self.final_shape_addition)
        # Classify all fake batch with D
        d_fake, _ = self.netD(fake.detach())
        '''#old best
        if epoch<=5:
            self.errD = self.adv_weight_list[0].float() * self.contrastive_D_loss(d_real, d_fake)
        else:
            self.errD = self.adv_weight_list[0].float() * 1e-6 * self.contrastive_D_loss(d_real, d_fake)'''
        if epoch<=self.epoch_step1:
            self.errD = self.adv_weight_list[0].float() * self.contrastive_D_loss(d_real_pos, d_fake) # ContrastiveGAN
        else:
            self.errD = self.adv_weight_list[0].float() * 1e-6 * self.contrastive_D_loss(d_real_pos, d_fake)
        self.errD.backward()

    def train_G(self, hdr_input, hdr_original_gray_norm, real_ldr_pos, real_ldr_neg, epoch):
        """
        Update G network: naturalness loss and structural loss
        :param hdr_input: HDR input (after pre-process) to be fed to G.
        :param hdr_original_gray_norm: HDR input (without pre-process) for post-process.
        """
        self.netG.zero_grad()
        # Since we just updated D, perform another forward pass of all-fake batch through D
        printer.print_g_progress(hdr_input, "hdr_inp")
        fake, fea_fake = self.netG(hdr_input.float(), diffY=self.final_shape_addition, diffX=self.final_shape_addition)
        fea_fake = fea_fake.reshape(-1,fea_fake.shape[2],fea_fake.shape[3],fea_fake.shape[4])
        fake = fake.reshape(-1,fake.shape[2],fake.shape[3],fake.shape[4])
        printer.print_g_progress(fake, "output")
        if self.train_with_D:
            d_fake_bp, d_fea_fake = self.netD(fake.float())
            d_real_pos_bp, d_fea_real_pos = self.netD(real_ldr_pos.reshape(-1,real_ldr_pos.shape[2],real_ldr_pos.shape[3],real_ldr_pos.shape[4]))
            d_real_neg_bp, d_fea_real_neg = self.netD(real_ldr_neg.reshape(-1,real_ldr_neg.shape[2],real_ldr_neg.shape[3],real_ldr_neg.shape[4]))
            _, d_fea_input = self.netD(hdr_input.float().reshape(-1,hdr_input.shape[2],hdr_input.shape[3],hdr_input.shape[4]))
            self.update_g_d_loss(d_fake_bp, d_real_pos_bp, d_real_neg_bp, d_fea_fake, d_fea_real_pos, d_fea_real_neg, d_fea_input, fea_fake, fake, hdr_input.float().reshape(-1,hdr_input.shape[2],hdr_input.shape[3],hdr_input.shape[4]),
                                 real_ldr_pos.reshape(-1,real_ldr_pos.shape[2],real_ldr_pos.shape[3],real_ldr_pos.shape[4]), real_ldr_neg.reshape(-1,real_ldr_neg.shape[2],real_ldr_neg.shape[3],real_ldr_neg.shape[4]), epoch)

        if self.manual_d_training:
            hdr_input = hdr_input.reshape(-1,hdr_input.shape[2],hdr_input.shape[3],hdr_input.shape[4])
            hdr_input = hdr_input[:, :1, :, :]
        else:
            hdr_input = hdr_input.reshape(-1,hdr_input.shape[2],hdr_input.shape[3],hdr_input.shape[4])
        hdr_original_gray_norm = hdr_original_gray_norm.reshape(-1,hdr_original_gray_norm.shape[2],hdr_original_gray_norm.shape[3],hdr_original_gray_norm.shape[4])
        self.update_struct_loss(hdr_input, hdr_original_gray_norm, fake)
        self.optimizerG.step()

    def get_hdr_input(self, data_hdr):
        hdr_input = data_hdr[params.gray_input_image_key]
        if self.manual_d_training and not self.pre_train_mode:
            weight_channel = torch.full(hdr_input.shape, self.d_weight_mul[0]).type_as(hdr_input)
            #hdr_input = torch.cat([hdr_input, weight_channel], dim=1)
            hdr_input = torch.cat([hdr_input, weight_channel], dim=2)
        return hdr_input.to(self.device)

    def update_g_d_loss(self, d_fake_bp, d_real_pos_bp, d_real_neg_bp, d_fea_fake, d_fea_real_pos, d_fea_real_neg, d_fea_input, fea_fake, fake, hdr_input, ldr_pos, ldr_neg, epoch):
        if epoch<=self.epoch_step1:
            self.errG_d = self.loss_g_d_factor * self.contrastive_D_loss(d_fake_bp, d_real_pos_bp)
            self.errG_d += self.loss_g_d_factor * 0.5 * self.infoNCE(d_fea_fake, d_fea_real_pos, d_fea_input, fake, hdr_input, cl_loss_type='InfoNCE', k=1, constant=1e-2)
            self.errG_d += self.loss_g_d_factor * 0.5 * (0.2 * self.infoNCE(d_fea_fake, d_fea_real_pos, d_fea_real_neg, fake, hdr_input, cl_loss_type='InfoNCE', k=1e3, constant=2))
            self.errG_d += self.loss_g_d_factor * 1e-6 * self.infoNCE2(fea_fake, fake, hdr_input, cl_loss_type='InfoNCE', k=1, constant=1e-2)
            l1_loss = nn.L1Loss()
            self.errG_d += self.loss_g_d_factor * 1e-6 * l1_loss(fake.mean(dim=[-1, -2]), ldr_pos.mean(dim=[-1, -2]))
            contrast_extracter = ContrastExtracter()
            fake_contrast = contrast_extracter(fake)
            ldr_contrast = contrast_extracter(ldr_pos)
            self.errG_d += self.loss_g_d_factor * 1e-6 * l1_loss(fake_contrast.mean(dim=[-1, -2]), ldr_contrast.mean(dim=[-1, -2]))
            self.errG_d += self.loss_g_d_factor * 1e-6 * self.pseudo_label_loss(fake, hdr_input)
        elif epoch<=self.epoch_step2:
            self.errG_d = self.loss_g_d_factor * 1e-6 *  self.contrastive_D_loss(d_fake_bp, d_real_pos_bp)
            self.errG_d += self.loss_g_d_factor * 0.5 * self.infoNCE(d_fea_fake, d_fea_real_pos, d_fea_input, fake, hdr_input, cl_loss_type='InfoNCE', k=1, constant=1e-2)
            self.errG_d += self.loss_g_d_factor * 0.5 * (0.2 * self.infoNCE(d_fea_fake, d_fea_real_pos, d_fea_real_neg, fake, hdr_input, cl_loss_type='InfoNCE', k=1e3, constant=2))
            self.errG_d += self.loss_g_d_factor * 0.1 * (5 * self.infoNCE2(fea_fake, fake, hdr_input, cl_loss_type='InfoNCE', k=1, constant=1e-2))
            l1_loss = nn.L1Loss()
            self.errG_d += self.loss_g_d_factor * 0.5 * (1e2 * l1_loss(fake.mean(dim=[-1, -2]), ldr_pos.mean(dim=[-1, -2])))
            contrast_extracter = ContrastExtracter()
            fake_contrast = contrast_extracter(fake)
            ldr_contrast = contrast_extracter(ldr_pos)
            self.errG_d += self.loss_g_d_factor * 0.5 * (2 * l1_loss(fake_contrast.mean(dim=[-1, -2]), ldr_contrast.mean(dim=[-1, -2])))
            self.errG_d += self.loss_g_d_factor * 1e-6 * self.pseudo_label_loss(fake, hdr_input)
        else:
            self.errG_d = self.loss_g_d_factor * 1e-6 *  self.contrastive_D_loss(d_fake_bp, d_real_pos_bp)
            l1_loss = nn.L1Loss()
            self.errG_d += self.loss_g_d_factor * 0.5 * (1e2 * l1_loss(fake.mean(dim=[-1, -2]), ldr_pos.mean(dim=[-1, -2])))
            self.errG_d += self.loss_g_d_factor * 0.5 * (1e2 * self.pseudo_label_loss(fake, hdr_input))
            tv_loss = L_TV()
            self.errG_d += self.loss_g_d_factor * 0.2 * (1e5 * tv_loss(fake))
        
        retain_graph = False
        if self.struct_loss_factor:
            retain_graph = True
        self.errG_d.backward(retain_graph=retain_graph)
        self.G_loss_d.append(self.errG_d.item())

    def pseudo_label_loss(self, fake, hdr_input):
        fake_imgs = fake.permute(0,2,3,1).detach().cpu().numpy()
        hdr_inputs = hdr_input.permute(0,2,3,1).detach().cpu().numpy()
        #split = 4 # base
        split = 2 
        ps = 256//split
        patches = []
        tmqi = TMQI()
        tmqi_n_scores = []
        for i in range(fake.shape[0]):
            for j in range(split):
                for k in range(split):
                    hdr_patch =  hdr_inputs[i,j*ps:(j+1)*ps,k*ps:(k+1)*ps,0]
                    fake_patch = fake_imgs[i,j*ps:(j+1)*ps,k*ps:(k+1)*ps,0]
                    tmqi_score, tmqi_s_score, tmqi_n_score, _, _ = tmqi(hdr_patch, fake_patch*255)
                    tmqi_n_scores.append(tmqi_n_score)
                    patches.append(fake[i:i+1,0:1,j*ps:(j+1)*ps,k*ps:(k+1)*ps])
        tmqi_n_scores_sorted = sorted(tmqi_n_scores)
        best_score = tmqi_n_scores_sorted[-1]
        pseudo_label = patches[tmqi_n_scores.index(best_score)]
        loss = 0
        l1_loss = nn.L1Loss()
        pseudo_label = pseudo_label.repeat(len(patches),1,1,1)
        patches = torch.cat(patches, 0)
        loss += l1_loss(patches.mean(dim=[-1, -2]), pseudo_label.mean(dim=[-1, -2]))
        contrast_extracter = ContrastExtracter()
        patches_contrast = contrast_extracter(patches)
        pseudo_label_contrast = contrast_extracter(pseudo_label)
        loss += l1_loss(patches_contrast.mean(dim=[-1, -2]), pseudo_label_contrast.mean(dim=[-1, -2]))
        return loss

    def infoNCE(self, fea_fake, fea_real, fea_neg, fake, hdr_input, cl_loss_type, k, constant):

        infoNCE_loss = 0

        feas_positive = []
        feas_negative = []
        fea_anchor = fea_fake
        feas_positive.append(fea_real)
        feas_negative.append(fea_neg)
        nce_loss = self.nce(fea_anchor, feas_positive, feas_negative, cl_loss_type, k, constant)
        infoNCE_loss += nce_loss

        return infoNCE_loss

    def infoNCE2(self, fea_fake, fake, hdr_input, cl_loss_type, k, constant):

        infoNCE_loss = 0

        fake_imgs = fake.permute(0,2,3,1).detach().cpu().numpy()
        hdr_inputs = hdr_input.permute(0,2,3,1).detach().cpu().numpy()
        
        tmqi = TMQI()
        tmqi_n_scores = []
        for i in range(fake.shape[0]):
            tmqi_score, tmqi_s_score, tmqi_n_score, _, _ = tmqi(hdr_inputs[i,:,:,0], fake_imgs[i,:,:,0]*255)
            tmqi_n_scores.append(tmqi_n_score)
        tmqi_n_scores_sorted = sorted(tmqi_n_scores)
        
        fea_anchor2 = fea_fake
        feas_positive2 = []
        feas_negative2 = []
        fea_positive = fea_fake[tmqi_n_scores.index(tmqi_n_scores_sorted[-1]),:,:,:].unsqueeze(0).repeat(fea_fake.shape[0],1,1,1)
        fea_negative = fea_fake[tmqi_n_scores.index(tmqi_n_scores_sorted[0]),:,:,:].unsqueeze(0).repeat(fea_fake.shape[0],1,1,1)
        feas_positive2.append(fea_positive)
        feas_negative2.append(fea_negative)
        nce_loss2 = self.nce(fea_anchor2, feas_positive2, feas_negative2, cl_loss_type, k, constant)
        infoNCE_loss += nce_loss2

        return infoNCE_loss

    def nce(self, fea_anchor, feas_positive, feas_negative, cl_loss_type, k, constant):

        loss = 0
        b, c, h, w = fea_anchor.shape

        neg_logits = []

        k = k
        c = constant

        for f_lr in feas_negative:
            neg_diff = torch.sum(
                (fea_anchor*f_lr)*(1/(c+k*torch.abs(fea_anchor-f_lr))), dim=1).mean(dim=[-1, -2]).unsqueeze(1)
            neg_logits.append(neg_diff)

        for f_hr in feas_positive:
            pos_logits = []
            pos_diff = torch.sum(
                (fea_anchor*f_hr)*(1/(c+k*torch.abs(fea_anchor-f_hr))), dim=1).mean(dim=[-1, -2]).unsqueeze(1)
            pos_logits.append(pos_diff)

            if cl_loss_type == 'InfoNCE':
                logits = torch.cat(pos_logits + neg_logits, dim=1)
                cl_loss = F.cross_entropy(logits, torch.zeros(b, device=logits.device, dtype=torch.long)) # self.ce_loss(logits)
            elif cl_loss_type == 'LMCL':
                cl_loss = self.lmcl_loss(pos_logits + neg_logits)
            else:
                raise TypeError(f'{self.args.cl_loss_type} is not found in loss/adversarial.py')
            loss += cl_loss
        return loss / len(feas_positive)

    def lmcl_loss(self, logits):
        """
        logits: BXK, the first column is the positive similarity
        """
        pos_sim = logits[0]
        neg_sim = torch.cat(logits[1:], dim=1)
        pos_logits = pos_sim.exp()  # Bx1
        neg_logits = torch.sum(neg_sim.exp(), dim=1, keepdim=True)  # Bx1
        loss = -torch.log(pos_logits / neg_logits).mean()
        return loss

    def update_struct_loss(self, hdr_input, hdr_input_original_gray_norm, fake):
        #hdr_input_original_gray_norm = hdr_input_original_gray_norm.reshape(-1,hdr_input_original_gray_norm.shape[2],hdr_input_original_gray_norm.shape[3],hdr_input_original_gray_norm.shape[4])
        if self.struct_loss_factor:
            #print(fake.shape)
            #print(hdr_input_original_gray_norm.shape)
            #print(hdr_input.shape)
            self.errG_struct = self.struct_loss_factor * self.struct_loss(fake, hdr_input_original_gray_norm,
                                                                          hdr_input, self.pyramid_weight_list)
            self.errG_struct.backward()
            self.G_loss_struct.append(self.errG_struct.item())

    def verify_checkpoint(self):
        if self.isCheckpoint:
            print("Loading model...")
            self.load_model()
            print("Model was loaded")
            print()

    def save_loss_plot(self, epoch, output_dir):
        loss_path = os.path.join(output_dir, "loss_plot")
        acc_path = os.path.join(output_dir, "accuracy")
        acc_file_name = "acc" + str(epoch)
        if self.pre_train_mode:
            acc_file_name = "pretrain_" + acc_file_name
        plot_util.plot_general_accuracy(self.G_accuracy, self.D_accuracy_fake, self.D_accuracy_real,
                                        acc_file_name,
                                        self.epoch, acc_path)
        if not self.pre_train_mode:
            plot_util.plot_general_losses(self.G_loss_d, self.G_loss_struct, self.G_loss_intensity, self.D_loss_fake,
                                          self.D_loss_real, "summary epoch_=_" + str(epoch), self.num_iter, loss_path,
                                          (self.loss_g_d_factor != 0), (self.struct_loss_factor != 0))

    def load_model(self):
        if self.isCheckpoint:
            self.checkpoint = torch.load(params.models_save_path)
            self.epoch = self.checkpoint['epoch']
            self.netD.load_state_dict(self.checkpoint['modelD_state_dict'])
            self.netG.load_state_dict(self.checkpoint['modelG_state_dict'])
            self.optimizerD.load_state_dict(self.checkpoint['optimizerD_state_dict'])
            self.optimizerG.load_state_dict(self.checkpoint['optimizerG_state_dict'])
            self.netD.train()
            self.netG.train()

    def update_accuracy(self):
        len_hdr_train_dset = len(self.train_data_loader_npy.dataset)
        len_ldr_train_dset = len(self.train_data_loader_ldr.dataset)
        self.accG = self.accG_counter / len_hdr_train_dset
        self.accDreal = self.accDreal_counter / len_ldr_train_dset
        self.accDfake = self.accDfake_counter / len_ldr_train_dset

        if self.d_model == "patchD":
            self.accG = self.accG / params.patchD_map_dim[self.d_nlayers]
            self.accDreal = self.accDreal / params.patchD_map_dim[self.d_nlayers]
            self.accDfake = self.accDfake / params.patchD_map_dim[self.d_nlayers]
        elif "multiLayerD_patchD" == self.d_model:
            self.accG = self.accG / params.get_multiLayerD_map_dim(num_D=self.num_D, d_nlayers=self.d_nlayers)
            self.accDreal = self.accDreal / params.get_multiLayerD_map_dim(num_D=self.num_D, d_nlayers=self.d_nlayers)
            self.accDfake = self.accDfake / params.get_multiLayerD_map_dim(num_D=self.num_D, d_nlayers=self.d_nlayers)
        elif "multiLayerD_dcgan" == self.d_model or "multiLayerD_simpleD" == self.d_model:
            self.accG = self.accG / self.num_D
            self.accDreal = self.accDreal / self.num_D
            self.accDfake = self.accDfake / self.num_D

        self.G_accuracy.append(self.accG)
        self.D_accuracy_real.append(self.accDreal)
        self.D_accuracy_fake.append(self.accDfake)

    def print_epoch_summary(self, epoch, epoch_iter):
        #print("Single [[epoch]] iteration took [%.4f] seconds\n" % (time.time() - start))
        if self.train_with_D:
            printer.print_epoch_losses_summary(epoch, self.num_epochs, self.errD.item(), 0,
                                               0, self.loss_g_d_factor, self.errG_d,
                                               self.struct_loss_factor, self.errG_struct, self.errG_intensity,
                                               self.errG_mu)
        else:
            printer.print_epoch_losses_summary(epoch, self.num_epochs, 0, 0, 0, 0, 0,
                                               self.struct_loss_factor, self.errG_struct, self.errG_intensity,
                                               self.errG_mu)
        #printer.print_epoch_acc_summary(epoch, self.num_epochs, self.accDfake, self.accDreal, self.accG)
        #if epoch % self.epoch_to_save == 0:
        self.tester.save_test_images(epoch, epoch_iter, self.output_dir, self.input_images_mean, self.netD, self.netG,
                                     0, self.struct_loss, self.num_epochs, self.to_crop)
        #self.save_loss_plot(epoch, self.output_dir)
        #print('process hdr images')
        self.tester.save_images_for_model(self.netG, self.output_dir, epoch, epoch_iter)
        #print('process hdr images done')
        model_save_util.save_model(params.models_save_path, epoch, epoch_iter, self.output_dir, self.netG, self.optimizerG,
                                    self.netD, self.optimizerD)
        # if epoch == self.final_epoch:
        #     model_save_util.save_model(params.models_save_path, epoch, self.output_dir, self.netG, self.optimizerG,
        #                                self.netD, self.optimizerD)
        #     self.save_data_for_assessment()

    def save_data_for_assessment(self):
        model_params = model_save_util.get_model_params(self.output_dir,
                                                        train_settings_path=os.path.join(self.output_dir,
                                                                                         "run_settings.npy"))
        model_params["test_mode_f_factor"] = False
        model_params["test_mode_frame"] = False
        net_path = os.path.join(self.output_dir, "models", "net_epoch_" + str(self.final_epoch) + ".pth")
        self.run_model_on_path("open_exr_exr_format", "exr", model_params, net_path)
        self.run_model_on_path("npy_pth", "npy", model_params, net_path)
        # self.run_model_on_path("test_source", "exr", model_params, net_path)

    def run_model_on_path(self, data_source, data_format, model_params, net_path):
        input_images_path = model_save_util.get_hdr_source_path(data_source)
        f_factor_path = model_save_util.get_f_factor_path(data_source, self.gamma_log, self.use_new_f,
                                                          self.use_hist_fit)
        output_images_path = os.path.join(self.output_dir, data_format + "_" + str(self.final_epoch))
        if not os.path.exists(output_images_path):
            os.mkdir(output_images_path)
        output_images_path_color_stretch = os.path.join(output_images_path, "color_stretch")
        if not os.path.exists(output_images_path_color_stretch):
            os.mkdir(output_images_path_color_stretch)
        model_save_util.run_model_on_path(model_params, self.device, net_path, input_images_path,
                                          output_images_path, f_factor_path, None,
                                          self.final_shape_addition)
        if data_format == "npy":
            #fid_res_color_stretch = fid_score.calculate_fid_given_paths(
            #    [self.fid_real_path, output_images_path_color_stretch],
            #    batch_size=20, cuda=False, dims=768)
            if os.path.exists(self.fid_res_path):
                data = np.load(self.fid_res_path, allow_pickle=True)[()]
                data[model_params["model_name"]] = fid_res_color_stretch
                np.save(self.fid_res_path, data)
            else:
                my_res = {model_params["model_name"]: fid_res_color_stretch}
                np.save(self.fid_res_path, my_res)


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

class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size