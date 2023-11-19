# full assembly of the sub-parts to form the complete net
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from .gcn_lib import Grapher_noBN, act_layer

from .unet_parts import *
from models import Blocks
import utils.printer
from utils import data_loader_util

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            #nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            #nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x#.reshape(B, C, N, 1)

class GCNBlock(torch.nn.Module):
    def __init__(self, ch=32):
        super(GCNBlock, self).__init__()
        self.k = 9 # neighbor num (default:9)
        self.conv = 'mr' # graph conv layer {edge, mr}
        self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
        self.norm = None # batch or instance normalization {batch, instance}
        self.bias = True # bias of conv layer True or False
        self.dropout = 0.0 # dropout rate
        self.use_dilation = True # use dilated knn or not
        self.epsilon = 0.2 # stochastic epsilon for gcn
        self.stochastic = False # stochastic for gcn, True or False
        self.drop_path = 0.1
        self.blocks = [1] # number of basic blocks in the module
        self.channels = [ch] # number of channels of deep features

        self.n_blocks = sum(self.blocks)
        reduce_ratios = [1]
        dpr = [x.item() for x in torch.linspace(self.drop_path/2, self.drop_path, self.n_blocks)]  # stochastic depth decay rule 
        num_knn = [int(x.item()) for x in torch.linspace(self.k, self.k, self.n_blocks)]  # number of knn's k
        max_dilation = 49 // max(num_knn)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.channels[0], 12//1, 12//1))
        HW = 12 // 1 * 12 // 1

        self.module = nn.ModuleList([])
        idx = 0
        for i in range(len(self.blocks)):
            for j in range(self.blocks[i]):
                self.module += [
                    Seq(Grapher_noBN(self.channels[i], num_knn[idx], min(idx // 4 + 1, max_dilation), self.conv, self.act, self.norm,
                                    self.bias, self.stochastic, self.epsilon, reduce_ratios[i], n=HW, drop_path=dpr[idx],
                                    relative_pos=True),
                          FFN(self.channels[i], self.channels[i], act=self.act, drop_path=dpr[idx])
                         )]
                idx += 1
        self.module = Seq(*self.module)

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        x = inputs + self.pos_embed
        B, C, H, W = x.shape
        for i in range(len(self.module)):
            x = self.module[i](x)

        return x

def fspecial_gauss(size, sigma, channels):
    # Function to mimic the 'fspecial' gaussian MATLAB function
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    g = torch.from_numpy(g/g.sum()).float().unsqueeze(0).unsqueeze(0)
    return g.repeat(channels,1,1,1)

def gaussian_filter(input, win):
    out = F.conv2d(input, win, stride=1, padding=0, groups=1)
    return out

def compute_contrast(X, win):
    b, c, h, w = X.shape
    X_reshape = X.reshape(b*c, 1, h, w)
    win = win.to(X.device)

    mu1 = gaussian_filter(X_reshape, win)
    mu1_sq = mu1.pow(2)
    #sigma1_sq = gaussian_filter(X * X, win) - mu1_sq
    sigma1_sq = gaussian_filter(X_reshape * X_reshape, win) - mu1_sq
    sigma1_sq = sigma1_sq.reshape(b, c, sigma1_sq.shape[2], sigma1_sq.shape[3])
        
    return sigma1_sq

class ContrastExtracter(torch.nn.Module):
    def __init__(self, channels=1):

        super(ContrastExtracter, self).__init__()
        self.win = fspecial_gauss(11, 1.5, channels)

    def forward(self, X):
        contrast_map = compute_contrast(X, win=self.win)
        return contrast_map

class UNet(nn.Module):
    def __init__(self, n_channels, output_dim, last_layer, depth, layer_factor, con_operator, filters, bilinear,
                 network, dilation, to_crop, unet_norm, stretch_g, activation, doubleConvTranspose,
                 padding_mode, convtranspose_kernel, up_mode=True, recurrent_ch_ratio=1/32):
        super(UNet, self).__init__()
        self.to_crop = to_crop
        self.con_operator = con_operator
        self.network = network
        down_ch = filters
        self.depth = depth
        padding = 1
        if doubleConvTranspose or up_mode:
            padding = 0
        self.padding = padding
        self.up_mode = up_mode
        self.inc = inconv(n_channels, down_ch, unet_norm, activation, padding, padding_mode, up_mode, doubleConvTranspose)
        ch = down_ch
        self.down_path = nn.ModuleList()
        for i in range(self.depth - 1):
            self.down_path.append(
                down(ch, ch * 2, network, dilation=dilation, unet_norm=unet_norm, activation=activation,
                     padding=padding, padding_mode=padding_mode, up_mode=up_mode, doubleConvTranspose=doubleConvTranspose)
            )
            ch = ch * 2
            if network == params.torus_network:
                dilation = dilation * 2
        self.down_path.append(last_down(ch, ch, network, dilation=dilation, unet_norm=unet_norm,
                                        activation=activation, padding=padding, padding_mode=padding_mode,
                                        up_mode=up_mode, doubleConvTranspose=doubleConvTranspose))

        self.gcn = GCNBlock(ch)

        self.recurrent_ch_ratio = recurrent_ch_ratio
        self.up_path = nn.ModuleList()
        output_padding = 0
        for i in range(self.depth):
            in_ch = ch * layer_factor
            if con_operator == params.square_and_square_root_manual_d:
                in_ch += 1
            if i >= self.depth - 2:
                self.up_path.append(
                    up(in_ch, down_ch, bilinear, layer_factor, network,
                       dilation=dilation, unet_norm=unet_norm, activation=activation,
                       doubleConvTranspose=doubleConvTranspose, padding=padding, padding_mode=padding_mode,
                       convtranspose_kernel=convtranspose_kernel, up_mode=up_mode)
                )
            else:
                # if i == 1 and not (up_mode):
                #     output_padding=1
                self.up_path.append(
                    up(in_ch, ch // 2, bilinear, layer_factor, network,
                       dilation=dilation, unet_norm=unet_norm, activation=activation,
                       doubleConvTranspose=doubleConvTranspose, padding=padding, padding_mode=padding_mode,
                       convtranspose_kernel=convtranspose_kernel, up_mode=up_mode, output_padding1=output_padding)
                )
            ch = ch // 2
            if network == params.torus_network:
                dilation = dilation // 2
        self.outc = outconv(down_ch, output_dim)
        self.last_sig = None
        if last_layer == 'tanh':
            self.last_sig = nn.Tanh()
        if last_layer == "sigmoid":
            self.last_sig = nn.Sigmoid()
        if last_layer == 'msig':
            self.last_sig = Blocks.MySig(3)
        # else:

        if stretch_g != "none":
            stretch_options = {"batchMax": Blocks.BatchMaxNormalization(),
                               "instanceMinMax": Blocks.MinMaxNormalization()}
            # self.clip = Blocks.Clip()
            self.stretch = stretch_options[stretch_g]
        else:
            self.stretch = None

        self.contrast_extracter = ContrastExtracter()

    def forward(self, x, apply_crop=True, diffY=0, diffX=0):
        #print(x.shape)
        output_results = []
        endecode_results_last = []
        features = []
        for k in range(x.shape[1]):
            x_frame = x[:,k,:,:,:]
            endecode_results_curr = []
                
            d_weight_mul = 1.0
            if self.con_operator == params.square_and_square_root_manual_d:
                d_weight_mul = x_frame[0, 1, 0, 0]
            # print("d_weight_mul", d_weight_mul)
            next_x = self.inc(x_frame)
            x_results = [next_x]

            endecode_results_curr.append(next_x[:,:int(next_x.shape[1]*self.recurrent_ch_ratio),:,:])
            
            for i, down_layer in enumerate(self.down_path):
                if k==0:
                    fea = next_x
                else:
                    #pass
                    #print(len(self.decode_results))
                    #print(self.depth)
                    #print(self.depth-i)
                    #print('--------')
                    #for j in range(len(self.decode_results)):
                    #    print(self.decode_results[j][:,:,2:-2,2:-2].shape)
                    #print('--------')
                    #next_x[:,:int(next_x.shape[1]*self.recurrent_ch_ratio),:,:] = next_x[:,:int(next_x.shape[1]*self.recurrent_ch_ratio),:,:]+self.decode_results[self.depth-i-1][:,:int(next_x.shape[1]*self.recurrent_ch_ratio),2:-2,2:-2].detach()
                    fea = torch.cat((endecode_results_last[i],next_x[:,int(next_x.shape[1]*self.recurrent_ch_ratio):,:,:]), 1)
                    #if i<2:
                    #    fea = torch.cat((next_x[:,:int(next_x.shape[1]*self.recurrent_ch_ratio),:,:]+self.decode_results[self.depth-i-1][:,:,2:-2,2:-2].detach(),next_x[:,int(next_x.shape[1]*self.recurrent_ch_ratio):,:,:]), 1)
                    #else:
                    #    fea = next_x
                next_x = down_layer(fea)
                x_results.append(next_x)
                endecode_results_curr.append(next_x[:,:int(next_x.shape[1]*self.recurrent_ch_ratio),:,:])
                

            #print('..........')
            #for j in range(len(x_results)):
            #    print(x_results[j].shape)
            #print('..........')

            up_x = x_results[(self.depth)]
            #print('--------')
            #print(up_x.shape)
            #print('--------')
            up_x = self.gcn(up_x)
            endecode_results_curr.append(up_x[:,:int(up_x.shape[1]*self.recurrent_ch_ratio),:,:])
            
            for i, up_layer in enumerate(self.up_path):
                if k==0:
                    fea = up_x
                else:
                    fea = torch.cat((endecode_results_last[len(self.down_path)+1+i],up_x[:,int(up_x.shape[1]*self.recurrent_ch_ratio):,:,:]), 1)
                up_x = up_layer(fea, x_results[(self.depth - (i + 1))], self.con_operator, self.network, d_weight_mul)
                endecode_results_curr.append(up_x[:,:int(up_x.shape[1]*self.recurrent_ch_ratio),:,:])

            fea1 = F.adaptive_avg_pool2d(up_x, (1,1))
            fea2 = self.contrast_extracter(up_x)
            fea2 = F.adaptive_avg_pool2d(fea2, (1,1))
            fea_final = torch.cat([fea1, fea2], dim=1)
            features.append(fea_final.unsqueeze(1))
            
            x_out = self.outc(up_x)
            if self.last_sig is not None:
                x_out = self.last_sig(x_out)
            if apply_crop and self.to_crop:
                x_out = data_loader_util.crop_input_hdr_batch(x_out, diffY=diffY, diffX=diffX)
            output_results.append(x_out.unsqueeze(1))
            endecode_results_last = endecode_results_curr.copy()
        output_results = torch.cat(output_results, 1)
        features = torch.cat(features, 1)
        return output_results, features
