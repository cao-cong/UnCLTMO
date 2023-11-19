import os
import cv2
import numpy as np
import torch
from torchvision.datasets import DatasetFolder
import torch.utils.data as data
from utils import data_loader_util, hdr_image_util, params
import torch.nn.functional as F
import glob
import random

IMG_EXTENSIONS_local = ('.npy')


def get_ldr_im(normalization, input_im, max_stretch, min_stretch):
    if normalization == "max_normalization":
        input_im = input_im / input_im.max()
    elif normalization == "bugy_max_normalization":
        input_im = input_im / 255
    elif normalization == "stretch":
        input_im = ((input_im - input_im.min()) / input_im.max()) * max_stretch - min_stretch
        input_im = np.clip(input_im, 0, 1)
    return input_im


def get_f(f_dict_path, im_name, factor_coeff):
    # use_hist_fit true by default
    data = np.load(f_dict_path, allow_pickle=True)
    if im_name in data[()]:
        f_factor = data[()][im_name]
        brightness_factor = f_factor * 255 * factor_coeff
    else:
        # TODO: add the option to calculate f here
        raise Exception("no lambda found for file %s in %s" % (im_name, f_dict_path))
    return brightness_factor

def preprocess(raw):
    input_full = raw.transpose((0, 3, 1, 2))
    input_full = torch.from_numpy(input_full)
    input_full = input_full.cuda()
    return input_full

def npy_loader(path, addFrame, hdrMode, ldrNegMode, normalization, min_stretch,
               max_stretch, factor_coeff, use_contrast_ratio_f, use_hist_fit, f_dict_path,
               final_shape_addition, real_video):
    """
    load npy files that contain the loaded HDR file, and binary image of windows centers.

    """
    #print('process')
    #print(real_video)
    if ldrNegMode:
        input_im_frames = []
        color_im_frames = []
        gray_original_im_norm_frames = []
        gray_original_im_frames = []
        for k in range(2):
            data = np.load(path, allow_pickle=True)
            color_im = data
            color_im = color_im.astype(np.float32)

            mode = np.random.randint(0,2)
            if mode==0:
                resize_h = 256
            else:
                resize_h = int(np.random.uniform(256,512))
            resize_w = resize_h
            color_im = cv2.resize(color_im, (resize_w, resize_h))
            
            patch_h = 256
            patch_w = patch_h
            h = color_im.shape[0]
            w = color_im.shape[1]
            if h==patch_h:
                pass
            else:
                xx = np.random.randint(0, w - patch_w) 
                yy = np.random.randint(0, h - patch_h)
                color_im = color_im[yy:yy+patch_h,xx:xx+patch_w,:]
            
            yuv_im = cv2.cvtColor(color_im, cv2.COLOR_RGB2YUV)
            input_im = yuv_im[:,:,:1]
            color_im = preprocess(np.expand_dims(color_im, 0))[0]
            input_im = preprocess(np.expand_dims(input_im, 0))[0]

            input_im = get_ldr_im(normalization, input_im, max_stretch, min_stretch)
            input_im_frames.append(input_im.unsqueeze(0))
            color_im_frames.append(color_im.unsqueeze(0))

        input_im_frames = torch.cat(input_im_frames, 0)
        color_im_frames = torch.cat(color_im_frames, 0)
        return input_im_frames, color_im_frames, input_im_frames, input_im_frames, 0
    else:
        if real_video:
            #print(1)
            input_im_frames = []
            color_im_frames = []
            gray_original_im_norm_frames = []
            gray_original_im_frames = []
            for k in range(2):
                
                basename = os.path.basename(path)
                if basename[0]!='0':
                    frame_id = int(basename[0:3])
                elif basename[1]!='0':
                    frame_id = int(basename[1:3])
                else:
                    frame_id = int(basename[2:3])
                #print(path)
                data = np.load(path, allow_pickle=True)
                path_next = path.replace(basename, '%03d.npy'%(frame_id+1))
                if os.path.exists(path_next):
                    path = path_next
                else:
                    path = path
                #input_im = data[()]["input_image"]
                #color_im = data[()]["display_image"]
                color_im = data
                #input_im = input_im.astype(np.float32)
                color_im = color_im.astype(np.float32)
                patch_w = 256
                w = color_im.shape[1]
                xx = np.random.randint(0, w - patch_w) 
                color_im = color_im[:,xx:xx+patch_w,:]

                yuv_im = cv2.cvtColor(color_im, cv2.COLOR_RGB2YUV)
                input_im = yuv_im[:,:,:1]
                color_im = preprocess(np.expand_dims(color_im, 0))[0]
                input_im = preprocess(np.expand_dims(input_im, 0))[0]
                #input_im_max, color_im_max = input_im.max(), color_im.max()
                #input_im = F.interpolate(input_im.unsqueeze(dim=0), size=(params.input_size, params.input_size), mode='bicubic',
                #                         align_corners=False).squeeze(dim=0).clamp(min=0, max=input_im_max)
                #color_im = F.interpolate(color_im.unsqueeze(dim=0), size=(params.input_size, params.input_size), mode='bicubic',
                #                         align_corners=False).squeeze(dim=0).clamp(min=0, max=color_im_max)
                if not hdrMode:
                    input_im = get_ldr_im(normalization, input_im, max_stretch, min_stretch)
                    input_im_frames.append(input_im.unsqueeze(0))
                    color_im_frames.append(color_im.unsqueeze(0))
                if hdrMode:
                    gray_original_im = hdr_image_util.to_gray_tensor(color_im)
                    gray_original_im_norm = gray_original_im / gray_original_im.max()
                    # TODO(): fix this part
                    #im_name = os.path.splitext(os.path.basename(path))[0]
                    scene_name = path.split('/')[-2]
                    #factor_coeff = factor_coeff * 0.1 * np.random.uniform(0.1,2)
                    brightness_factor = get_f(f_dict_path, scene_name, factor_coeff)
                    gray_original_im = gray_original_im - gray_original_im.min()
                    a = torch.log10((gray_original_im / gray_original_im.max()) * brightness_factor + 1)
                    input_im = a / a.max()
                    if addFrame:
                        input_im = data_loader_util.add_frame_to_im(input_im, final_shape_addition, final_shape_addition)
                    #return input_im, color_im, gray_original_im_norm, gray_original_im, data[()]["gamma_factor"]
                    input_im_frames.append(input_im.unsqueeze(0))
                    color_im_frames.append(color_im.unsqueeze(0))
                    gray_original_im_norm_frames.append(gray_original_im_norm.unsqueeze(0))
                    gray_original_im_frames.append(gray_original_im.unsqueeze(0))
            input_im_frames = torch.cat(input_im_frames, 0)
            color_im_frames = torch.cat(color_im_frames, 0)
            if not hdrMode:
                return input_im_frames, color_im_frames, input_im_frames, input_im_frames, 0
            if hdrMode:
                gray_original_im_norm_frames = torch.cat(gray_original_im_norm_frames, 0)
                gray_original_im_frames = torch.cat(gray_original_im_frames, 0)
                return input_im_frames.float(), color_im_frames.float(), gray_original_im_norm_frames.float(), gray_original_im_frames.float(), brightness_factor
        else:
            #print(2)
            input_im_frames = []
            color_im_frames = []
            gray_original_im_norm_frames = []
            gray_original_im_frames = []
            for k in range(2):
                data = np.load(path, allow_pickle=True)
                #input_im = data[()]["input_image"]
                #color_im = data[()]["display_image"]
                color_im = data
                #input_im = input_im.astype(np.float32)
                color_im = color_im.astype(np.float32)
                if color_im.shape[0]==256:
                    pass
                else:
                    mode = np.random.randint(0,2)
                    if mode==0:
                        resize_h = 256
                    else:
                        resize_h = int(np.random.uniform(256,512))
                    resize_w = resize_h
                    color_im = cv2.resize(color_im, (resize_w, resize_h))
                patch_h = 256
                patch_w = patch_h
                h = color_im.shape[0]
                w = color_im.shape[1]
                if h==patch_h:
                    pass
                else:
                    xx = np.random.randint(0, w - patch_w) 
                    yy = np.random.randint(0, h - patch_h)
                    color_im = color_im[yy:yy+patch_h,xx:xx+patch_w,:]
                
                yuv_im = cv2.cvtColor(color_im, cv2.COLOR_RGB2YUV)
                input_im = yuv_im[:,:,:1]
                color_im = preprocess(np.expand_dims(color_im, 0))[0]
                input_im = preprocess(np.expand_dims(input_im, 0))[0]
                #input_im_max, color_im_max = input_im.max(), color_im.max()
                #input_im = F.interpolate(input_im.unsqueeze(dim=0), size=(params.input_size, params.input_size), mode='bicubic',
                #                         align_corners=False).squeeze(dim=0).clamp(min=0, max=input_im_max)
                #color_im = F.interpolate(color_im.unsqueeze(dim=0), size=(params.input_size, params.input_size), mode='bicubic',
                #                         align_corners=False).squeeze(dim=0).clamp(min=0, max=color_im_max)
                if not hdrMode:
                    input_im = get_ldr_im(normalization, input_im, max_stretch, min_stretch)
                    input_im_frames.append(input_im.unsqueeze(0))
                    color_im_frames.append(color_im.unsqueeze(0))
                if hdrMode:
                    gray_original_im = hdr_image_util.to_gray_tensor(color_im)
                    gray_original_im_norm = gray_original_im / gray_original_im.max()
                    # TODO(): fix this part
                    im_name = os.path.splitext(os.path.basename(path))[0]
                    #factor_coeff = factor_coeff * 0.1 * np.random.uniform(0.1,2)
                    brightness_factor = get_f(f_dict_path, im_name, factor_coeff)
                    gray_original_im = gray_original_im - gray_original_im.min()
                    a = torch.log10((gray_original_im / gray_original_im.max()) * brightness_factor + 1)
                    input_im = a / a.max()
                    if addFrame:
                        input_im = data_loader_util.add_frame_to_im(input_im, final_shape_addition, final_shape_addition)
                    #return input_im, color_im, gray_original_im_norm, gray_original_im, data[()]["gamma_factor"]
                    input_im_frames.append(input_im.unsqueeze(0))
                    color_im_frames.append(color_im.unsqueeze(0))
                    gray_original_im_norm_frames.append(gray_original_im_norm.unsqueeze(0))
                    gray_original_im_frames.append(gray_original_im.unsqueeze(0))
            input_im_frames = torch.cat(input_im_frames, 0)
            color_im_frames = torch.cat(color_im_frames, 0)
            if not hdrMode:
                return input_im_frames, color_im_frames, input_im_frames, input_im_frames, 0
            if hdrMode:
                gray_original_im_norm_frames = torch.cat(gray_original_im_norm_frames, 0)
                gray_original_im_frames = torch.cat(gray_original_im_frames, 0)
                return input_im_frames.float(), color_im_frames.float(), gray_original_im_norm_frames.float(), gray_original_im_frames.float(), brightness_factor


#class ProcessedDatasetFolder(DatasetFolder):
class ProcessedDatasetFolder(data.Dataset):
    """
    A customized data loader, to load .npy file that contains a dict
    of numpy arrays that represents hdr_images and window_binary_images.
    """

    def __init__(self, root, dataset_properties,
                 hdrMode, ldrNegMode, loader=npy_loader):

        self.imgs = root
        self.addFrame = dataset_properties["add_frame"]
        self.hdrMode = hdrMode
        self.ldrNegMode = ldrNegMode
        self.loader = loader
        self.normalization = dataset_properties["normalization"]
        self.max_stretch = dataset_properties["max_stretch"]
        self.min_stretch = dataset_properties["min_stretch"]
        self.factor_coeff = dataset_properties["factor_coeff"]
        self.use_contrast_ratio_f = dataset_properties["use_contrast_ratio_f"]
        self.use_hist_fit = dataset_properties["use_hist_fit"]
        self.f_train_dict_path = dataset_properties["f_train_dict_path"]
        self.final_shape_addition = dataset_properties["final_shape_addition"]
        self.hdr_video_path = glob.glob('../../data/tone_mapping/train_HDRvideo/*/*.npy')
        self.srgb_video_path = glob.glob('../../data/tone_mapping/train_sRGBvideo/*/*.npy')
        self.f_train_hdrvideo_dict_path = "data/input_images_lambdas_trainHDRvideo.npy"
        self.negative_ldr_path = glob.glob('../../data/tone_mapping/SICE_patches512_npy/*.npy')
        for _ in range(3):
            self.negative_ldr_path = self.negative_ldr_path + self.negative_ldr_path
        #print('negative ldr paths:{}'.format(self.negative_ldr_path))
        #print('.............')
        #print(len(self.imgs))
        #print(len(self.hdr_video_path))
        #print(len(self.srgb_video_path))
        #print('.............')
        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            sample: {'hdr_image': im, 'binary_wind_image': binary_im}
        """
        #path, target = self.samples[index]
        if self.ldrNegMode:
            path = self.negative_ldr_path[index]
            f_train_dict_path = self.f_train_dict_path
            real_video = False
        else:
            choice = np.random.uniform(0,1)
            if choice<0.5:
                path = self.imgs[index]
                f_train_dict_path = self.f_train_dict_path
                real_video = False
            else:
                if self.hdrMode:
                    path = self.hdr_video_path[index]
                else:
                    path = self.srgb_video_path[index]
                f_train_dict_path = self.f_train_hdrvideo_dict_path
                real_video = True
        #print(real_video)
        input_im, color_im, gray_original_norm, gray_original, gamma_factor = self.loader(path, self.addFrame,
                                                                                          self.hdrMode,
                                                                                          self.ldrNegMode,
                                                                                          self.normalization,
                                                                                          self.min_stretch,
                                                                                          self.max_stretch,
                                                                                          self.factor_coeff,
                                                                                          self.use_contrast_ratio_f,
                                                                                          self.use_hist_fit,
                                                                                          f_train_dict_path,
                                                                                          self.final_shape_addition,
                                                                                          real_video=real_video)
        #print('......')
        #print(input_im.shape)
        #print(color_im.shape)
        #print(gray_original_norm.shape)
        #print(gray_original.shape)
        #print('......')
        return {"input_im": input_im, "color_im": color_im, "original_gray_norm": gray_original_norm,
                "original_gray": gray_original, "gamma_factor": gamma_factor}
                
    def __len__(self):
        if self.ldrNegMode:
            return len(self.negative_ldr_path)
        else:
            return len(self.imgs)
