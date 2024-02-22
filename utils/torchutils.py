import pdb
import sys
import os
from einops.einops import rearrange
import torch
import torch.nn as nn
import torch.distributed as dist
import cv2
from PIL import Image



import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as TF
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

import math
import random

# from albumentations import (
#     HorizontalFlip,
#     RandomSizedCrop,
#     RandomBrightnessContrast,   
#     Compose, 
#     Blur,
#     RandomGamma,
# )

# def seg_aug(image, mask):
#     aug = Compose([
#               HorizontalFlip(p=0.5),
#               RandomBrightnessContrast(p=0.3),
#               ])

#     augmented = aug(image=image, mask=mask)
#     return augmented


# def crop_aug(image, mask, h, w, min_max_height, w2h_ratio=2):
#     aug = Compose([
#               HorizontalFlip(p=0.5),
#               RandomBrightnessContrast(p=0.3),              
#               RandomSizedCrop(height=h, width=w, min_max_height=min_max_height, w2h_ratio=2),
#               Blur(blur_limit=3),
#               RandomGamma(p=0.3)
#               ])

#     augmented = aug(image=image, mask=mask)
#     return augmented


# def crop_aug_web_img(image, h, w, min_max_height, w2h_ratio=2):
#     aug = Compose([
#               HorizontalFlip(p=0.5),
#             #   RandomBrightnessContrast(p=0.3),              
#               RandomSizedCrop(height=h, width=w, min_max_height=min_max_height, w2h_ratio=2),
#               ])

#     augmented = aug(image=image)
#     return augmented


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def vis_image(image, cmap=None):
    dpi = 80.0
    xpixels, ypixels = 720, 1280
    fig = plt.figure(figsize=(ypixels/dpi, xpixels/dpi), dpi=dpi)
    if cmap is not None:
        plt.imshow(image.cpu().data[0].numpy().astype(int), cmap=cmap)
    else:
        plt.imshow(image.astype(int))
    plt.show()

def decode_image(image, mean=IMG_MEAN):
    output = image.detach().cpu().data.numpy()
    output = np.transpose(output, (1, 2, 0)) + mean
    output = output.clip(0,255)
    output = output[:, :, ::-1]
    return output

def decode_image_vit(image, normalization):
    normalization = normalization.copy()
    for k, v in normalization.items():
        v = np.round(255 * np.array(v), 2)
        normalization[k] = tuple(v)
    # pdb.set_trace()
    output = image.detach().cpu().data.numpy()
    output = np.transpose(output, (1, 2, 0)) * normalization["std"] + normalization["mean"]
    output = output.clip(0,255)
    # output = output[:, :, ::-1]
    return output.astype(np.uint8)

def vis_torch_mask(masks, output_dir, cmap=False):
    save_mask = np.asarray(masks.cpu().data.numpy()).squeeze()
    if cmap is True:
        output_mask = cv2.applyColorMap(np.uint8(save_mask * 255), cv2.COLORMAP_JET)[:, :, ::-1]
        output_mask = Image.fromarray(output_mask)
    else:
        output_mask = Image.fromarray(np.uint8(save_mask * 255), 'L')
    output_mask.save(output_dir)
    
    return save_mask, output_mask


class TensorRandomResizeLong():
    def __init__(self, min_long=None, max_long=None):
        self.min_long = min_long
        self.max_long = max_long
    def __call__(self, img_tensor):
        target_long = random.randint(self.min_long, self.max_long)
        w, h = img_tensor.shape[3], img_tensor.shape[2]
        if w < h:
            target_shape = (int(round(w * target_long / h)), target_long)
        else:
            target_shape = (target_long, int(round(h * target_long / w)))
        
        interp = nn.Upsample(size=(target_shape[1], target_shape[0]), mode='bicubic', align_corners=True)

        img_tensor = interp(img_tensor)
        # img = img.resize(target_shape, resample=Image.HAMMING)

        return img_tensor

def gap2d(x, keepdims=False):
    out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)
    return out

def var2d(x, keepdims=False):
    out = torch.var(x.view(x.size(0), x.size(1), -1), -1)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)
    return out

# DDP setting
def setup(rank, world_size, port):
    if sys.platform == 'win32':
        # Distributed package only covers collective communications with Gloo
        # backend and FileStore on Windows platform. Set init_method parameter
        # in init_process_group to a local file.
        # Example init_method="file:///f:/libtmp/some_file"
        init_method="file:///{your local file path}"

        # initialize the process group
        dist.init_process_group(
            "gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size
        )
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = port

        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

    
def generate_spatial_batch(featmap_H, featmap_W):
    """Generate additional visual coordinates feature maps.
    Function taken from
    https://github.com/chenxi116/TF-phrasecut-public/blob/master/util/processing_tools.py#L5
    and slightly modified
    """
    spatial_batch_val = torch.zeros((1, 8, featmap_H, featmap_W))
    for h in range(featmap_H):
        for w in range(featmap_W):
            xmin = w / featmap_W * 2 - 1
            xmax = (w + 1) / featmap_W * 2 - 1
            xctr = (xmin + xmax) / 2
            ymin = h / featmap_H * 2 - 1
            ymax = (h + 1) / featmap_H * 2 - 1
            yctr = (ymin + ymax) / 2
            spatial_batch_val[0, :, h, w] = torch.tensor([xmin, ymin, xmax, ymax,
                                                        xctr, yctr, 1 / featmap_W, 1 / featmap_H])

    return spatial_batch_val
    # return Variable(tensor).cuda()


def resize_and_pad(images, input_h, input_w):
    # Resize and pad images to input_h x input_w size
    B, C, H, W = images.size()
    scale = min(input_h / H, input_w / W)
    resized_h = int(round(H * scale))
    resized_w = int(round(W * scale))
    pad_h = int(math.floor(input_h - resized_h) / 2)
    pad_w = int(math.floor(input_w - resized_w) / 2)

    resize_image = TF.resize(images, [resized_h, resized_w])
    new_im = torch.zeros((B, C, input_h, input_w)).float() if images.is_cuda is False \
            else torch.zeros((B, C, input_h, input_w)).cuda().float()
    new_im[..., pad_h:pad_h+resized_h, pad_w:pad_w+resized_w] = resize_image

    return new_im

def resize_and_crop(images, input_h, input_w):
    # Resize and crop images to input_h x input_w size
    B, C, H, W = images.size()
    scale = max(input_h / H, input_w / W)
    resized_h = int(round(H * scale))
    resized_w = int(round(W * scale))
    crop_h = int(math.floor(resized_h - input_h) / 2)
    crop_w = int(math.floor(resized_w - input_w) / 2)
    resize_image = TF.resize(images, [resized_h, resized_w])
    new_im = torch.zeros((B, C, input_h, input_w)).float() if images.is_cuda is False \
            else torch.zeros((B, C, input_h, input_w)).cuda().float()
    new_im[...] = resize_image[..., crop_h:crop_h+input_h, crop_w:crop_w+input_w]
    
    return new_im

def resize_and_crop_nearest(images, input_h, input_w):
    # Resize and crop images to input_h x input_w size
    B, C, H, W = images.size()
    scale = max(input_h / H, input_w / W)
    resized_h = int(round(H * scale))
    resized_w = int(round(W * scale))
    crop_h = int(math.floor(resized_h - input_h) / 2)
    crop_w = int(math.floor(resized_w - input_w) / 2)
    resize_image = TF.resize(images, [resized_h, resized_w], interpolation=TF.InterpolationMode.NEAREST)
    new_im = torch.zeros((B, C, input_h, input_w)).float() if images.is_cuda is False \
            else torch.zeros((B, C, input_h, input_w)).cuda().float()
    new_im[...] = resize_image[..., crop_h:crop_h+input_h, crop_w:crop_w+input_w]
    
    return new_im

# def resize_and_crop(im, input_h, input_w):
#     # Resize and crop images to input_h x input_w size
#     H, W = im.shape[:2]
#     scale = max(input_h / H, input_w / W)
#     resized_h = int(np.round(H * scale))
#     resized_w = int(np.round(W * scale))
#     crop_h = int(np.floor(resized_h - input_h) / 2)
#     crop_w = int(np.floor(resized_w - input_w) / 2)

#     resized_images = skimage.transform.resize(im, [resized_h, resized_w])
#     if im.ndimages > 2:
#         new_images = np.zeros((input_h, input_w, im.shape[2]), dtype=resized_im.dtype)
#     else:
#         new_images = np.zeros((input_h, input_w), dtype=resized_im.dtype)
#     new_im[...] = resized_im[crop_h:crop_h+input_h, crop_w:crop_w+input_w, ...]

#     return new_im


def init_He(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)



def compute_mask_IU(masks, target):
    assert(target.shape[-2:] == masks.shape[-2:])
    I = np.sum(np.logical_and(masks, target))
    U = np.sum(np.logical_or(masks, target))
    return I, U

def compute_mask_IU_torch(masks, target):
    assert(target.shape[-2:] == masks.shape[-2:])
    I = torch.sum(torch.logical_and(masks, target))
    U = torch.sum(torch.logical_or(masks, target))
    return I, U

def set_gpu_mode(mode):
    global use_gpu
    global device
    global gpu_id
    global distributed
    global dist_rank
    global world_size
    gpu_id = int(os.environ.get("SLURM_LOCALID", 0))
    dist_rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))

    distributed = world_size > 1
    use_gpu = mode
    device = torch.device(f"cuda:{gpu_id}" if use_gpu else "cpu")
    torch.backends.cudnn.benchmark = True

def generate_1pix_loc_labels(l_pred, labels_loc):
    B = labels_loc.size(0)
    flat_l_pred = rearrange(l_pred, "b n h w -> b (n h w)")
    

    pix_loc_labels = torch.zeros_like(labels_loc)
    ce_labels_loc = (-100 * torch.ones(B)).cuda()

    for i in range(B):
        i_label_loc = (labels_loc[i] == 1).nonzero(as_tuple=False).float()
        center_label = i_label_loc.mean(0, keepdims=True).round().squeeze()
        if i_label_loc.sum() > 0:
            pix_loc_labels[i, int(center_label[0]), int(center_label[1]), int(center_label[2])] = 1
            flat_labels_loc = rearrange(pix_loc_labels[i], "n h w -> (n h w)")
            flat_labels_idx = torch.where(flat_labels_loc == 1)[0]
            ce_labels_loc[i] = flat_labels_idx
    
    return flat_l_pred, ce_labels_loc.long()

# pred_idx = (resized_pred == 1).nonzero(as_tuple=False).float()
# center_pred_x = pred_idx.mean(0, keepdims=True)
# center_pred_x

# pred_idx = (resized_pred == 1).nonzero(as_tuple=False).float()
# center_pred_x = (pred_idx[:,2].mean())
# center_pred_y = pred_idx[:,3].mean()
# center_pred_x, center_pred_y
# center_pred = torch.zeros_like(resized_pred)
# center_pred[:,:,int(center_pred_x), int(center_pred_y)] = 1 

# label_decode = np.transpose(np.array(center_pred[0]), (1, 2, 0))
# plt.imshow(label_decode[:,:,0])
# plt.show()

# def patch_label_gen(labels, patch_size=16, threshold=0.5):
#     threshold = threshold * (patch_size*patch_size)
#     patchgen = nn.Conv2d(1,1, kernel_size=patch_size, stride=patch_size, bias=False)
#     patchgen.weight = torch.nn.Parameter(torch.ones_like(patchgen.weight))
#     patchgen.weight.requires_grad = False
#     patchlabel = patchgen(labels.float())

#     return patchlabel

class Patch_label_gen(nn.Module):
    def __init__(self, patch_size=16, threshold=0.5):
        super(Patch_label_gen, self).__init__()
        self.threshold = threshold
        self.patchgen = nn.Conv2d(1,1, kernel_size=patch_size, stride=patch_size, bias=False)
        self.patchgen.weight = torch.nn.Parameter(torch.ones_like(self.patchgen.weight))
        self.patchgen.weight.requires_grad = False

    def forward(self, x):
        x = self.patchgen(x)
        out = (x > self.threshold).float()

        return out