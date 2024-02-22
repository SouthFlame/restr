import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformers.utils import padding, unpadding
from timm.models.layers import trunc_normal_

from einops import rearrange


class ReSTR(nn.Module):
    def __init__(
        self,
        encoder_v,
        encoder_l,
        fusion_module,
        n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder_v.patch_size
        self.encoder_v = encoder_v
        self.encoder_l = encoder_l
        self.fusion_module = fusion_module

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder_v.", self.encoder_v).union(
            append_prefix_no_weight_decay("fusion_module.", self.fusion_module)
        )
        return nwd_params

    def forward(self, im, lang):
        H, W = im.size(2), im.size(3)

        x_v = self.encoder_v(im, return_features=True)
        if len(x_v.shape) > 3:
            x_v = rearrange(x_v, "b n h w -> b (h w) n")
        x_l = self.encoder_l(lang)

        pred = self.fusion_module(x_v, x_l, self.encoder_v.distilled, (H, W), lang)
        return pred

    def get_1x_lr_params(self):
        b = []
        b.append(self.fusion_module.parameters())
        b.append(self.encoder_l.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def get_01x_lr_params(self):
        b = []
        b.append(self.encoder_v.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params(), 'lr': args.lr},
                {'params': self.get_01x_lr_params(), 'lr': 0.1*args.lr}] 