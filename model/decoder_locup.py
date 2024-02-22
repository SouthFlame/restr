
import math
import torch
from torch import nn, einsum
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pdb
from timm.models.layers import trunc_normal_
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.autograd import Variable

from utils.torchutils import generate_spatial_batch
from model.transformers.blocks import Block, convBlock, upconvBlock, CrossBlock
from model.transformers.utils import init_weights, positionalencoding1d, positionalencoding2d


# (a) Vanilla self-attention encoder
class Decoder_vlst_locup(nn.Module):
    def __init__(
        self,
        # n_l,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
        num_shared,
        n_seed = 1,
        is_bilstm = False,
        lang_pad_mask = False,
        n_up = 4,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_l = 20
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5
        self.lang_pad_mask = lang_pad_mask

        self.proj_v = nn.Linear(d_encoder, d_model)
        self.proj_l = nn.Linear(2*d_model, d_model) if is_bilstm is True else nn.Linear(d_model, d_model)

        self.seed_emb = nn.Parameter(torch.randn(1, n_seed, d_model))

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_cls = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )
        self.decoder_norm = nn.LayerNorm(d_model)
        self.num_shared = num_shared

        # conv decoder
        self.upconvs = nn.ModuleList()
        for i in range(n_up):
            self.upconvs.append(
                upconvBlock(2*d_model//2**(i), d_model//2**(i)),
            )

        self.last_convs = nn.Conv2d(d_model//2**(n_up-1), 1, kernel_size=1, bias=False)
        
        self.apply(init_weights)
        trunc_normal_(self.seed_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x_v, x_l, is_distilled, im_size, lang=None):
        H, W = im_size
        GS = H // self.patch_size

        # Remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + is_distilled
        x_v = x_v[:, num_extra_tokens:]

        # 1. Normalization like previous Referring Image Segmentation to fuse the features from different domain
        x_v = F.normalize(x_v, dim=2) 
        x_l = F.normalize(x_l, dim=2) 

        # 2. Projection x_v and x_l
        x_v = self.proj_v(x_v)
        x_l = self.proj_l(x_l)

        if self.lang_pad_mask is True:
            x_l[lang.unsqueeze(2).expand(-1, -1, x_l.size(2))==0] = 0
            
        seed_emb = self.seed_emb.expand(x_v.size(0), -1, -1)

        # 4. Feed-foward to Transfomer encoder block
        x = torch.cat((x_v, x_l, seed_emb), 1)
        for _ in range(self.num_shared):
            for blk in self.blocks:
                x = blk(x)
        x = self.decoder_norm(x)

        # 5. Dice for each modality and Projection
        patches, out_l, adapt_cls = x[:, : -self.n_l-1], x[:, -self.n_l-1 :-1], x[:, -1:]
        patches = patches @ self.proj_patch 
        adapt_cls = adapt_cls @ self.proj_cls 

        patch_pred = einsum('b p d, b w d -> b p w', patches, adapt_cls)

        masked_patches = patch_pred * patches
        
        masks = torch.cat((x_v, masked_patches), 2)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))
        for upconv in self.upconvs:
            masks = upconv(masks)
        masks = self.last_convs(masks)

        patch_pred = rearrange(patch_pred, "b (h w) n -> b n h w", h=int(GS))

        return masks, None, patch_pred

# (b) independent between vision and memory path
class Decoder_independvltlst_locup(nn.Module):
    def __init__(
        self,
        # n_l,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
        num_shared,
        n_seed = 1,
        is_bilstm = False,
        lang_pad_mask = False,
        n_up = 4,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_l = 20
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5
        self.lang_pad_mask = lang_pad_mask

        self.proj_v = nn.Linear(d_encoder, d_model)
        self.proj_l = nn.Linear(2*d_model, d_model) if is_bilstm is True else nn.Linear(d_model, d_model)

        self.seed_emb = nn.Parameter(torch.randn(1, n_seed, d_model))

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_cls = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocksvl = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )
        self.blockslm = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )
        self.decoder_norm = nn.LayerNorm(d_model)
        self.num_shared = num_shared

        # conv decoder
        self.upconvs = nn.ModuleList()
        for i in range(n_up):
            self.upconvs.append(
                upconvBlock(2*d_model//2**(i), d_model//2**(i)),
            )

        self.last_convs = nn.Conv2d(d_model//2**(n_up-1), 1, kernel_size=1, bias=False)
        
        self.apply(init_weights)
        trunc_normal_(self.seed_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x_v, x_l, is_distilled, im_size, lang=None):
        H, W = im_size
        GS = H // self.patch_size

        # Remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + is_distilled
        x_v = x_v[:, num_extra_tokens:]

        # 1. Normalization like previous Referring Image Segmentation to fuse the features from different domain
        x_v = F.normalize(x_v, dim=2) 
        x_l = F.normalize(x_l, dim=2) 

        # 2. Projection x_v and x_l
        x_v = self.proj_v(x_v) 
        x_l = self.proj_l(x_l) 

        if self.lang_pad_mask is True:
            x_l[lang.unsqueeze(2).expand(-1, -1, x_l.size(2))==0] = 0
            
        seed_emb = self.seed_emb.expand(x_v.size(0), -1, -1)

        # 4. Feed-foward to Transfomer encoder block
        x = torch.cat((x_v, x_l), 1)
        for _ in range(self.num_shared):
            for blk in self.blocksvl:
                x = blk(x)
        x = self.decoder_norm(x)
        patches, x_l_v = x[:, :-self.n_l], x[:, -self.n_l:]

        x = torch.cat((x_l, seed_emb), 1)
        for _ in range(self.num_shared):
            for blk in self.blockslm:
                x = blk(x)
        x = self.decoder_norm(x)

        # 5. Dice for each modality and Projection
        out_l, adapt_cls = x[:, :-1], x[:, -1:]

        patches = patches @ self.proj_patch 
        adapt_cls = adapt_cls @ self.proj_cls 

        patch_pred = einsum('b p d, b w d -> b p w', patches, adapt_cls)

        masked_patches = patch_pred * patches
        
        masks = torch.cat((x_v, masked_patches), 2)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))
        for upconv in self.upconvs:
            masks = upconv(masks)
        masks = self.last_convs(masks)

        patch_pred = rearrange(patch_pred, "b (h w) n -> b n h w", h=int(GS))

        return masks, None, patch_pred

# (c) indirectly conjugating between vision and memory path
class Decoder_vltlst_locup(nn.Module):
    def __init__(
        self,
        # n_l,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
        num_shared,
        n_seed = 1,
        is_bilstm = False,
        lang_pad_mask = False,
        n_up = 4,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_l = 20
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5
        self.lang_pad_mask = lang_pad_mask

        self.proj_v = nn.Linear(d_encoder, d_model)
        self.proj_l = nn.Linear(2*d_model, d_model) if is_bilstm is True else nn.Linear(d_model, d_model)

        self.seed_emb = nn.Parameter(torch.randn(1, n_seed, d_model))

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_cls = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocksvl = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )
        self.blockslm = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )
        self.decoder_norm = nn.LayerNorm(d_model)
        self.num_shared = num_shared

        # conv decoder
        self.upconvs = nn.ModuleList()
        for i in range(n_up):
            self.upconvs.append(
                upconvBlock(2*d_model//2**(i), d_model//2**(i)),
            )

        self.last_convs = nn.Conv2d(d_model//2**(n_up-1), 1, kernel_size=1, bias=False)
        
        self.apply(init_weights)
        trunc_normal_(self.seed_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x_v, x_l, is_distilled, im_size, lang=None):
        H, W = im_size
        GS = H // self.patch_size

        # Remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + is_distilled
        x_v = x_v[:, num_extra_tokens:]

        # 1. Normalization like previous Referring Image Segmentation to fuse the features from different domain
        x_v = F.normalize(x_v, dim=2)
        x_l = F.normalize(x_l, dim=2)
        
        # 2. Projection x_v and x_l
        x_v = self.proj_v(x_v) 
        x_l = self.proj_l(x_l) 

        if self.lang_pad_mask is True:
            x_l[lang.unsqueeze(2).expand(-1, -1, x_l.size(2))==0] = 0
            
        seed_emb = self.seed_emb.expand(x_v.size(0), -1, -1)

        # 4. Feed-foward to Transfomer encoder block
        x = torch.cat((x_v, x_l), 1)
        for _ in range(self.num_shared):
            for blk in self.blocksvl:
                x = blk(x)
        patches, x_l = x[:, :-self.n_l], x[:, -self.n_l:]

        x = torch.cat((x_l, seed_emb), 1)
        for _ in range(self.num_shared):
            for blk in self.blockslm:
                x = blk(x)
        x = self.decoder_norm(x)

        # 5. Dice for each modality and Projection
        out_l, adapt_cls = x[:, :-1], x[:, -1:]

        patches = patches @ self.proj_patch 
        adapt_cls = adapt_cls @ self.proj_cls

        patch_pred = einsum('b p d, b w d -> b p w', patches, adapt_cls)

        masked_patches = patch_pred * patches
        
        masks = torch.cat((x_v, masked_patches), 2)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))
        for upconv in self.upconvs:
            masks = upconv(masks)
        masks = self.last_convs(masks)

        patch_pred = rearrange(patch_pred, "b (h w) n -> b n h w", h=int(GS))

        return masks, None, patch_pred


class MM_fusion(nn.Module):
    def __init__(
        self,
        patch_size,
        d_v_encoder,
        d_l_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
        is_shared = False,
        n_seed = 1,
        lang_pad_mask = False,
        n_up = 4,

    ):
        super().__init__()
        self.d_v_encoder = d_v_encoder
        self.d_l_encoder = d_l_encoder
        self.patch_size = patch_size
        self.n_l = 20
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5
        self.lang_pad_mask = lang_pad_mask

        self.proj_v = nn.Linear(d_v_encoder, d_model)
        self.proj_l = nn.Linear(d_l_encoder, d_model)

        self.seed_emb = nn.Parameter(torch.randn(1, n_seed, d_model))

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_cls = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocksvl = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )
        self.blockslm = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )
        self.decoder_norm = nn.LayerNorm(d_model)
        self.is_shared = is_shared

        # conv decoder
        self.upconvs = nn.ModuleList()
        for i in range(n_up):
            self.upconvs.append(
                upconvBlock(2*d_model//2**(i), d_model//2**(i)),
            )

        self.last_convs = nn.Conv2d(d_model//2**(n_up-1), 1, kernel_size=1, bias=False)
        
        self.apply(init_weights)
        trunc_normal_(self.seed_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x_v, x_l, is_distilled, im_size, lang=None):
        H, W = im_size
        GS = H // self.patch_size

        # Remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + is_distilled
        x_v = x_v[:, num_extra_tokens:]

        # 1. Normalization like previous Referring Image Segmentation to fuse the features from different domain
        x_v = F.normalize(x_v, dim=2)
        x_l = F.normalize(x_l, dim=2)
        
        # 2. Projection x_v and x_l
        x_v = self.proj_v(x_v) 
        x_l = self.proj_l(x_l) 

        if self.lang_pad_mask is True:
            x_l[lang.unsqueeze(2).expand(-1, -1, x_l.size(2))==0] = 0
            
        seed_emb = self.seed_emb.expand(x_v.size(0), -1, -1)

        # 4. Feed-foward to Transfomer encoder block
        x = torch.cat((x_v, x_l), 1)
        for _ in range(self.num_shared):
            for blk in self.blocksvl:
                x = blk(x)
        patches, x_l = x[:, :-self.n_l], x[:, -self.n_l:]

        x = torch.cat((x_l, seed_emb), 1)
        for _ in range(self.num_shared):
            for blk in self.blockslm:
                x = blk(x)
        x = self.decoder_norm(x)

        # 5. Slice to extract the output of seed emb as adaptive classifier 
        out_l, adapt_cls = x[:, :-1], x[:, -1:]

        patches = patches @ self.proj_patch 
        adapt_cls = adapt_cls @ self.proj_cls

        patch_pred = einsum('b p d, b w d -> b p w', patches, adapt_cls)

        masked_patches = patch_pred * patches
        
        patches = torch.cat((x_v, masked_patches), 2)
        patches = rearrange(patches, "b (h w) n -> b n h w", h=int(GS))
        for upconv in self.upconvs:
            masks = upconv(patches)
        pixel_pred = self.last_convs(patches)

        patch_pred = rearrange(patch_pred, "b (h w) n -> b n h w", h=int(GS))

        return pixel_pred, None, patch_pred