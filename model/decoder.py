
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
from model.transformers.blocks_eval import Block_eval
from model.transformers.utils import init_weights, positionalencoding1d, positionalencoding2d

class Decoder_TR(nn.Module):
    def __init__(
        self,
        # n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_lang,
        d_ff,
        drop_path_rate,
        dropout,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = 20                     # 수정 필요 나중에
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        # self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_lang = nn.Linear(d_lang, d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)

        self.proj_mask = nn.Linear(d_model, 1)

        # self.mask_norm = nn.LayerNorm(20)

        self.apply(init_weights)
        # trunc_normal_(self.cls_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x_v, x_l, im_size, lang=None):
        H, W = im_size
        GS = H // self.patch_size
        # pdb.set_trace()
        x_l = self.proj_lang(x_l)   # B, N_l, d_model
        # cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x_v, x_l), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls :]
        patches = patches @ self.proj_patch
        # cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        # cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = self.proj_mask(patches)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))

        return masks

class Decoder_TR2(nn.Module):
    def __init__(
        self,
        # n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_lang,
        d_ff,
        drop_path_rate,
        dropout,
        do_posemb_v,
        num_patches,
        no_grad_w,
        is_wp_norm,
        is_mask_norm,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = 20                     # 수정 필요 나중에
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5
        self.do_posemb_v = do_posemb_v
        self.no_grad_w = no_grad_w

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )
        if self.do_posemb_v is True:
            self.pos_emb_v = nn.Parameter(torch.randn(1, num_patches, d_model))
            trunc_normal_(self.pos_emb_v, std=0.02)

        # self.sent_emb = nn.Parameter(torch.randn(1, 1, d_model))

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)

        self.proj_mask = nn.Sequential( 
            nn.Linear(d_model, d_model//2), 
            nn.ReLU(inplace=True), 
            nn.Linear(d_model//2, d_model//4), 
            nn.ReLU(inplace=True), 
            nn.Linear(d_model//4, 1), 
            )

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x_v, x_l, im_size, lang=None):
        H, W = im_size
        GS = H // self.patch_size
        # pdb.set_trace()
        # x_l = self.proj_lang(x_l)   # B, N_l, d_model
        if self.do_posemb_v is True:
            pos_emb_v = self.pos_emb_v
            x_v = x_v + pos_emb_v

        # sent_emb = self.sent_emb.expand(x_v.size(0), -1, -1)

        x = torch.cat((x_v, x_l), 1)

        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls :]
        patches = patches @ self.proj_patch

        patches = patches / patches.norm(dim=-1, keepdim=True)
        # sent_feat = sent_feat / sent_feat.norm(dim=-1, keepdim=True)
        
        masks = self.proj_mask(patches)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))

        return masks

class Decoder_maskTR(nn.Module):
    def __init__(
        self,
        # n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_lang,
        d_ff,
        drop_path_rate,
        dropout,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = 20                     # 수정 필요 나중에
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        # self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        # self.proj_lang = nn.Linear(d_lang, d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)

        # self.proj_mask = nn.Linear(d_model, 1)
        self.proj_mask = nn.Linear(20, 1)
        # self.mask_norm = nn.LayerNorm(1)

        self.apply(init_weights)
        # trunc_normal_(self.cls_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x_v, x_l, is_distilled, im_size, lang=None):
        H, W = im_size
        GS = H // self.patch_size
        # pdb.set_trace()
        # x_l = self.proj_lang(x_l)   # B, N_l, d_model

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + is_distilled
        x_v = x_v[:, num_extra_tokens:]

        # pdb.set_trace()
        # x_l = self.proj_lang(x_l)   # B, N_l, d_model
        # cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x_v, x_l), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls :]
        # pdb.set_trace()
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes
        word_patches = einsum('b p d, b w d -> b p w', patches, cls_seg_feat)

        word_patches = word_patches / word_patches.norm(dim=-1, keepdim=True)
        # cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = self.proj_mask(word_patches)
        # masks = self.mask_norm(masks)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))

        return masks

# embedding token을  visual feature에 바로 사용해 버리는 커널로 사용.
class Decoder_weighTR(nn.Module):
    def __init__(
        self,
        # n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_lang,
        d_ff,
        drop_path_rate,
        dropout,
        do_posemb_v,
        do_posemb_l,
        num_patches,
        no_grad_w,
        is_mask_norm,
        num_shared,
        no_vlnorm,
        is_sinusoidal,
        n_mm = 1,
        is_bilstm=False,
        lang_pad_mask =False,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = 20                     # 수정 필요 나중에
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5
        self.do_posemb_v = do_posemb_v
        self.do_posemb_l = do_posemb_l
        self.no_grad_w = no_grad_w
        self.no_vlnorm = no_vlnorm

        if self.do_posemb_v is True:
            if is_sinusoidal is False:
                self.pos_emb_v = nn.Parameter(torch.randn(1, num_patches, d_model))
                trunc_normal_(self.pos_emb_v, std=0.02)
            else:
                self.pos_emb_v = positionalencoding2d(d_model, int(num_patches**0.5), int(num_patches**0.5)).unsqueeze(0)
                self.pos_emb_v = self.pos_emb_v.reshape(1,d_model,-1)
                self.pos_emb_v = self.pos_emb_v.permute(0,2,1).cuda()

        if self.do_posemb_l is True:
            if is_sinusoidal is False:
                self.pos_emb_l = nn.Parameter(torch.randn(1, 20, d_model))
                trunc_normal_(self.pos_emb_l, std=0.02)
            else:
                self.pos_emb_l = positionalencoding1d(d_model, 20).unsqueeze(0).cuda()

        self.proj_v = nn.Linear(d_encoder, d_model)
        self.proj_l = nn.Linear(2*d_model, d_model) if is_bilstm is True else nn.Linear(d_model, d_model)

        self.mm_emb = nn.Parameter(torch.randn(1, n_mm, d_model))

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_sent = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )
        self.decoder_norm = nn.LayerNorm(d_model)

        self.is_mask_norm = is_mask_norm
        self.num_shared = num_shared
        if is_mask_norm == True:
            self.mask_norm = nn.LayerNorm(1)
        
        self.apply(init_weights)
        trunc_normal_(self.mm_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x_v, x_l, is_distilled, im_size, lang=None):
        H, W = im_size
        GS = H // self.patch_size

        # Remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + is_distilled
        x_v = x_v[:, num_extra_tokens:]

        # Projection x_v and x_l
        x_v = self.proj_v(x_v)
        x_l = self.proj_l(x_l)

        # Normalization like previous Referring Image Segmentation to fuse the features from different domain
        x_v = F.normalize(x_v, dim=2) if self.no_vlnorm is False else x_v
        x_l = F.normalize(x_l, dim=2) if self.no_vlnorm is False else x_l

        # If do_* is True we add positional embedding on the projected features from the both modality.
        if self.do_posemb_v is True:
            pos_emb_v = self.pos_emb_v
            x_v = x_v + pos_emb_v
        if self.do_posemb_l is True:
            pos_emb_l = self.pos_emb_l
            x_l = x_l + pos_emb_l

        mm_emb = self.mm_emb.expand(x_v.size(0), -1, -1)
        # pdb.set_trace()
        x = torch.cat((x_v, x_l, mm_emb), 1)

        for _ in range(self.num_shared):
            for blk in self.blocks:
                x = blk(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat, sent_feat = x[:, : -self.n_cls-1], x[:, -self.n_cls-1 :-1], x[:, -1:]
        patches = patches @ self.proj_patch
        sent_feat = sent_feat @ self.proj_sent

        patches = patches / patches.norm(dim=-1, keepdim=True)
        # sent_feat = sent_feat / sent_feat.norm(dim=-1, keepdim=True) 

        masks = einsum('b p d, b w d -> b p w', patches, sent_feat)

        if self.is_mask_norm == True:
            masks = self.mask_norm(masks)

        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))

        return masks

# projection of visual feature
class Decoder_weighTR2(nn.Module):
    def __init__(
        self,
        # n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_lang,
        d_ff,
        drop_path_rate,
        dropout,
        do_posemb_v,
        do_posemb_l,
        num_patches,
        no_grad_w,
        is_mask_norm,
        num_shared,
        no_vlnorm,
        is_sinusoidal,
        n_mm = 1,
        is_bilstm=False,
        lang_pad_mask =False,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = 20                     # 수정 필요 나중에
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5
        self.do_posemb_v = do_posemb_v
        self.do_posemb_l = do_posemb_l
        self.no_grad_w = no_grad_w
        self.no_vlnorm = no_vlnorm
        self.lang_pad_mask = lang_pad_mask

        if self.do_posemb_v is True:
            if is_sinusoidal is False:
                self.pos_emb_v = nn.Parameter(torch.randn(1, num_patches, d_model))
                trunc_normal_(self.pos_emb_v, std=0.02)
            else:
                self.pos_emb_v = positionalencoding2d(d_model, int(num_patches**0.5), int(num_patches**0.5)).unsqueeze(0)
                self.pos_emb_v = self.pos_emb_v.reshape(1,d_model,-1)
                self.pos_emb_v = self.pos_emb_v.permute(0,2,1).cuda()

        if self.do_posemb_l is True:
            if is_sinusoidal is False:
                self.pos_emb_l = nn.Parameter(torch.randn(1, 20, d_model))
                trunc_normal_(self.pos_emb_l, std=0.02)
            else:
                self.pos_emb_l = positionalencoding1d(d_model, 20).unsqueeze(0).cuda()

        self.proj_v = nn.Linear(d_encoder, d_model)
        self.proj_l = nn.Linear(2*d_model, d_model) if is_bilstm is True else nn.Linear(d_model, d_model)

        self.mm_emb = nn.Parameter(torch.randn(1, n_mm, d_model))

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_sent = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )
        self.decoder_norm = nn.LayerNorm(d_model)

        self.is_mask_norm = is_mask_norm
        self.num_shared = num_shared
        if is_mask_norm == True:
            self.mask_norm = nn.LayerNorm(1)
        
        self.apply(init_weights)
        trunc_normal_(self.mm_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x_v, x_l, is_distilled, im_size, lang=None):
        H, W = im_size
        GS = H // self.patch_size

        # Remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + is_distilled
        x_v = x_v[:, num_extra_tokens:]

        # Normalization like previous Referring Image Segmentation to fuse the features from different domain
        x_v = F.normalize(x_v, dim=2) if self.no_vlnorm is False else x_v
        x_l = F.normalize(x_l, dim=2) if self.no_vlnorm is False else x_l

        # Projection x_v and x_l
        x_v = self.proj_v(x_v)
        x_l = self.proj_l(x_l)

        # If do_* is True we add positional embedding on the projected features from the both modality.
        if self.do_posemb_v is True:
            pos_emb_v = self.pos_emb_v
            x_v = x_v + pos_emb_v
        if self.do_posemb_l is True:
            pos_emb_l = self.pos_emb_l
            x_l = x_l + pos_emb_l
        if self.lang_pad_mask is True:
            x_l[lang.unsqueeze(2).expand(-1, -1, x_l.size(2))==0] = 0
        # concatenate vis + lang + mmemb
        mm_emb = self.mm_emb.expand(x_v.size(0), -1, -1)
        x = torch.cat((x_v, x_l, mm_emb), 1)

        # Feed-foward to Transfomer encoder block
        for _ in range(self.num_shared):
            for blk in self.blocks:
                x = blk(x)
        x = self.decoder_norm(x)

        # Dice for each modality and Projection
        patches, cls_seg_feat, sent_feat = x[:, : -self.n_cls-1], x[:, -self.n_cls-1 :-1], x[:, -1:]
        patches = patches @ self.proj_patch
        sent_feat = sent_feat @ self.proj_sent

        patches = patches / patches.norm(dim=-1, keepdim=True)
        # sent_feat = sent_feat / sent_feat.norm(dim=-1, keepdim=True)

        masks = einsum('b p d, b w d -> b p w', patches, sent_feat)

        if self.is_mask_norm == True:
            masks = self.mask_norm(masks)

        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))

        return masks


# (a-1) concat-conv
class Decoder_vlconcc(nn.Module):
    def __init__(
        self,
        # n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_lang,
        d_ff,
        drop_path_rate,
        dropout,
        do_posemb_v,
        do_posemb_l,
        num_patches,
        no_grad_w,
        num_shared,
        no_vlnorm,
        no_dnorm,
        no_projvl,
        no_projps,
        is_sinusoidal,
        vlnorm_dim,
        n_mm = 1,
        is_bilstm = False,
        lang_pad_mask = False,
        n_up = 4,
        is_eval = False,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = 20
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5
        self.do_posemb_v = do_posemb_v
        self.do_posemb_l = do_posemb_l
        self.no_grad_w = no_grad_w
        self.no_vlnorm = no_vlnorm
        self.no_dnorm = no_dnorm
        self.no_projvl = no_projvl
        self.no_projps = no_projps
        self.lang_pad_mask = lang_pad_mask
        self.vlnorm_dim = vlnorm_dim
        self.is_eval = is_eval

        if self.do_posemb_v is True:
            if is_sinusoidal is False:
                self.pos_emb_v = nn.Parameter(torch.randn(1, num_patches, d_model))
                trunc_normal_(self.pos_emb_v, std=0.02)
            else:
                self.pos_emb_v = positionalencoding2d(d_model, int(num_patches**0.5), int(num_patches**0.5)).unsqueeze(0)
                self.pos_emb_v = self.pos_emb_v.reshape(1,d_model,-1)
                self.pos_emb_v = self.pos_emb_v.permute(0,2,1).cuda()

        if self.do_posemb_l is True:
            if is_sinusoidal is False:
                self.pos_emb_l = nn.Parameter(torch.randn(1, 20, d_model))
                trunc_normal_(self.pos_emb_l, std=0.02)
            else:
                self.pos_emb_l = positionalencoding1d(d_model, 20).unsqueeze(0).cuda()

        self.proj_v = nn.Linear(d_encoder, d_model)
        self.proj_l = nn.Linear(2*d_model, d_model) if is_bilstm is True else nn.Linear(d_model, d_model)

        # self.proj_m = nn.Linear(2*d_model, d_model)
        # Define spatial coordinate tensor

        self.proj_m = nn.Sequential(
            nn.Linear(2*d_model+8, d_model),
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model//2),
            nn.Linear(d_model//2, d_model//4),
            nn.LeakyReLU(inplace=True),
            nn.Linear(d_model//4, 1)
        )

        self.decoder_norm = nn.LayerNorm(d_model)
        self.proj_mask = nn.Linear(self.n_cls, 1)
        
        self.apply(init_weights)

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
        x_v = F.normalize(x_v, dim=self.vlnorm_dim) if self.no_vlnorm is False else x_v
        x_l = F.normalize(x_l, dim=self.vlnorm_dim) if self.no_vlnorm is False else x_l

        # 2. Projection x_v and x_l
        x_v = self.proj_v(x_v) if self.no_projvl is False else x_v
        x_l = self.proj_l(x_l) 

        
        x_l = torch.mean(x_l, dim=1,keepdim=True)
        x_l = x_l.expand(-1,x_v.size(1),-1)
        spa = generate_spatial_batch(GS, GS).repeat(x_v.size(0), 1, 1, 1)
        spa = Variable(spa).cuda()
        spa = rearrange(spa, "b n h w -> b (h w) n")
        # pdb.set_trace()
        x_m = torch.cat((x_v, x_l, spa), 2)

        masks = self.proj_m(x_m)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))
        # pdb.set_trace()
        if self.is_eval is False:
            return masks
        else:
            return masks, None, None


# (a) Vanilla self-attention encoder
class Decoder_vlst(nn.Module):
    def __init__(
        self,
        # n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_lang,
        d_ff,
        drop_path_rate,
        dropout,
        do_posemb_v,
        do_posemb_l,
        num_patches,
        no_grad_w,
        num_shared,
        no_vlnorm,
        no_dnorm,
        no_projvl,
        no_projps,
        is_sinusoidal,
        vlnorm_dim,
        n_mm = 1,
        is_bilstm = False,
        lang_pad_mask = False,
        n_up = 4,
        is_eval = False,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = 20
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5
        self.do_posemb_v = do_posemb_v
        self.do_posemb_l = do_posemb_l
        self.no_grad_w = no_grad_w
        self.no_vlnorm = no_vlnorm
        self.no_dnorm = no_dnorm
        self.no_projvl = no_projvl
        self.no_projps = no_projps
        self.lang_pad_mask = lang_pad_mask
        self.vlnorm_dim = vlnorm_dim
        self.is_eval = is_eval

        if self.do_posemb_v is True:
            if is_sinusoidal is False:
                self.pos_emb_v = nn.Parameter(torch.randn(1, num_patches, d_model))
                trunc_normal_(self.pos_emb_v, std=0.02)
            else:
                self.pos_emb_v = positionalencoding2d(d_model, int(num_patches**0.5), int(num_patches**0.5)).unsqueeze(0)
                self.pos_emb_v = self.pos_emb_v.reshape(1,d_model,-1)
                self.pos_emb_v = self.pos_emb_v.permute(0,2,1).cuda()

        if self.do_posemb_l is True:
            if is_sinusoidal is False:
                self.pos_emb_l = nn.Parameter(torch.randn(1, 20, d_model))
                trunc_normal_(self.pos_emb_l, std=0.02)
            else:
                self.pos_emb_l = positionalencoding1d(d_model, 20).unsqueeze(0).cuda()

        self.proj_v = nn.Linear(d_encoder, d_model)
        self.proj_l = nn.Linear(2*d_model, d_model) if is_bilstm is True else nn.Linear(d_model, d_model)

        self.mm_emb = nn.Parameter(torch.randn(1, n_mm, d_model))

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_sent = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        if self.is_eval is False:
            self.blocks = nn.ModuleList(
                [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
            )
        else:
            self.blocks = nn.ModuleList(
                [Block_eval(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
            )

        
        self.decoder_norm = nn.LayerNorm(d_model)
        self.num_shared = num_shared
        
        self.apply(init_weights)
        trunc_normal_(self.mm_emb, std=0.02)

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
        x_v = F.normalize(x_v, dim=self.vlnorm_dim) if self.no_vlnorm is False else x_v
        x_l = F.normalize(x_l, dim=self.vlnorm_dim) if self.no_vlnorm is False else x_l

        # 2. Projection x_v and x_l
        x_v = self.proj_v(x_v) if self.no_projvl is False else x_v
        x_l = self.proj_l(x_l) 

        # 3. If do_* is True we add positional embedding on the projected features from the both modality.
        # if self.do_posemb_v is True:
        #     pos_emb_v = self.pos_emb_v
        #     x_v = x_v + pos_emb_v
        # if self.do_posemb_l is True:
        #     pos_emb_l = self.pos_emb_l
        #     x_l = x_l + pos_emb_l

        if self.lang_pad_mask is True:
            x_l[lang.unsqueeze(2).expand(-1, -1, x_l.size(2))==0] = 0
            
        mm_emb = self.mm_emb.expand(x_v.size(0), -1, -1)

        # 4. Feed-foward to Transfomer encoder block
        x = torch.cat((x_v, x_l, mm_emb), 1)
        if self.is_eval is False:
            for _ in range(self.num_shared):
                for blk in self.blocks:
                    x = blk(x)
        else:
            attn_list = []
            for _ in range(self.num_shared):
                for blk in self.blocks:
                    x, attn = blk(x)
                    attn_list.append(attn)
                attn = torch.cat(attn_list, dim=0)

        x = self.decoder_norm(x) if self.no_dnorm is False else x

        # 5. Dice for each modality and Projection
        patches, cls_seg_feat, sent_feat = x[:, : -self.n_cls-1], x[:, -self.n_cls-1 :-1], x[:, -1:]
        patches = patches @ self.proj_patch if self.no_projps is False else patches
        sent_feat = sent_feat @ self.proj_sent if self.no_projps is False else sent_feat

        masks = einsum('b p d, b w d -> b p w', patches, sent_feat)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))

        if self.is_eval is False:
            return masks
        else:
            return masks, [attn], None

# (b) independent between vision and memory path
class Decoder_independvltlst(nn.Module):
    def __init__(
        self,
        # n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_lang,
        d_ff,
        drop_path_rate,
        dropout,
        do_posemb_v,
        do_posemb_l,
        num_patches,
        no_grad_w,
        num_shared,
        no_vlnorm,
        no_dnorm,
        no_projvl,
        no_projps,
        is_sinusoidal,
        vlnorm_dim,
        n_mm = 1,
        is_bilstm = False,
        lang_pad_mask = False,
        n_up = 4,
        is_eval = False
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = 20
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5
        self.do_posemb_v = do_posemb_v
        self.do_posemb_l = do_posemb_l
        self.no_grad_w = no_grad_w
        self.no_vlnorm = no_vlnorm
        self.no_dnorm = no_dnorm
        self.no_projvl = no_projvl
        self.no_projps = no_projps
        self.lang_pad_mask = lang_pad_mask
        self.vlnorm_dim = vlnorm_dim
        self.is_eval = is_eval

        if self.do_posemb_v is True:
            if is_sinusoidal is False:
                self.pos_emb_v = nn.Parameter(torch.randn(1, num_patches, d_model))
                trunc_normal_(self.pos_emb_v, std=0.02)
            else:
                self.pos_emb_v = positionalencoding2d(d_model, int(num_patches**0.5), int(num_patches**0.5)).unsqueeze(0)
                self.pos_emb_v = self.pos_emb_v.reshape(1,d_model,-1)
                self.pos_emb_v = self.pos_emb_v.permute(0,2,1).cuda()

        if self.do_posemb_l is True:
            if is_sinusoidal is False:
                self.pos_emb_l = nn.Parameter(torch.randn(1, 20, d_model))
                trunc_normal_(self.pos_emb_l, std=0.02)
            else:
                self.pos_emb_l = positionalencoding1d(d_model, 20).unsqueeze(0).cuda()

        self.proj_v = nn.Linear(d_encoder, d_model)
        self.proj_l = nn.Linear(2*d_model, d_model) if is_bilstm is True else nn.Linear(d_model, d_model)

        self.mm_emb = nn.Parameter(torch.randn(1, n_mm, d_model))

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_sent = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        if self.is_eval is False:
            self.blocksvl = nn.ModuleList(
                [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
            )
            self.blockslm = nn.ModuleList(
                [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
            )
        else:
            self.blocksvl = nn.ModuleList(
                [Block_eval(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
            )
            self.blockslm = nn.ModuleList(
                [Block_eval(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
            )

        self.decoder_norm = nn.LayerNorm(d_model)
        self.num_shared = num_shared

        self.apply(init_weights)
        trunc_normal_(self.mm_emb, std=0.02)

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
        x_v = F.normalize(x_v, dim=self.vlnorm_dim) if self.no_vlnorm is False else x_v
        x_l = F.normalize(x_l, dim=self.vlnorm_dim) if self.no_vlnorm is False else x_l

        # 2. Projection x_v and x_l
        x_v = self.proj_v(x_v) if self.no_projvl is False else x_v
        x_l = self.proj_l(x_l) 

        # 3. If do_* is True we add positional embedding on the projected features from the both modality.
        if self.do_posemb_v is True:
            pos_emb_v = self.pos_emb_v
            x_v = x_v + pos_emb_v
        if self.do_posemb_l is True:
            pos_emb_l = self.pos_emb_l
            x_l = x_l + pos_emb_l

        if self.lang_pad_mask is True:
            x_l[lang.unsqueeze(2).expand(-1, -1, x_l.size(2))==0] = 0
            
        mm_emb = self.mm_emb.expand(x_v.size(0), -1, -1)

        # 4. Feed-foward to Transfomer encoder block
        x = torch.cat((x_v, x_l), 1)
        if self.is_eval is False:
            for _ in range(self.num_shared):
                for blk in self.blocksvl:
                    x = blk(x)
        else:
            attnvl_list = []
            for _ in range(self.num_shared):
                for blk in self.blocksvl:
                    x, attn = blk(x)
                    attnvl_list.append(attn)
                attnvl = torch.cat(attnvl_list, dim=0)

        x = self.decoder_norm(x) if self.no_dnorm is False else x
        patches, x_l_v = x[:, :-self.n_cls], x[:, -self.n_cls:]

        x = torch.cat((x_l, mm_emb), 1)
        if self.is_eval is False:
            for _ in range(self.num_shared):
                for blk in self.blockslm:
                    x = blk(x)
        else:
            attnlm_list = []
            for _ in range(self.num_shared):
                for blk in self.blockslm:
                    x, attn = blk(x)
                    attnlm_list.append(attn)
                attnlm = torch.cat(attnlm_list, dim=0)

        x = self.decoder_norm(x) if self.no_dnorm is False else x

        # 5. Dice for each modality and Projection
        cls_seg_feat, sent_feat = x[:, :-1], x[:, -1:]

        patches = patches @ self.proj_patch if self.no_projps is False else patches
        sent_feat = sent_feat @ self.proj_sent if self.no_projps is False else sent_feat

        masks = einsum('b p d, b w d -> b p w', patches, sent_feat)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))

        if self.is_eval is False:
            return masks
        else:
            return masks, [attnvl, attnlm], None

# (c) indirectly conjugating between vision and memory path
class Decoder_vltlst(nn.Module):
    def __init__(
        self,
        # n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_lang,
        d_ff,
        drop_path_rate,
        dropout,
        do_posemb_v,
        do_posemb_l,
        num_patches,
        no_grad_w,
        num_shared,
        no_vlnorm,
        no_dnorm,
        no_projvl,
        no_projps,
        is_sinusoidal,
        vlnorm_dim,
        n_mm = 1,
        is_bilstm = False,
        lang_pad_mask = False,
        n_up = 4,
        is_eval = False
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = 20
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5
        self.do_posemb_v = do_posemb_v
        self.do_posemb_l = do_posemb_l
        self.no_grad_w = no_grad_w
        self.no_vlnorm = no_vlnorm
        self.no_dnorm = no_dnorm
        self.no_projvl = no_projvl
        self.no_projps = no_projps
        self.lang_pad_mask = lang_pad_mask
        self.vlnorm_dim = vlnorm_dim
        self.is_eval = is_eval

        if self.do_posemb_v is True:
            if is_sinusoidal is False:
                self.pos_emb_v = nn.Parameter(torch.randn(1, num_patches, d_model))
                trunc_normal_(self.pos_emb_v, std=0.02)
            else:
                self.pos_emb_v = positionalencoding2d(d_model, int(num_patches**0.5), int(num_patches**0.5)).unsqueeze(0)
                self.pos_emb_v = self.pos_emb_v.reshape(1,d_model,-1)
                self.pos_emb_v = self.pos_emb_v.permute(0,2,1).cuda()

        if self.do_posemb_l is True:
            if is_sinusoidal is False:
                self.pos_emb_l = nn.Parameter(torch.randn(1, 20, d_model))
                trunc_normal_(self.pos_emb_l, std=0.02)
            else:
                self.pos_emb_l = positionalencoding1d(d_model, 20).unsqueeze(0).cuda()

        self.proj_v = nn.Linear(d_encoder, d_model)
        self.proj_l = nn.Linear(2*d_model, d_model) if is_bilstm is True else nn.Linear(d_model, d_model)

        self.mm_emb = nn.Parameter(torch.randn(1, n_mm, d_model))

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_sent = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]

        if self.is_eval is False:
            self.blocksvl = nn.ModuleList(
                [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
            )
            self.blockslm = nn.ModuleList(
                [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
            )
        else:
            self.blocksvl = nn.ModuleList(
                [Block_eval(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
            )
            self.blockslm = nn.ModuleList(
                [Block_eval(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
            )

        self.decoder_norm = nn.LayerNorm(d_model)
        self.num_shared = num_shared
        
        self.apply(init_weights)
        trunc_normal_(self.mm_emb, std=0.02)

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
        x_v = F.normalize(x_v, dim=self.vlnorm_dim) if self.no_vlnorm is False else x_v
        x_l = F.normalize(x_l, dim=self.vlnorm_dim) if self.no_vlnorm is False else x_l

        # 2. Projection x_v and x_l
        x_v = self.proj_v(x_v) if self.no_projvl is False else x_v
        x_l = self.proj_l(x_l) 

        # 3. If do_* is True we add positional embedding on the projected features from the both modality.
        if self.do_posemb_v is True:
            pos_emb_v = self.pos_emb_v
            x_v = x_v + pos_emb_v
        if self.do_posemb_l is True:
            pos_emb_l = self.pos_emb_l
            x_l = x_l + pos_emb_l

        if self.lang_pad_mask is True:
            x_l[lang.unsqueeze(2).expand(-1, -1, x_l.size(2))==0] = 0
            
        mm_emb = self.mm_emb.expand(x_v.size(0), -1, -1)

        # 4. Feed-foward to Transfomer encoder block
        x = torch.cat((x_v, x_l), 1)
        if self.is_eval is False:
            for _ in range(self.num_shared):
                for blk in self.blocksvl:
                    x = blk(x)
        else:
            attnvl_list = []
            for _ in range(self.num_shared):
                for blk in self.blocksvl:
                    x, attn = blk(x)
                    attnvl_list.append(attn)
                attnvl = torch.cat(attnvl_list, dim=0)
        patches, x_l = x[:, :-self.n_cls], x[:, -self.n_cls:]

        x = torch.cat((x_l, mm_emb), 1)
        if self.is_eval is False:
            for _ in range(self.num_shared):
                for blk in self.blockslm:
                    x = blk(x)
        else:
            attnlm_list = []
            for _ in range(self.num_shared):
                for blk in self.blockslm:
                    x, attn = blk(x)
                    attnlm_list.append(attn)
                attnlm = torch.cat(attnlm_list, dim=0)
        x = self.decoder_norm(x) if self.no_dnorm is False else x

        # 5. Dice for each modality and Projection
        cls_seg_feat, sent_feat = x[:, :-1], x[:, -1:]

        patches = patches @ self.proj_patch if self.no_projps is False else patches
        sent_feat = sent_feat @ self.proj_sent if self.no_projps is False else sent_feat

        masks = einsum('b p d, b w d -> b p w', patches, sent_feat)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))

        if self.is_eval is False:
            return masks
        else:
            return masks, [attnvl, attnlm], None









