from pathlib import Path
import yaml
import torch
import math
import os
import torch.nn as nn

from timm.models.helpers import load_pretrained, load_custom_pretrained
from timm.models.vision_transformer import default_cfgs
from timm.models.registry import register_model
from timm.models.vision_transformer import _create_vision_transformer

from model.vit import VisionTransformer
from model.transformers.utils import checkpoint_filter_fn
from model.restr import ReSTR
from utils.load_model import load_model

from model.glove_transformer import *
from model.decoder import *
from model.decoder_locup import *
from model.fusion_module import *

def create_vit(v_model_cfg):
    v_model_cfg = v_model_cfg.copy()
    backbone = v_model_cfg.pop("backbone")
    normalization = v_model_cfg.pop("normalization")

    v_model_cfg["n_cls"] = 1000
    mlp_expansion_ratio = 4
    v_model_cfg["d_ff"] = mlp_expansion_ratio * v_model_cfg["d_model"]

    if backbone in default_cfgs:
        default_cfg = default_cfgs[backbone]
    else:
        default_cfg = dict(
            pretrained=False,
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.0,
            drop_block_rate=None,
        )

    default_cfg["input_size"] = (
        3,
        v_model_cfg["image_size"][0],
        v_model_cfg["image_size"][1],
    )
    model = VisionTransformer(**v_model_cfg)
    if backbone == "vit_base_patch8_384":
        path = os.path.expandvars("$TORCH_HOME/hub/checkpoints/vit_base_patch8_384.pth")
        state_dict = torch.load(path, map_location="cpu")
        filtered_dict = checkpoint_filter_fn(state_dict, model)
        model.load_state_dict(filtered_dict, strict=True)
    elif "deit" in backbone:
        load_pretrained(model, default_cfg, filter_fn=checkpoint_filter_fn)
    else:
        load_custom_pretrained(model, default_cfg)

    return model

def create_rnn(l_model_cfg):
    l_model_cfg = l_model_cfg.copy()
    
    #### Language embedding option
    #### vocab_size = 8803 if lang_model_cfg["dataset_name"] == 'referit' else 12112
    mlp_expansion_ratio = 4
    l_model_cfg["d_ff"] = mlp_expansion_ratio * l_model_cfg["d_lang"]
    model_L = TR_GloVe(**l_model_cfg)

    return model_L
    

def create_fusion_module(encoder_v, encoder_l, mm_fusion_cfg):

    mm_fusion_cfg = mm_fusion_cfg.copy()
    name = mm_fusion_cfg.pop("name")
    # image_size = mm_fusion_cfg.pop("image_size")
    mm_fusion_cfg["d_v_encoder"] = encoder_v.d_model
    mm_fusion_cfg["d_l_encoder"] = encoder_l.d_model
    # mm_fusion_cfg["d_lang"] = encoder_l.d_lang
    mm_fusion_cfg["patch_size"] = encoder_v.patch_size
    
    dim = encoder_v.d_model
    mm_fusion_cfg["d_model"] = dim
    # pdb.set_trace()
    mm_fusion_cfg["d_ff"] = 4 * dim

    # pdb.set_trace()
    # lower_name = name.lower()
    # if lower_name == 'decoder_vlcc':
    #     model_D = Decoder_vlconcc(**mm_fusion_cfg) 
    # elif lower_name == 'decoder_vlst':
    #     model_D = Decoder_vlst(**mm_fusion_cfg) 
    # elif lower_name == 'decoder_indvltlst':
    #     model_D = Decoder_independvltlst(**mm_fusion_cfg) 
    # elif lower_name == 'decoder_vltlst':
    #     model_D = Decoder_vltlst(**mm_fusion_cfg) 

    # elif lower_name == 'decoder_vlst_locup':
    #     model_D = Decoder_vlst_locup(**mm_fusion_cfg) 
    # elif lower_name == 'decoder_indvltlst_locup':
    #     model_D = Decoder_independvltlst_locup(**mm_fusion_cfg) 
    # elif lower_name == 'decoder_vltlst_locup':
    #     model_D = Decoder_vltlst_locup(**mm_fusion_cfg) 

    # else:
    #     print("Wrong decoder config name", name)
    #     assert False
    model_MM = MM_fusion(**mm_fusion_cfg)
    return model_MM


def create_restr(model_cfg):
    model_cfg = model_cfg.copy()
    v_model_cfg = model_cfg.pop("v_backbone")
    l_model_cfg = model_cfg.pop("l_backbone")
    mm_fusion_cfg = model_cfg.pop("mm_fusion")

    encoder_v = create_vit(v_model_cfg)
    encoder_l = create_rnn(l_model_cfg)
    mm_fusion = create_fusion_module(encoder_v, encoder_l, mm_fusion_cfg)
    model = ReSTR(encoder_v, encoder_l, mm_fusion, n_cls=1000)

    return model