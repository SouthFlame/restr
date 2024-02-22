
import os
import os.path as osp
import argparse
from eval.evaluate import evaluate

import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp

import config
from dataset.referit_dataset_vit import ReferDataSet_vit

from model.factory import create_restr
from utils.loss import AverageMeter, adjust_learning_rate
from utils.torchutils import Patch_label_gen, set_seed

import timeit
import wandb

start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
BATCH_SIZE = 8
DATA_DIRECTORY = './data/mscoco/Gref_batch'
SET = 'train'
VALSET = 'val'
INPUT_SIZE = '480,480'
LEARNING_RATE = 1e-5
END_LEARNING_RATE = 0
MOMENTUM = 0.9
POWER = 0.9

NUM_STEPS = 400000
SAVE_PRED_EVERY = 5000
WEIGHT_DECAY = 0.0005
EXPERIMENT_NAME = 'restr'
LOG_EVERY = 100

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")

    parser.add_argument("--v_backbone", type=str, default="vit_base_patch16_384")
    parser.add_argument("--l_backbone", type=str, default="transformer_glove")
    parser.add_argument("--mm_fusion", type=str, default="decoder_transformer")

    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY)
    parser.add_argument("--set", type=str, default=SET)
    parser.add_argument("--valset", type=str, default=VALSET)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--input_size", type=str, default=INPUT_SIZE)

    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--end_lr", type=float, default=END_LEARNING_RATE)
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS)
    parser.add_argument("--warm_iter", type=int, default=NUM_STEPS // 10)
    parser.add_argument("--adamW", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=0)
    parser.add_argument("--momentum", type=float, default=MOMENTUM)
    parser.add_argument("--power", type=float, default=POWER)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--is_cos", action="store_true")

    parser.add_argument("--is_shared", action="store_true")
    parser.add_argument("--tr_n_layers", type=int, default=2)

    parser.add_argument("--alpha_loc", type=float, default=0.1)
    parser.add_argument("--patch_thres", type=float, default=0.8)
    parser.add_argument("--no_decoder", action="store_true")

    parser.add_argument("--is_vis", action="store_true")
    parser.add_argument("--exp_name", type=str, default=EXPERIMENT_NAME)
    parser.add_argument("--save_every", type=int, default=SAVE_PRED_EVERY)
    parser.add_argument("--log_every", type=int, default=LOG_EVERY)
    parser.add_argument("--wandb_proj", type=str, default="RefImgSeg")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--amp", action="store_true")

    return parser.parse_args()

args = get_arguments()
set_seed(args.seed)

def make_model_cfg(args, cfg, input_size, dataset_name):
    model_cfg = {}
    v_backbone = args.v_backbone
    v_model_cfg = cfg["v_backbone"][v_backbone]
    v_model_cfg["image_size"] = input_size
    v_model_cfg["backbone"] = v_backbone
    model_cfg["v_backbone"] = v_model_cfg
    
    l_backbone = args.l_backbone
    l_model_cfg = cfg["l_backbone"][l_backbone]
    # l_model_cfg["backbone"] = l_backbone
    l_model_cfg["n_heads"] = v_model_cfg["n_heads"]
    l_model_cfg["emb_name"] = dataset_name
    l_model_cfg["d_model"] = v_model_cfg["d_model"]
    model_cfg["l_backbone"] = l_model_cfg

    fusion_module = args.mm_fusion
    mm_fusion_cfg = cfg["fusion_module"]
    mm_fusion_cfg["name"] = fusion_module
    mm_fusion_cfg["is_shared"] =args.is_shared
    mm_fusion_cfg["is_decoder"] = not args.no_decoder
    mm_fusion_cfg["n_heads"] = v_model_cfg["n_heads"]
    model_cfg["mm_fusion"] = mm_fusion_cfg
    
    return model_cfg


def main():
    """Create the model and start the training."""
    
    dataset_name = ((args.data_dir).split("/")[-1]).split("_")[0]
    print("Training dataset: {} | In {} of threads".format(dataset_name, torch.get_num_threads()))
    print("<Argument check>\n", vars(args))

    wandb_proj_name = args.wandb_proj + "_" + dataset_name
    wandb.init(project=wandb_proj_name, name=args.exp_name, config=args, entity='')

    h, w = map(int, args.input_size.split(',')) 
    input_size = (h, w)

    cfg = config.load_config()
    model_cfg = make_model_cfg(args, cfg, input_size, dataset_name)

    if args.no_decoder:
        args.alpha_loc = 0
    
    # Create network.
    model = create_restr(model_cfg)
    model.train()
    model.cuda()
    cudnn.enabled = True
    cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    num_model = sum(p.numel() for p in model.parameters())
    print("# of parameters: ", num_model)

    snapshot_dir = osp.join('./weights', args.exp_name)
    eval_dir = osp.join('./eval_dir_val', args.exp_name)
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    ## Call dataloader {Gref, unc, unc+, referit}
    trainloader = data.DataLoader(ReferDataSet_vit(args.data_dir, args.set, max_iters=args.num_steps * args.batch_size), 
                    batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    trainloader_iter = enumerate(trainloader)

    val_max_iters = 10000 if dataset_name == 'referit' else None
    valloader = data.DataLoader(ReferDataSet_vit(args.data_dir, args.valset, max_iters=val_max_iters), 
                            batch_size=1, shuffle=False, num_workers=1)

    ## Define Optimizer 
    optimizer = optim.AdamW(model.optim_parameters(args)
                            , lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    optimizer.zero_grad()

    ### Generating patch-level labels
    patch_label_gen = Patch_label_gen(model_cfg["v_backbone"]["patch_size"], threshold=args.patch_thres).cuda()
    
    ### Inpterpolation for predictions and labels
    interp_pred = nn.Upsample(size = input_size, mode='bilinear', align_corners=True)
    interp_label = nn.Upsample(size = input_size, mode='nearest')
    
    ### START Training
    best_IoU = 0.0
    losses = dict()
    losses['pixel'] = AverageMeter()
    losses['patch'] = AverageMeter()
    for i_iter in range(1, args.num_steps+1):
        model.train()
        
        lr = adjust_learning_rate(optimizer, i_iter-1, args.lr, args.end_lr, args.num_steps, args.power, args.warm_iter, args.is_cos)
        
        # Load dataset
        _, batch = next(trainloader_iter)
        images, labels, size, texts, sents, name = batch
        images = Variable(images).cuda()
        labels = Variable(labels.float()).cuda()
        p_labels = patch_label_gen(labels)
        p_labels = Variable(p_labels.float()).cuda()
        texts =Variable(texts).cuda()
        
        # Extract visual feature
        with torch.cuda.amp.autocast(enabled=args.amp):
            pixel_pred, _, patch_pred = model(images, texts)
            pixel_pred = interp_pred(pixel_pred)
            labels = interp_label(labels)

            # Loss for localization
            bce_loss_patch = torch.nn.BCEWithLogitsLoss()
            loss_patch = bce_loss_patch(patch_pred, p_labels)
            
            # Loss for segmentation
            bce_loss_pixel = torch.nn.BCEWithLogitsLoss()
            loss_pixel = bce_loss_pixel(pixel_pred, labels)

            loss = loss_pixel + args.alpha_loc * loss_patch
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        losses['pixel'].update(loss_pixel)
        losses['patch'].update(loss_patch)

        # Logger
        if i_iter % args.log_every == 0:
            print('iter = {0:7d}/{1:7d}, loss_pixel={2:.5f}, batch_size:{3:}'.format(i_iter, args.num_steps, losses['pixel'].avg, args.batch_size))
            wandb.log({
            "Loss_avg@pixel": losses['pixel'].avg,
            "Loss_avg@patch": losses['patch'].avg,
            "lr": lr,
            }, step=i_iter)
            losses['pixel'].reset()
            losses['patch'].reset()

        #Snapshot
        if (i_iter == 1) or (i_iter % args.save_every == 0):
            model.eval()
            output_eval_dir = os.path.join(eval_dir, str(i_iter))
            cum_IoU = evaluate(output_eval_dir, i_iter, model, valloader=valloader, H=input_size[0], W=input_size[1]
                , is_vis=args.is_vis, save_im_sent = True)
                
            print('taking snapshot ...')
            if i_iter !=1:
                wandb.log({"cum_IoU_val": cum_IoU}, step=i_iter)
            if i_iter == 0:
                best_IoU = 0
            if cum_IoU > best_IoU:
                best_IoU = cum_IoU 
                torch.save(model.state_dict(),osp.join(snapshot_dir, dataset_name + "_" + str(i_iter) + '.pth')) if i_iter > 10000 else None  
            elif i_iter > args.num_steps * 0.9:
                torch.save(model.state_dict(),osp.join(snapshot_dir, dataset_name + "_" + str(i_iter) + '.pth')) if i_iter > 10000 else None 

    end = timeit.default_timer()
    print(end-start,'seconds')

if __name__ == '__main__':
    main()
