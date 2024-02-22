import os
import os.path as osp
import pdb
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from utils.torchutils import resize_and_pad, vis_torch_mask
import numpy.ma as ma
import skimage.segmentation as segmentation
import matplotlib.pyplot as plt

def save_mask_torch(image_torch, name, evel_dir, sent=None):
    image_torch = image_torch.cpu().data.numpy()
    if sent is not None:
        name = name + '-' + sent
    out_image = Image.fromarray(np.uint8(image_torch * 255), 'L')
    out_image.save('%s/%s_hard.png' % (evel_dir+'/pred_h', name))


def save_mask_softpatch_torch(image_torch, name, evel_dir, sent=None):
    vis_torch_mask(image_torch, '%s/%s_patchsoft.png' % (evel_dir+'/pred_cmap', name), cmap=True)
    image_torch = image_torch.cpu().data.numpy()
    if sent is not None:
        name = name + '-' + sent
    out_image = Image.fromarray(np.uint8(image_torch * 255), 'L')
    out_image.save('%s/%s_sigm.png' % (evel_dir+'/pred_sigm', name))

def save_mask_soft_torch(image_torch, name, evel_dir, sent=None):
    vis_torch_mask(image_torch, '%s/%s_soft.png' % (evel_dir+'/pred_cmap', name), cmap=True)
    image_torch = image_torch.cpu().data.numpy()
    if sent is not None:
        name = name + '-' + sent
    out_image = Image.fromarray(np.uint8(image_torch * 255), 'L')
    out_image.save('%s/%s_sigm.png' % (evel_dir+'/pred_sigm', name))

def save_img_gt(image, gt, name, sent, evel_dir):
    save_image = Image.fromarray(image.astype(np.uint8))
    save_gt = Image.fromarray(np.uint8(gt * 255), 'L')
    img_name = name + '-' + sent
    save_image.save('%s/%s.png' % (evel_dir+'/img', img_name))
    save_gt.save('%s/%s.png' % (evel_dir+'/gt', img_name))

def inter_vis(output_eval_dir, name, sent, sent_feat, word_patches):
    w_map_dir = osp.join(output_eval_dir, 'inter_analysis', name[0])
    if not os.path.exists(w_map_dir):
        os.makedirs(w_map_dir+'/norm_w_map')
        os.makedirs(w_map_dir+'/cls_norm_w_map')
        os.makedirs(w_map_dir+'/norm_weig_w_map')
        os.makedirs(w_map_dir+'/cls_norm_weig_w_map')

    val_sent_feat = sent_feat.permute(0,2,1).unsqueeze(2)
    val_sent_feat.shape, word_patches.shape
    weig_word_patches = val_sent_feat * word_patches
    weig_word_patches.shape

    # Word feature Normalization
    norm_word_patches = (word_patches - word_patches.min()) / (word_patches - word_patches.min()).max()
    chw_min_val = word_patches.min(2, keepdim=True)[0].min(3, keepdim=True)[0]
    chw_max_val = word_patches.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
    chw_norm_word_patches = (word_patches - chw_min_val) / (chw_max_val - chw_min_val)

    norm_weig_word_patches = (weig_word_patches - weig_word_patches.min()) / (weig_word_patches - weig_word_patches.min()).max()
    chw_min_val = weig_word_patches.min(2, keepdim=True)[0].min(3, keepdim=True)[0]
    chw_max_val = weig_word_patches.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
    chw_norm_weig_word_patches = (weig_word_patches - chw_min_val) / (chw_max_val - chw_min_val)

    for i in range(20):
        _, norm_w_patch = vis_torch_mask(norm_word_patches[:,i,:,:].permute(1,2,0)
                                        , '%s/%s.png' % (w_map_dir+'/norm_w_map', str(i)), cmap=True)
        _, chw_norm_w_patch = vis_torch_mask(chw_norm_word_patches[:,i,:,:].permute(1,2,0)
                                            , '%s/%s.png' % (w_map_dir+'/cls_norm_w_map', str(i)), cmap=True)
        _, norm_weig_w_patch = vis_torch_mask(norm_weig_word_patches[:,i,:,:].permute(1,2,0)
                                        , '%s/%s.png' % (w_map_dir+'/norm_weig_w_map', str(i)), cmap=True)
        _, chw_norm_weig_w_patch = vis_torch_mask(chw_norm_weig_word_patches[:,i,:,:].permute(1,2,0)
                                            , '%s/%s.png' % (w_map_dir+'/cls_norm_weig_w_map', str(i)), cmap=True)

    #sent
    sent_vect = sent_feat.cpu().data.numpy()
    f = open('%s/%s_sent_feat.txt' % (w_map_dir, sent),"w")
    for val in list(sent_vect.squeeze()):
        f.write('%.3f'%(val)+' ')
    f.close()



# mask.shape = (H, W, C) , type = array
# image.shape = (H, W, C) , type = PIL.Image
def masked_img_and_pred(output_dir, mask, image, sent):
    masked_image = ma.masked_where(mask == 0, mask)
    # plt.imshow(masked_image)
    contour = segmentation.find_boundaries(mask, connectivity=4, mode='thick', background=0)
    masked_contour = ma.masked_where(contour == 0, contour)
    plt.imshow(image, interpolation='none')
    plt.imshow(masked_image, 'Dark2', interpolation='none', alpha=0.3)
    plt.imshow(masked_contour, 'Dark2', interpolation='none', alpha=0.9)
    plt.title(sent)
    plt.axis('off')
    plt.savefig('%s-%s_masked_G.png' % (output_dir, sent), dpi = 250, bbox_inches = 'tight', pad_inches = 0)
    

def sent_dist(model, valloader, output_dir):
    valloader_iter = enumerate(valloader)

    sent_feat_list = []
    for i, batch in valloader_iter:
        images, labels, size, texts, sents, name = batch
        images.size(), labels.size(), size, texts, sents, name[0]
        orig_images = images
        B, _, orig_H, orig_W = images.size()
        images = resize_and_pad(images, 320, 320)
        images = Variable(images).cuda()
        labels = Variable(labels.float()).cuda()
        texts =Variable(texts).cuda()

        masks, word_patches, sent_feat = model.forward_vis(images, texts)
        sent_feat = sent_feat[0].cpu().data
    #     print(sent_feat)
        sent_feat_list.append(sent_feat)
        if i == 20000:
            break

    sent_feats = torch.cat(sent_feat_list)
    sent_feats = sent_feats.numpy()

    # sent ele dist
    mean_sent_feat = []
    for i in range(20):
        sent_feat_cls = sent_feats[:,i]
        rnd_sent_feat_cls = np.round(sent_feat_cls, 2)
    #     print(rnd_sent_feat_cls.min(), rnd_sent_feat_cls.max())
        unique, counts = np.unique(rnd_sent_feat_cls, return_counts = True)
    #     print(list(unique), counts)
        plt.bar(range(len(counts)), counts)
        ax = plt.subplot()
    #     ax.set_xticks(list(counts))
        ax.set_xticklabels(list(unique), rotation=70)
        plt.savefig('%s/%s.pdf' % (output_dir+'/sent_feat_dist', str(i)))
        plt.cla()
        mean_sent_feat.append(sent_feat_cls.mean())

    # mean dist
    x =np.arange(0,20)
    linear_model=np.polyfit(x,mean_sent_feat,3)
    linear_model_fn=np.poly1d(linear_model)
    ax = plt.subplot()
    plt.grid(True, axis='y', alpha=0.5, linestyle='--')
    plt.plot(range(len(mean_sent_feat)), mean_sent_feat, 'bo')
    plt.plot(x,linear_model_fn(x))
    ax.set_xticks(list(x))
    plt.savefig('%s/%s.pdf' % (output_dir+'/sent_feat_dist', "mean_sent"))


def attn_analysis(attn, h, w):
    n = h*w
    for i_attn in range(len(attn)):
        attn_torch = attn[i_attn]
        # pdb.set_trace()
        avg_attn_torch = attn_torch.mean(dim=1)     # average accross multi-head dimension
        avg_attn_torch
        sentemb_attn_torch = avg_attn_torch[:, -1, :]     # sent emb index -1 : [N_layers, N_v+N_l]
        # sum_vis_attn, sum_lan_attn = sentemb_attn[:,:n].sum(axis=1), sentemb_attn[:,n:-1].sum(axis=1)   # [N_layers, ]
        # avg_vis_attn, avg_lan_attn = sentemb_attn[:,:n].mean(axis=1), sentemb_attn[:,n:-1].mean(axis=1) # [N_layers, ]
        # fst_sum_vis_attn, fst_sum_lan_attn = sum_vis_attn[0], sum_lan_attn[0]
        # fst_avg_vis_attn, fst_avg_lan_attn = avg_vis_attn[0], avg_lan_attn[0]


        # sentemb_attn = avg_attn[:, :, -1]     # sent emb index -1 : [N_layers, N_v+N_l]
        # sum_vis_attn, sum_lan_attn = sentemb_attn[:,:n].sum(axis=1), sentemb_attn[:,n:-1].sum(axis=1)   # [N_layers, ]
        # avg_vis_attn, avg_lan_attn = sentemb_attn[:,:n].mean(axis=1), sentemb_attn[:,n:-1].mean(axis=1) # [N_layers, ]
        # fst_sum_vis_attn, fst_sum_lan_attn = sum_vis_attn[0], sum_lan_attn[0]
        # fst_avg_vis_attn, fst_avg_lan_attn = avg_vis_attn[0], avg_lan_attn[0]


        return sentemb_attn_torch
