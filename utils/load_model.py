import torch
from model.deeplab_refseg import Res_Deeplab


def load_model(model_name, num_classes=21, restore_from=None, gpu=0, feat_dim=1000, convbias=False):
    # Choose Backbone and Load initial parameter
    if model_name.lower() == 'resnet':
        if gpu == 0: print("ResNet101 backbone is loaded!!")
        model_V = Res_Deeplab(num_classes=21)
        restore_from = './model/ResNet101_init.pth'
        saved_state_dict = torch.load(restore_from)
        new_params = model_V.state_dict().copy()
        for i in saved_state_dict:
            #Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # print i_parts
            if (not num_classes == 21) or (not i_parts[1] == 'layer5') or (not i_parts[0]=='conv_last'):
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        model_V.load_state_dict(new_params)
        #model.float()
        #model.eval() # use_global_stats = True

    elif model_name.lower() == 'deeplab':
        if gpu == 0: print("Deeplab backbone is loaded!!")
        model_V = Res_Deeplab(num_classes=21, feat_dim=feat_dim, lstconvbias=convbias)
        restore_from = './model/Deeplab_VOC_init.pth'
        saved_state_dict = torch.load(restore_from)
        new_params = model_V.state_dict().copy()
        for i in saved_state_dict:
            #Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # print i_parts
            if not i_parts[0] == 'conv_last':
                new_params['.'.join(i_parts[:])] = saved_state_dict[i]
        model_V.load_state_dict(new_params)

    else:
        model = None
        if gpu == 0: print("Model is not loaded!!")

    return model_V
