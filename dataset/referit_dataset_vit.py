import os.path as osp
import numpy as np
from glob import glob
from torch.utils import data
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
# from utils.torchutils import crop_aug
# from utils.imutils import resize_and_crop, resize_and_pad
from utils.imutils import resize_and_pad
STATS = {
    "vit": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    "deit": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    "oldbgr": {"mean": (0.485, 0.456, 0.406), "std": (1, 1, 1)}
}

class ReferDataSet_vit(data.Dataset):
    def __init__(self, root, splitset, max_iters=None, crop_size=None, label_crop_size=None, ignore_label=255):
        self.root = root
        # self.crop_h, self.crop_w = crop_size
        self.crop_size = crop_size
        self.label_crop_size = label_crop_size
        self.transform = True
        self.normalization = STATS["vit"].copy()
        for k, v in self.normalization.items():
            v = np.round(255 * np.array(v), 2)
            self.normalization[k] = tuple(v)
        self.h_flip = transforms.RandomHorizontalFlip(p=1)
        self.aug_f = transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5)) 
        self.ignore_label = ignore_label
        # self.mean = mean
        self.set = splitset
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

        self.data_list = glob(osp.join(root, self.set+'_batch','*'))
        if not max_iters==None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            if max_iters < len(self.data_list):
                self.data_list = self.data_list[:max_iters]
        
        # import pdb; pdb.set_trace()

    # def aug(self, image, label, w=1024, h=512):
    #     return crop_aug(image, label, h, w, min_max_height=(320, 640), w2h_ratio=2)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        datafiles = np.load(self.data_list[index])

        name = self.data_list[index].split('/')[-1]
        name = name.split('.')[0]

        # pdb.set_trace()
        image = Image.fromarray(datafiles["im_batch"]).convert('RGB')
        label = datafiles["mask_batch"]
        text = datafiles["text_batch"]
        sent = datafiles["sent_batch"]

        # image = np.array(image, dtype=np.uint8)
        # augmented = self.aug(image, label, w=self.resize[0], h=self.resize[1])
        # image = augmented['image']
        # label = augmented['mask']

        image = np.asarray(image, np.float32)
        orig_size = image.shape
        label = np.expand_dims(np.asarray(label, np.int32), axis = 0)

        if self.crop_size is not None:
            image = resize_and_pad(image, self.crop_size[0], self.crop_size[1])
        
        if self.label_crop_size is not None:
            label = resize_and_pad(label, self.label_crop_size[0],self.crop_size[1])

        # import pdb; pdb.set_trace()

        # print(name)

        # image = 
        # image = image[:, :, ::-1]  # change to BGR
        image -= self.normalization["mean"]
        image /= self.normalization["std"]
        image = image.transpose((2, 0, 1))


        return image.copy(), label.copy(), orig_size, text, str(sent[0]), name