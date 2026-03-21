import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T

class VOCDataset(Dataset):
    def __init__(self, root, image_set='train', size=256):
        self.root = root
        self.size = size

        split_file = os.path.join(root, 'ImageSets/Segmentation', image_set+".txt")

        with open(split_file) as f:
            self.images = f.read().splitlines()

        self.img_transform = T.Compose([T.Resize((size, size)), T.ToTensor()])
        self.mask_transform = T.Resize((size,size), interpolation=Image.NEAREST)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_id = self.images[idx]

        img_path = os.path.join(self.root, 'JPEGImages', img_id+'.jpg')
        mask_path = os.path.join(self.root, 'SegmentationClass', img_id+'.png')

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        image = self.img_transform(image)
        mask = self.mask_transform(mask)

        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return image, mask