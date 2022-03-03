import os
import cv2
import numpy as np
from utils import get_bbox_from_line, bboxes_to_tensor
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        imgs_folder = os.path.join(root_dir, 'images')
        labels_folder = os.path.join(root_dir, 'labels')
        self.imgs_list = os.listdir(imgs_folder)

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, idx):
        img_name = self.imgs_list[idx]
        img_path = os.path.join(self.root_dir, 'images', img_name)
        label_path = os.path.join(self.root_dir, 'labels', img_name.replace('jpg', 'txt'))
        raw_img = cv2.imread(img_path).astype(np.float32) / 255.0
        img = raw_img.copy()
        if self.transform:
            img = self.transform(image=img)['image']

        with open(label_path, 'r') as f:
            lines = f.readlines()
            bboxes = []
            for line in lines:
                bbox = get_bbox_from_line(line)
                bboxes.append(bbox)

        target = bboxes_to_tensor(bboxes)
        return img, target