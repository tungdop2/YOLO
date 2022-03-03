import albumentations as A
from albumentations.pytorch.transforms import ToTensor, ToTensorV2
import numpy as np
import cv2
import os
import torch

from utils import outputs_tensor_to_bbox
from model import YOLO
from non_max_suppression import non_max_suppression
import config as cfg

import argparse

def save_txt(bboxes, save_dir, save_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, save_name), 'w') as f:
        for bbox in bboxes:
            cls, x, y, w, h, conf = bbox
            f.write('{} {} {} {} {} {}\n'.format(cls, x, y, w, h, conf))
    f.close()

def detect(weights='yolov2.pt', path='datasets/val/images/', save_path='detect/yolov2/', save_txt=False):
    S = cfg.S
    BOX = cfg.BOX
    CLS = cfg.CLS
    H, W = cfg.H, cfg.W
    VAL_DIR = cfg.VAL_DIR
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    OUTPUT_THRESH = cfg.OUTPUT_THRESH
    CONF_THRESH = cfg.CONF_THRESH
    NMS_THRESH = cfg.NMS_THRESH


    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = YOLO(BOX=BOX, CLS=CLS)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()


    model.to(device)

    val_transforms = A.Compose([
        A.Resize(height=H, width=W),
        ToTensorV2(p=1.0),
    ])

    with torch.no_grad():
        for img in os.listdir(path):
            img_name = img
            img = os.path.join(path, img)
            raw = cv2.imread(img)
            img = raw.astype(np.float32) / 255.
            img = val_transforms(image=img)['image']

            output = model(img.unsqueeze(0).to(device))
            output = output.cpu().detach()
            bboxes = outputs_tensor_to_bbox(output[0], OUTPUT_THRESH)
            bboxes = non_max_suppression(bboxes, CONF_THRESH, NMS_THRESH)
            for bbox in bboxes:
                cls, x, y, w, h, conf = bbox
                x = x - w / 2
                y = y - h / 2
                x, y, w, h = x * raw.shape[1], y * raw.shape[0], w * raw.shape[1], h * raw.shape[0]

                cv2.rectangle(raw, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                cv2.putText(raw, '{:.2f}'.format(conf), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(save_path, img_name), raw)
            print('Saved image to {}'.format(os.path.join(save_path, img_name)))
            if save_txt:
                save_txt(bboxes, os.path.join(save_path, 'labels'), img_name.replace('.jpg', '.txt'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov2.pt')
    parser.add_argument('--path', type=str, default='datasets/val/images/')
    parser.add_argument('--save_path', type=str, default='detect/yolov2/')
    parser.add_argument('--save_txt', type=bool, default=False)
    args = parser.parse_args()
    detect(args.weights)
