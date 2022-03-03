import config as cfg

import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm


from dataset import CustomDataset
from model import YOLO, init_normal
from loss import custom_loss
from visualize import visualize
from utils import outputs_tensor_to_bbox


import argparse

def train(pretrain=None, save_model='model.pt'):
    TRAIN_DIR = cfg.TRAIN_DIR
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    BOX = cfg.BOX
    CLS = cfg.CLS
    H = cfg.H
    W = cfg.W
    OUTPUT_THRESH = cfg.OUTPUT_THRESH
    CONF_THRESH = cfg.CONF_THRESH 
    NMS_THRESH = cfg.NMS_THRESH
    epochs = cfg.epochs
    lr = cfg.lr
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    torch.autograd.set_detect_anomaly(True)

    train_transforms = A.Compose([
        A.Resize(height=H, width=W),
        ToTensorV2(p=1.0),
        # A.HorizontalFlip(p=0.5),
        # A.ToGray(p=0.1),
        # A.Blur(p=0.1),
    ])

    train_dataset = CustomDataset(TRAIN_DIR, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = YOLO(BOX=BOX, CLS=CLS)
    best_loss = 1e10
    best_model = None

    if pretrain is not None:
        model.load_state_dict(torch.load(pretrain, map_location=device))
        print("Load pretrained model from {}".format(pretrain))
    else:
        model.apply(init_normal)
        print("Initialize model with normal distribution")

    model = model.to(device)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[epochs // 3, epochs // 3 * 2], gamma=0.1)

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        for imgs, t0, t1 in iter(train_loader):
            imgs = imgs.to(device)
            t0 = t0.to(device)
            t1 = t1.to(device)
            o0, o1 = model(imgs)
            with torch.set_grad_enabled(True):
                loss = custom_loss(o0, t0, ANCHOR_BOXS=cfg.ANCHOR_BOXS[0])
                loss += custom_loss(o1, t1, ANCHOR_BOXS=cfg.ANCHOR_BOXS[1])
                optim.zero_grad()
                loss.backward()
                epoch_loss += loss.item()
                optim.step()

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = model.state_dict()
            if save_model is not None:
                torch.save(best_model, save_model)
                print("Save model to {}".format(save_model))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--save_model', type=str, default='yolov3.pt')
    args = parser.parse_args()
    train(args.pretrain)
