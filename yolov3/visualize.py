import cv2
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
import torch

def visualize(img, bboxes, save=False, save_name=None):
    color_dict = cfg.color_dict
    if torch.is_tensor(img):
        img = img.numpy()
    vis_img = img.copy()
    W, H = img.shape[1], img.shape[0]
    vis_img = (vis_img * 255).astype(np.uint8)

    for bbox in bboxes:
        # print(bbox)
        cls, x, y, w, h, conf = bbox
        cls = int(cls)
        x = int(x * W)
        y = int(y * H)
        w = int(w * W)
        h = int(h * H)
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color_dict[cls], 2)
        text = "{} {:.2f}".format(cls, conf)
        text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.24, 1)
        cv2.rectangle(vis_img, (x1, y1 - text_size[1] - baseline), (x1 + text_size[0], y1), color_dict[cls], -1)
        cv2.putText(vis_img, text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.24, (0, 0, 0), 1)
    
    if save:
        print('Saving image to {}'.format(save_name))
        cv2.imwrite('{}'.format(save_name), vis_img)
    else:
        plt.figure(figsize=(10, 10))
        plt.imshow(vis_img[:, :, ::-1])
        plt.show()