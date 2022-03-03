import numpy as np
import torch

import config as cfg

# S = cfg.S
# BOX = cfg.BOX
# CLS = cfg.CLS
# ANCHOR_BOXS = cfg.ANCHOR_BOXS

def get_bbox_from_line(line):
    line = line.split()
    cls = int(line[0])
    x, y, w, h = float(line[1]), float(line[2]), float(line[3]), float(line[4])
    return np.array([cls, x, y, w, h, 1])


def get_best_anchor(bbox, ANCHOR_BOXS=cfg.ANCHOR_BOXS):
    best_iou = 0
    best_anchor = 0
    cls, x, y, w, h, conf = bbox
    bbox_area = w * h
    for i, anchors in enumerate(ANCHOR_BOXS):
        anchor_area = anchors[0] * anchors[1]
        iou_w = min(w, anchors[0])
        iou_h = min(h, anchors[1])
        iou_area = iou_w * iou_h
        iou = iou_area / (bbox_area + anchor_area - iou_area)
        if iou > best_iou:
            best_iou = iou
            best_anchor = i
    return best_anchor


def bboxes_to_tensor(bboxes):
    S = cfg.S
    BOX = cfg.BOX
    CLS = cfg.CLS
    targets = torch.zeros((S, S, BOX, 5 + CLS))
    for bbox in bboxes:
        cls, x, y, w, h, conf = bbox
        grid_x = int(x * S)
        grid_y = int(y * S)
        w = w * S
        h = h * S
        cell_x = (x - grid_x / S) * S
        cell_y = (y - grid_y / S) * S
        box_idx = get_best_anchor([cls, x, y, w, h, conf])
        # print(cls, box_idx)
        targets[grid_y, grid_x, box_idx, 0:4] = torch.tensor([cell_x, cell_y, w, h])
        targets[grid_y, grid_x, box_idx, 4] = 1
        cls = int(cls)
        targets[grid_y, grid_x, box_idx, 5 + cls] = 1
    return targets

def targets_tensor_to_bbox(targets):
    S = cfg.S
    BOX = cfg.BOX
    bboxes = []
    for grid_x in range(S):
        for grid_y in range(S):
            # box_idx = 0
            for box_idx in range(BOX):
                obj_conf = targets[grid_y, grid_x, box_idx, 4]
                if obj_conf > 0:
                    # print(grid_y, grid_x, box_idx)
                    cell_x, cell_y, w, h = targets[grid_y, grid_x, box_idx, 0:4]
                    cls = targets[grid_y, grid_x, box_idx, 5:].argmax()
                    x = grid_x / S + cell_x / S
                    y = grid_y / S + cell_y / S
                    w = w / S
                    h = h / S
                    bboxes.append([cls, x, y, w, h, 1])

    return np.array(bboxes)


def outputs_tensor_to_bbox(outputs, OUTPUT_THRESH=0.5):
    S = cfg.S
    BOX = cfg.BOX
    ANCHOR_BOXS = cfg.ANCHOR_BOXS
    bboxes = []
    for grid_x in range(S):
        for grid_y in range(S):
            for box_idx in range(BOX):
                obj_conf = torch.sigmoid(outputs[grid_y, grid_x, box_idx, 4])
                cls_conf = torch.softmax(
                    outputs[grid_y, grid_x, box_idx, 5:], dim=-1).max()
                combime_conf = obj_conf * cls_conf
                if combime_conf > OUTPUT_THRESH:
                    cell_x, cell_y, w, h = outputs[grid_y,
                                                   grid_x, box_idx, 0:4]
                    cls = torch.softmax(
                        outputs[grid_y, grid_x, box_idx, 5:], dim=-1).argmax()
                    cell_x, cell_y = torch.sigmoid(
                        cell_x), torch.sigmoid(cell_y)
                    w, h = torch.exp(
                        w) * ANCHOR_BOXS[box_idx][0], torch.exp(h) * ANCHOR_BOXS[box_idx][1]
                    x = grid_x / S + cell_x / S
                    y = grid_y / S + cell_y / S
                    w = w / S
                    h = h / S
                    if x < 0 or y < 0 or x > 1 or y > 1 or w < 0 or h < 0 or w > 1 or h > 1:
                        continue
                    bboxes.append([cls.item(), x.item(), y.item(),
                                  w.item(), h.item(), combime_conf.item()])
    return bboxes
