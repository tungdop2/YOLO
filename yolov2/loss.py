import torch
import config as cfg


def post_process_output(output, device='cpu'):
    BOX = cfg.BOX
    ANCHOR_BOXS = cfg.ANCHOR_BOXS
    # xy
    xy = torch.sigmoid(output[:, :, :, :, :2]+1e-6)

    # wh
    wh = output[:, :, :, :, 2:4]
    anchors_wh = torch.Tensor(ANCHOR_BOXS).view(1, 1, 1, BOX, 2).to(device)
    wh = torch.exp(wh)*anchors_wh

    # objectness confidence
    obj_prob = torch.sigmoid(output[:, :, :, :, 4:5]+1e-6)

    # class distribution
    cls_dist = torch.softmax(output[:, :, :, :, 5:], dim=-1)
    return xy, wh, obj_prob, cls_dist


def post_process_target(target_tensor):
    xy = target_tensor[:, :, :, :, :2]
    wh = target_tensor[:, :, :, :, 2:4]
    obj_prob = target_tensor[:, :, :, :, 4:5]
    cls_dist = target_tensor[:, :, :, :, 5:]
    return xy, wh, obj_prob, cls_dist


def custom_loss(outputs, targets, device='cpu'):
    lambda_coord = 5.0
    lambda_noobj = 2.0
    lambda_obj = 5.0
    lambda_cls = 1.0

    pred_xy, pred_wh, pred_obj, pred_cls = post_process_output(outputs, device=device)
    target_xy, target_wh, target_obj, target_cls = post_process_target(targets)

    # best iou
    pred_tl = pred_xy - pred_wh/2
    pred_br = pred_xy + pred_wh/2
    target_tl = target_xy - target_wh/2
    target_br = target_xy + target_wh/2
    intersect_wh = torch.min(pred_br, target_br) - \
        torch.max(pred_tl, target_tl)
    intersect_wh = torch.max(intersect_wh, torch.zeros_like(intersect_wh))
    intersect_area = intersect_wh[:, :, :,
                                  :, 0] * intersect_wh[:, :, :, :, 1]
    pred_area = pred_wh[:, :, :, :, 0] * pred_wh[:, :, :, :, 1]
    target_area = target_wh[:, :, :, :, 0] * target_wh[:, :, :, :, 1]
    iou = intersect_area / (pred_area + target_area - intersect_area)
    best_iou = torch.max(iou, dim=3, keepdim=True)[0]
    best_iou_index = torch.unsqueeze(torch.eq(best_iou, iou).float(), -1)
    true_box_conf = best_iou_index * target_obj

    # compute loss
    loss_xy = torch.sum(torch.square(pred_xy - target_xy) * true_box_conf)
    loss_wh = torch.sum(torch.square(pred_wh - target_wh) * true_box_conf)
    loss_noobj = torch.sum(torch.square(
        pred_obj - target_obj) * (1 - true_box_conf))
    loss_obj = torch.sum(torch.square(pred_obj - target_obj) * true_box_conf)
    loss_cls = -torch.sum((target_cls * torch.log(pred_cls + 1e-6) +
                          (1 - target_cls) * torch.log(1 - pred_cls + 1e-6)) * true_box_conf)
    # print('Loss xy: ', loss_xy.item())
    # print('Loss wh: ', loss_wh.item())
    # print('Loss noobj: ', loss_noobj.item())
    # print('Loss obj: ', loss_obj.item())
    # print('Loss cls: ', loss_cls.item())
    # compute total loss
    loss = loss_xy * lambda_coord + loss_wh * lambda_coord * 5 + loss_obj * \
        lambda_obj + loss_cls * lambda_cls + loss_noobj * lambda_noobj
    return loss
