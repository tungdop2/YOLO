def compute_iou(box1, box2):
    """Compute IOU between box1 and box2"""
    cls1, xc1, yc1, w1, h1, conf1 = box1
    cls2, xc2, yc2, w2, h2, conf2 = box2

    x11 = xc1 - w1/2
    x12 = xc1 + w1/2
    y11 = yc1 - h1/2
    y12 = yc1 + h1/2
    x21 = xc2 - w2/2
    x22 = xc2 + w2/2
    y21 = yc2 - h2/2
    y22 = yc2 + h2/2
    # print(x11, x12, y11, y12, x21, x22, y21, y22)
    area1, area2 = w1*h1, w2*h2
    intersect_w = max(0, min(x12, x22) - max(x11, x21))
    intersect_h = max(0, min(y12, y22) - max(y11, y21))
    intersect_area = intersect_w*intersect_h
    iou = intersect_area/(area1 + area2 - intersect_area)
    return iou


def non_max_suppression(b, CONF_THRESH=0.5, NMS_THRESH=0.2):
    """remove ovelap bboxes"""
    boxes = b.copy()
    boxes = sorted(boxes, key=lambda x: x[5], reverse=True)
    for i, current_box in enumerate(boxes):
        # print(i, current_box[5])
        if current_box[5] < CONF_THRESH:
            current_box[5] = 0
        if current_box[5] <= 0:
            continue
        for j in range(i+1, len(boxes)):
            iou = compute_iou(current_box, boxes[j])
            if iou > NMS_THRESH:
                boxes[j][5] = 0
    boxes = [box for box in boxes if box[5] > 0]
    return boxes