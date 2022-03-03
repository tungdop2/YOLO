import numpy as np
import os

def get_bboxes(path, height=416, width=416):
    bboxes = []
    with open(path, 'r') as f:
        for line in f:
            line = line.split()
            bboxes.append(line)

    bboxes = np.array(bboxes, dtype=np.float32)
    # bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * width
    # bboxes[:, [2, 4]] = bboxes[:, [2, 4]] * height
    return bboxes
            
def get_all_bboxes(path):
    all_bboxes = np.empty((0, 5))

    for file_ in os.listdir(path):
        if file_.endswith(".txt") and file_ != "classes.txt":
            file_path = os.path.join(path, file_)
            bboxes = get_bboxes(file_path)
            if len(bboxes) > 0:
                all_bboxes = np.concatenate((all_bboxes, bboxes), axis=0)


    return all_bboxes

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def get_boxes(bboxes):
    boxes = bboxes.copy()
    boxes = np.delete(boxes, [0], axis=1)
    boxes = xywh2xyxy(boxes)
    return boxes

def translate_boxes(boxes):
    """
    param:
        boxes: numpy array of shape (r, 4)
    return:
    numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)

def iou(box, clusters):
    """
   Calculate IOU
    param:
        box: tuple or array, shifted to the origin (i. e. width and height)
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
 
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    iou_ = intersection / (box_area + cluster_area - intersection + 1e-10)
 
    return iou_

def avg_iou(boxes, clusters):
    """
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])

def kmeans(boxes, k, dist=np.median):
    """
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        k: number of clusters
        dist: distance function
    return:
        numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]
    
    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))
    np.random.seed(42)

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]  # Initializing K poly class (method is randomly selected from the original data set)
    i = 0
    while i < 1:
        for row in range(rows):
        # Distance metric formula: D (Box, Centroid) = 1-IOU (Box, Centroid). The smaller the distance from the center of the cluster, the bigger the IOU value, so use 1 - IOU, so that the smaller the distance, the greater the IOU value.
            distances[row] = 1 - iou(boxes[row], clusters)  
        # Assign the label box to the nearest cluster center (that is, the code is to select (for each box) to the cluster center).
        # print(distances)
        nearest_clusters = np.argmin(distances, axis=1)
        # print(last_clusters)
        # print(nearest_clusters)

        # Until the cluster center changes to 0 (that is, the cluster center is unchanged).
        if (last_clusters == nearest_clusters).all():
            break
        # Update Cluster Center (here the median mediterraneous number of classes as new clustering center)
        for cluster in range(k):
            clusters[cluster]=dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters=nearest_clusters
        i +=1 

    return clusters

def define_anchor_boxes(path, k, cells=13):
    bboxes = get_all_bboxes(path)
    boxes = get_boxes(bboxes)
    boxes_t = translate_boxes(boxes)
    anchors = kmeans(boxes_t, k=k)
    anchors = anchors.astype(np.float32)
    anchors[:, 0] = anchors[:, 0] * cells
    anchors[:, 1] = anchors[:, 1] * cells
    # anchors = anchors / 2.
    return anchors

if __name__ == "__main__":
    path = "./datasets/train/labels"
    k = 3
    anchors = define_anchor_boxes(path, k)
    print(anchors)