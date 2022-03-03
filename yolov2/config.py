DATA_DIR = 'datasets/'
TRAIN_DIR = 'datasets/train/'
VAL_DIR = 'datasets/val/'

S = 13  # grid size
BOX = 3 # number of anchor boxes
ANCHOR_BOXS = [
    [1.7, 1.04],
    [3.471, 2.717],
    [2.21, 1.404]
]
CLS = 1  # number of class
H, W = 416, 416
OUTPUT_THRESH = 0.3
CONF_THRESH = 0.3
NMS_THRESH = 0.7
batch_size = 1
num_workers = 0
color_dict = {
    0: (0, 0, 255),
    1: (0, 255, 0),
    2: (255, 0, 0),
    3: (0, 255, 255),
    4: (255, 0, 255),
}

epochs = 100
lr = 0.001