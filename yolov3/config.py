DATA_DIR = 'datasets/'
TRAIN_DIR = 'datasets/train/'
VAL_DIR = 'datasets/val/'

S = [13, 26]  # grid size
BOX = 3  # number of anchor boxes
ANCHOR_BOXS = [
    [[1.7308248, 1.1209781],
    [3.4804633, 2.7271557],
    [2.2199712, 1.4054054]],

    [[3.4616497, 2.2419562],
    [6.9609265, 5.4543114],
    [4.4399424, 2.8108108]]]
CLS = 1  # number of class
H, W = 416, 416
OUTPUT_THRESH = 0.3
CONF_THRESH = 0.3
NMS_THRESH = 0.5
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
