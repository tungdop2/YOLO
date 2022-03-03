import torch
import torch.nn as nn

def init_normal(m):
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight.data, 0, 0.01)

# This is tiny-yolov3
class YOLO(nn.Module):
    def __init__(self, BOX=3, CLS=1):
        super(YOLO, self).__init__()
        self.BOX = BOX
        self.CLS = CLS
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.slowpool = nn.MaxPool2d(2, 1)
        self.pad = nn.ReflectionPad2d((0, 1, 0, 1))
        # self.pad = nn.Identity()
        self.norm1 = nn.BatchNorm2d(16, momentum=0.1)
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(32, momentum=0.1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1, bias=False)
        self.norm3 = nn.BatchNorm2d(64, momentum=0.1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.norm4 = nn.BatchNorm2d(128, momentum=0.1)
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.norm5 = nn.BatchNorm2d(256, momentum=0.1)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.norm6 = nn.BatchNorm2d(512, momentum=0.1)
        self.conv6 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.norm7 = nn.BatchNorm2d(1024, momentum=0.1)
        self.conv7 = nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
        self.norm8 = nn.BatchNorm2d(256, momentum=0.1)
        self.conv8 = nn.Conv2d(1024, 256, 1, 1, 0, bias=False)
        self.norm9 = nn.BatchNorm2d(512, momentum=0.1)
        self.conv9 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.detection1 = nn.Conv2d(512, BOX * (5 + CLS), 1, 1, 0, bias=False)
        self.norm10 = nn.BatchNorm2d(128, momentum=0.1)
        self.conv10 = nn.Conv2d(256, 128, 1, 1, 0, bias=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.norm11 = nn.BatchNorm2d(256, momentum=0.1)
        self.conv11 = nn.Conv2d(128 + 256, 256, 3, 1, 1, bias=False)
        self.detection2 = nn.Conv2d(256, BOX * (5 + CLS), 1, 1, 0, bias=False)
        
        

    def forward(self, x):
        x = self.relu(self.pool(self.norm1(self.conv1(x))))
        x = self.relu(self.pool(self.norm2(self.conv2(x))))
        x = self.relu(self.pool(self.norm3(self.conv3(x))))
        x = self.relu(self.pool(self.norm4(self.conv4(x))))
        route = self.relu(self.conv5(x))
        x = self.relu(self.pool(self.norm5(self.conv5(x))))
        x = self.relu(self.slowpool(self.pad(self.norm6(self.conv6(x)))))
        x = self.relu(self.norm7(self.conv7(x)))
        x  = self.relu(self.norm8(self.conv8(x)))
        output1  = self.relu(self.norm9(self.conv9(x)))
        output1 = self.detection1(output1)
        # x = 
        x = self.relu(self.upsample(self.norm10(self.conv10(x))))
        x = torch.cat((x, route), 1)
        x = self.relu(self.norm11(self.conv11(x)))
        output2 = self.detection2(x)

        output1 = output1.permute(0,2,3,1)
        output1 = output1.view(-1, 13, 13, self.BOX, 5 + self.CLS)
        output2 = output2.permute(0,2,3,1)
        output2 = output2.view(-1, 26, 26, self.BOX, 5 + self.CLS)
        return output1, output2
