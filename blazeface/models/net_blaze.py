import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(inp, oup, stride, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def depth_conv2d(inp, oup, kernel=1, stride=1, pad=0):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size = kernel, stride = stride, padding=pad, groups=inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, kernel_size=1)
    )


def conv_dw(inp, oup, stride, kernel_size=5, padding=2):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size, stride, padding, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(oup)
    )


def conv_pw(inp, oup, stride, kernel_size=5, padding=2):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(oup)
    )


class BlazeBlock(nn.Module):
    def __init__(self, inp, oup, double_oup=None, stride=1, kernel_size=5):
        super(BlazeBlock, self).__init__()
        assert stride in [1, 2] 
        self.stride = stride
        self.inp = inp 
        self.use_pooling = self.stride != 1
        self.shortcut_oup = double_oup or oup
        self.actvation = nn.ReLU(inplace=True)

        if double_oup == None: 
            
            self.conv = nn.Sequential( 
                    conv_dw(inp, oup, stride, kernel_size)
                )
        else:
            self.conv = nn.Sequential(
                    conv_dw(inp, oup, stride, kernel_size),
                    nn.ReLU(inplace=True),
                    conv_pw(oup, double_oup, 1, kernel_size),
                    nn.ReLU(inplace=True)
                )
        
        if self.use_pooling:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(kernel_size=stride, stride=stride),
                nn.Conv2d(in_channels=inp, out_channels=self.shortcut_oup, kernel_size=1, stride=1),
                nn.BatchNorm2d(self.shortcut_oup),
                nn.ReLU(inplace=True)
            ) 


    def forward(self,x):

        h = self.conv(x)

        if self.use_pooling:
            x = self.shortcut(x)

        z = h + x
        # print(z.size())
        return self.actvation(h + x)
        
        
class Blaze(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(Blaze, self).__init__()
        self.phase = phase
        self.num_classes = 2

        self.conv1 = conv_bn(3, 24, stride=2)
        self.conv2 = BlazeBlock(24, 24)
        self.conv3 = BlazeBlock(24, 24)
        self.conv4 = BlazeBlock(24, 48, stride=2)
        self.conv5 = BlazeBlock(48, 48)
        self.conv6 = BlazeBlock(48, 48)
        self.conv7 = BlazeBlock(48, 24, 96, stride=2)
        self.conv8 = BlazeBlock(96, 24, 96)
        self.conv9 = BlazeBlock(96, 24, 96)
        self.conv10 = BlazeBlock(96, 24, 96, stride=2)
        self.conv11 = BlazeBlock(96, 24, 96)
        self.conv12 = BlazeBlock(96, 24, 96)
        self.loc, self.conf, self.landm = self.multibox(self.num_classes)

    def multibox(self, num_classes):
        loc_layers = []
        conf_layers = []
        landm_layers = []

        loc_layers += [depth_conv2d(96, 2 * 4, kernel=3, pad=1)]
        conf_layers += [depth_conv2d(96, 2 * num_classes, kernel=3, pad=1)]
        landm_layers += [depth_conv2d(96, 2 * 10, kernel=3, pad=1)]

        loc_layers += [depth_conv2d(96, 6 * 4, kernel=3, pad=1)]
        conf_layers += [depth_conv2d(96, 6 * num_classes, kernel=3, pad=1)]
        landm_layers += [depth_conv2d(96, 6 * 10, kernel=3, pad=1)]

        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers), nn.Sequential(*landm_layers)

    
    def forward(self,inputs):
        detections = list()
        loc = list()
        conf = list()
        landm = list()

        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        detections.append(x9)

        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        detections.append(x12)


        for (x, l, c, lam) in zip(detections, self.loc, self.conf, self.landm):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            landm.append(lam(x).permute(0, 2, 3, 1).contiguous())

        bbox_regressions = torch.cat([o.view(o.size(0), -1, 4) for o in loc], 1)
        classifications = torch.cat([o.view(o.size(0), -1, 2) for o in conf], 1)
        ldm_regressions = torch.cat([o.view(o.size(0), -1, 10) for o in landm], 1)



        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output