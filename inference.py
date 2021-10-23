# You can run this file to operate your own image

import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class ResBlock(nn.Module):
    """Residual(残差) Block"""

    def __init__(self, inChannels, outChannels):
        """Initialize residual block"""
        super(ResBlock, self).__init__()
        # The super () function is used to call the parent class(父类)
        # to solve the problem of multiple inheritance(多继承).

        # If u want get more about convolution, batch normalization and other concepts(概念), please Baidu it.

        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=1, bias=False)
        # Why there  uses 1 x 1 kernel?
        # Because 1 x 1 kernel can not only deeper the feature map, but also can shallow the feature map.
        # More details please Baidu it.

        self.conv2 = nn.Conv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1, bias=False)
        # padding means "expand" the area, e.x. a 3x3 feature map padding=1 -> 5x5

        self.conv3 = nn.Conv2d(outChannels, outChannels, kernel_size=1, bias=False)
        self.relu = nn.PReLU()
        # The activation function can introduce nonlinear factors to solve the problems
        # that can not be solved by linear model.

    def forward(self, x):
        """Forward Spread"""

        resudial = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(x)
        out = self.relu(out)

        out = self.conv3(x)

        out += resudial
        out = self.relu(out)

        return out


class Generator(nn.Module):
    """Generate Model(4x)"""

    def __init__(self):
        """Initialize Model Configuration(配置)"""

        super(Generator, self).__init__()
        # Convolution Model 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4, padding_mode='reflect', stride=1)
        self.relu = nn.PReLU()

        # Residual Model
        self.resBlock = self._makeLayer_(ResBlock, 64, 64, 5)

        # Convolution Model 2
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.relu2 = nn.PReLU()

        # Subpixel(子像素) convolution
        self.convPos1 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=2, padding_mode='reflect')
        self.pixelShuffler1 = nn.PixelShuffle(2)
        self.reluPos1 = nn.PReLU()

        self.convPos2 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.pixelShuffler2 = nn.PixelShuffle(2)
        self.reluPos2 = nn.PReLU()

        self.finConv = nn.Conv2d(64, 3, kernel_size=9, stride=1)

    def _makeLayer_(self, block, inChannels, outChannels, blocks):
        """Construct Residual Block"""
        layers = []
        layers.append(block(inChannels, outChannels))

        for i in range(1, blocks):
            layers.append(block(outChannels, outChannels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward Procession"""
        x = self.conv1(x)
        x = self.relu(x)
        residual = x

        out = self.resBlock(x)

        out = self.conv2(out)
        out += residual

        out = self.convPos1(out)
        out = self.pixelShuffler1(out)
        out = self.reluPos1(out)

        out = self.convPos2(out)
        out = self.pixelShuffler2(out)
        out = self.reluPos2(out)

        out = self.finConv(out)

        return out


device = torch.device("cpu")
net = Generator()
net.load_state_dict(torch.load("./model/netG_epoch_4_61.pth", map_location=torch.device('cpu')))


def imshow(path, sourceImg=True):
    """Show results"""
    preTransform = transforms.Compose([transforms.ToTensor()])
    pilImg = Image.open(path)
    img = preTransform(pilImg).unsqueeze(0).to(device)

    source = net(img)[0, :, :, :]
    source = source.cpu().detach().numpy()  # Turn to numpy
    source = source.transpose((1, 2, 0))  # Transform shape
    source = np.clip(source, 0, 1)  # Correct pictures

    if sourceImg:
        temp = np.clip(img[0, :, :, :].cpu().detach().numpy().transpose((1, 2, 0)), 0, 1)
        shape = temp.shape
        source[-shape[0]:, :shape[1], :] = temp
        plt.imshow(source)
        img = Image.fromarray(np.uint8(source * 255))
        img.save('./result/' + path.split('/')[-1][:-4] + '_result.jpg')  # Save array as pictures
        return

    plt.imshow(source)
    img = Image.fromarray(np.uint8(source * 255))
    img.save(path[:-4] + '_result.jpg')  # Save arrays as pictures


imshow("yuzi.jpg", sourceImg=True)