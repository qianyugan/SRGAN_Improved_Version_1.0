# 此模型源于论文：https://arxiv.org/pdf/1609.04802.pdf
# 中文详解：https://perper.site/2019/03/01/SRGAN-%E8%AF%A6%E8%A7%A3/
# 代码参考：https://blog.csdn.net/NikkiElwin/article/details/112910957?spm=1001.2014.3001.5501
# 此模型是基于 SRGAN 的超分辨率重构模型。
# 数据集：./AnimeTest/，包含了 814 张二次元头像
# 模型保存至 ./model/
# 迭代生成的图片效果保存至 ./result/
# 张志衡复现并添加注释，英语可能有纰漏，望谅解！

import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.models.vgg import vgg16
import torchvision.transforms as transforms

# Image treatment: Crop images and transform to tensor
transform = transforms.Compose([transforms.RandomCrop(96), transforms.ToTensor()])


class PreprocessDataset(Dataset):  # Meaning class PreprocessDataset inherit from class Dataset
    """Preprocess dataset"""

    def __init__(self, imgPath, transforms=transform, ex=10):
        """Initialize preprocess dataset"""

        self.transforms = transform

        for _, _, files in os.walk(imgPath):  # Walking through directory imgPath
            self.imgs = [imgPath + file for file in files] * ex  # ... * ex means expand the dataset 10x

        np.random.shuffle(self.imgs)  # shuffle means make the dataset unordered

    def __len__(self):
        """Get len of dataset"""
        return len(self.imgs)

    def __getitem__(self, index):
        """Get images data"""
        tempImg = self.imgs[index]
        tempImg = Image.open(tempImg)

        sourceImg = self.transforms(tempImg)  # Process the raw images
        cropImg = torch.nn.MaxPool2d(4, stride=4)(sourceImg)
        # MaxPool2d the first and then Conv2d make the same result.
        # MaxPool2d first means "Subsampled(下采样), which can minus the process time

        return cropImg, sourceImg


path = './datasetPictures/'
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
batch = 814  # batch is belong to class "quantity", not "size". DO NOT confuse it with "epochs"
epochs = 100

# Construct Dataset
processDataset = PreprocessDataset(imgPath=path)
trainData = DataLoader(processDataset, batch_size=batch)

# Construct iterator and take out one of samples
dataiter = iter(trainData)
testImgs, labels = dataiter.next()
# .next() function: get next element from iterator
# "_" replaced the position of labels
# If there is no sample in iterator, it will return StopIteration

testImgs = testImgs.to(device)  # Use of testImgs is to make the generate against result visualize


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


class ConvBlock(nn.Module):
    """Construct Convolution Block"""

    def __init__(self, inChannels, outChannels, stride=1):
        """Initialize Residual Block"""

        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=stride, padding=1, padding_mode='reflect',
                              bias=False)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        """Forward Spread"""

        out = self.conv(x)
        out = self.relu(out)

        return out


class Discriminator(nn.Module):
    """Discriminator means ”鉴别器“ """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.relu1 = nn.LeakyReLU()

        self.convBlock1 = ConvBlock(64, 64, stride=2)
        self.convBlock2 = ConvBlock(64, 128, stride=1)
        self.convBlock3 = ConvBlock(128, 128, stride=2)
        self.convBlock4 = ConvBlock(128, 256, stride=1)
        self.convBlock5 = ConvBlock(256, 256, stride=2)
        self.convBlock6 = ConvBlock(256, 512, stride=1)
        self.convBlock7 = ConvBlock(512, 512, stride=2)

        self.avePool = nn.AdaptiveAvgPool2d(1)
        # AdaptiveAvgPool can automatically infer the adaptive parameters

        self.conv2 = nn.Conv2d(512, 1024, kernel_size=1)
        self.relu2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(1024, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)
        x = self.convBlock4(x)
        x = self.convBlock5(x)
        x = self.convBlock6(x)
        x = self.convBlock7(x)

        x = self.avePool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.sigmoid(x)

        return x


# Construct Model
netD = Discriminator()
netG = Generator()
netD.to(device)
netG.to(device)

# Construct Iterator
optimizerG = optim.Adam(netG.parameters())
optimizerD = optim.Adam(netD.parameters())

# Construct Loss Function
lossF = nn.MSELoss().to(device)

# Construct Network Model in VGG Loss
vgg = vgg16(pretrained=True).to(device)
lossNetwork = nn.Sequential(*list(vgg.features)[:31]).eval()

for param in lossNetwork.parameters():
    param.requires_grad = False  # Make VGG stop learning

for epoch in range(epochs):
    netD.train()
    netG.train()
    processBar = tqdm(enumerate(trainData, 1))
    # tqdm is a useful progress bar tool, there it received parameters from enumerator

    for i, (cropImg, sourceImg) in processBar:
        cropImg, sourceImg = cropImg.to(device), sourceImg.to(device)

        fakeImg = netG(cropImg).to(device)

        # Iterate Discriminator Network
        netD.zero_grad()
        realOut = netD(sourceImg).mean()
        fakeOut = netD(fakeImg).mean()
        dLoss = 1 - realOut + fakeOut
        dLoss.backward(retain_graph=True)

        # Iterate Generator Network
        netG.zero_grad()
        gLossSR = lossF(fakeImg, sourceImg)
        gLossGAN = 0.001 * torch.mean(1 - fakeOut)
        gLossVGG = 0.006 * lossF(lossNetwork(fakeImg), lossNetwork(sourceImg))
        gLoss = gLossSR + gLossGAN + gLossVGG
        gLoss.backward()

        optimizerD.step()
        optimizerG.step()

        # Data Visualize
        processBar.set_description(desc='[%d/%d] Loss_D: %.4f LossG: %.4f D(x): %.4f D(G(x)): %.4f' % (
            epoch, epochs, dLoss.item(), gLoss.item(), realOut.item(), fakeOut.item()))

    # Output processed images to directory
    # As to these steps, I could only say"Sorry, I also don't understand them:-("

    with torch.no_grad():
        fig = plt.figure(figsize=(10, 10))
        plt.axis("off")
        # Hidden x, y, z... axis on figures
        fakeImgs = netG(testImgs).detach().cpu()
        plt.imshow(np.transpose(vutils.make_grid(fakeImgs, padding=2, normalize=True), (1, 2, 0)), animated=True)
        plt.savefig('./Img/Result_epoch % 05d.jpg' % epoch, bbox_inches='tight', pad_inches=0)
        print('[INFO] Images saved successfully!')

    # Save model files. Only save state dictionaries, that could save plenty of memories.
    torch.save(netG.state_dict(), 'model/netG_epoch_%d_%d.pth' % (4, epoch))
    torch.save(netD.state_dict(), 'model/netD_epoch_%d_%d.pth' % (4, epoch))
