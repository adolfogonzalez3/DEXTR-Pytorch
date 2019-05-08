
# PyTorch includes
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.nn.functional import upsample

from networks.deeplab_resnet import ClassifierModule, PSPModule
from models.shufflenetV2 import shufflenetv2_x1_0

def create_densenet(nInputChannels):
    densenet = models.densenet121(pretrained=False)
    densenet.features[0] = nn.Conv2d(nInputChannels, 64, kernel_size=3,
                                     stride=2)
    classifier = PSPModule(1024)
    return nn.Sequential(densenet.features[:-3], classifier)

def create_squeezenet(nInputChannels):
    squeezenet = models.squeezenet1_1(pretrained=False)
    squeezenet.features[0] = nn.Conv2d(nInputChannels, 64, kernel_size=3,
                                     stride=2)
    classifier = PSPModule(512)
    return nn.Sequential(squeezenet.features, classifier)

def create_shufflenet(nInputChannels):
    shufflenet = shufflenetv2_x1_0(pretrained=False)
    first_conv = nn.Conv2d(nInputChannels, 24, kernel_size=3,
                           stride=2, padding=1, bias=False)
    classifier = PSPModule(1024)
    net = nn.Sequential(first_conv,
                        shufflenet.maxpool,
                        shufflenet.stage2, shufflenet.stage3,
                        shufflenet.stage4, shufflenet.conv5,
                        classifier)
    return net
