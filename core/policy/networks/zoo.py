import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
import torchvision.models as models
from .seg import RoadSegExtendedHead, SECNN
from .litehrnet import LiteHRNet
from .utils import StatelessWrapper

from core.exps.rss.roadres import LitRoadS
from efficientnet_pytorch import EfficientNet
from configs.arch import HRNET_CFG
from copy import deepcopy


def ResNet34(obs_channels, feature_vector_size=1000, pretrained=False):
    net = models.resnet34(pretrained=pretrained)
    with torch.no_grad():
        # first layer
        if obs_channels != 3:
            nConv = nn.Conv2d(
                obs_channels,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
            nConv.weight[:, :-1, ...] = net.conv1.weight[...]
            net.conv1 = nConv
        # last layer
        if feature_vector_size != 1000:
            net.fc = nn.Linear(
                in_features=net.fc.in_features,
                out_features=feature_vector_size,
                bias=True,
            )
    net = StatelessWrapper(net)
    return net


def ResNestSeg(feature_vector_size, pretrained):
    backbone = LitRoadS.load_from_checkpoint(pretrained)
    backbone.freeze()
    model = RoadSegExtendedHead(backbone, feature_vector_size)
    return model


def SECNN_1INP(feature_vector_size=None, pretrained=None, obs_channels=None):
    if pretrained:
        model = torch.load(pretrained)
    else:
        assert obs_channels and feature_vector_size
        model = SECNN(obs_channels, feature_vector_size)
    return model


def EfficientNetEx(name, feature_vector_size, obs_channels=3, pretrained=None):
    if pretrained:
        model = EfficientNet.from_pretrained(name, in_channels=obs_channels, num_classes=feature_vector_size)
    else:
        model = EfficientNet.from_name(
            name, in_channels=obs_channels, num_classes=feature_vector_size  # override_params
        )
    model = StatelessWrapper(model)
    return model


def LiteHRNetEx(name, obs_channels=3, pretrained=None):
    name = name.lower()
    assert name in HRNET_CFG
    cfg = deepcopy(HRNET_CFG[name])
    cfg.update(in_channels=obs_channels)
    model = LiteHRNet(**cfg)
    if pretrained:
        model.init_weights(pretrained)
    return model


def TIMM_ZOO(name, obs_channels=3, feature_vector_size=1000, pretrained=False, **kwargs):
    return timm.create_model(
        name, in_chans=obs_channels, num_classes=feature_vector_size, pretrained=pretrained, **kwargs
    )
