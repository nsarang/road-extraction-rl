#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import logging
import pdb

import torch
import torch.nn as nn

from urllib.request import urlretrieve
from inplace_abn import InPlaceABNSync
from .extensions.switchablenorms.switchable_norm import SwitchNorm2d


class ModuleHelper(object):
    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        if bn_type == "torchbn":
            return nn.Sequential(nn.BatchNorm2d(num_features, **kwargs), nn.ReLU())
        elif bn_type == "torchsyncbn":
            return nn.Sequential(nn.SyncBatchNorm(num_features, **kwargs), nn.ReLU())
        elif bn_type == "sn":
            return nn.Sequential(SwitchNorm2d(num_features, **kwargs), nn.ReLU())
        elif bn_type == "gn":
            return nn.Sequential(nn.GroupNorm(num_groups=8, num_channels=num_features, **kwargs), nn.ReLU())
        elif bn_type == "fn":
            raise RuntimeError("Not support Filter-Response-Normalization: {}.".format(bn_type))
        elif bn_type == "inplace_abn":
            return InPlaceABNSync(num_features, **kwargs)
        else:
            raise RuntimeError("Not support BN type: {}.".format(bn_type))

    @staticmethod
    def BatchNorm2d(bn_type="torch", ret_cls=False):
        if bn_type == "torchbn":
            return nn.BatchNorm2d

        elif bn_type == "torchsyncbn":
            return nn.SyncBatchNorm
        elif bn_type == "syncbn":
            return BatchNorm2d
        elif bn_type == "sn":
            return SwitchNorm2d
        elif bn_type == "gn":
            return functools.partial(nn.GroupNorm, num_groups=32)
        elif bn_type == "inplace_abn":
            if ret_cls:
                return InPlaceABNSync
            return functools.partial(InPlaceABNSync, activation="identity")
        else:
            raise RuntimeError("Not support BN type: {}.".format(bn_type))

    @staticmethod
    def load_model(model, pretrained=None, all_match=True, network="resnet101"):
        if pretrained is None:
            return model

        if all_match:
            pretrained_dict = torch.load(pretrained)
            model_dict = model.state_dict()
            load_dict = dict()
            for k, v in pretrained_dict.items():
                if "resinit.{}".format(k) in model_dict:
                    load_dict["resinit.{}".format(k)] = v
                else:
                    load_dict[k] = v
            model.load_state_dict(load_dict)

        else:
            pretrained_dict = torch.load(pretrained)

            # settings for "wide_resnet38"  or network == "resnet152"
            if network == "wide_resnet":
                pretrained_dict = pretrained_dict["state_dict"]

            model_dict = model.state_dict()

            if network == "hrnet_plus":
                # pretrained_dict['conv1_full_res.weight'] = pretrained_dict['conv1.weight']
                # pretrained_dict['conv2_full_res.weight'] = pretrained_dict['conv2.weight']
                load_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}

            elif network == "hrnet" or network == "xception" or network == "resnest":
                load_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
                logging.info("Missing keys: {}".format(list(set(model_dict) - set(load_dict))))

            elif network == "dcnet" or network == "resnext":
                load_dict = dict()
                for k, v in pretrained_dict.items():
                    if "resinit.{}".format(k) in model_dict:
                        load_dict["resinit.{}".format(k)] = v
                    else:
                        if k in model_dict:
                            load_dict[k] = v
                        else:
                            pass

            elif network == "wide_resnet":
                load_dict = {
                    ".".join(k.split(".")[1:]): v
                    for k, v in pretrained_dict.items()
                    if ".".join(k.split(".")[1:]) in model_dict
                }
            else:
                load_dict = {
                    ".".join(k.split(".")[1:]): v
                    for k, v in pretrained_dict.items()
                    if ".".join(k.split(".")[1:]) in model_dict
                }

            # used to debug
            if int(os.environ.get("debug_load_model", 0)):
                logging.info("Matched Keys List:")
                for key in load_dict.keys():
                    logging.info("{}".format(key))
            model_dict.update(load_dict)
            model.load_state_dict(model_dict)

        return model

    @staticmethod
    def load_url(url, map_location=None):
        model_dir = os.path.join("~", ".PyTorchCV", "models")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        filename = url.split("/")[-1]
        cached_file = os.path.join(model_dir, filename)
        if not os.path.exists(cached_file):
            logging.info('Downloading: "{}" to {}\n'.format(url, cached_file))
            urlretrieve(url, cached_file)

        logging.info("Loading pretrained model:{}".format(cached_file))
        return torch.load(cached_file, map_location=map_location)

    @staticmethod
    def constant_init(module, val, bias=0):
        nn.init.constant_(module.weight, val)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def xavier_init(module, gain=1, bias=0, distribution="normal"):
        assert distribution in ["uniform", "normal"]
        if distribution == "uniform":
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def normal_init(module, mean=0, std=1, bias=0):
        nn.init.normal_(module.weight, mean, std)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def uniform_init(module, a=0, b=1, bias=0):
        nn.init.uniform_(module.weight, a, b)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def kaiming_init(module, mode="fan_in", nonlinearity="leaky_relu", bias=0, distribution="normal"):
        assert distribution in ["uniform", "normal"]
        if distribution == "uniform":
            nn.init.kaiming_uniform_(module.weight, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=nonlinearity)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)
