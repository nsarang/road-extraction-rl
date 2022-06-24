import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


upsample = lambda x, scale: F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=True)


model_urls = {
    "res2net50_v1b_26w_4s": "https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth",
    "res2net101_v1b_26w_4s": "https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth",
}


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        baseWidth=26,
        scale=4,
        stype="normal",
        BatchNorm=None,
    ):
        """Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == "stage" and stride > 1:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(
                nn.Conv2d(width, width, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
            )
            bns.append(BatchNorm(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == "stage":
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == "normal":
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.stride == 1 and self.scale != 1 and self.stype == "stage":
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == "stage":
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):
    def __init__(
        self, block, layers, output_stride=8, baseWidth=26, scale=4, BatchNorm=nn.BatchNorm2d, num_classes=1000
    ):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm
        )
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm
        )
        self.layer4 = self._make_MG_unit(
            block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                dilation,
                downsample=downsample,
                BatchNorm=BatchNorm,
                stype="stage",
                baseWidth=self.baseWidth,
                scale=self.scale,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    dilation=dilation,
                    BatchNorm=BatchNorm,
                    baseWidth=self.baseWidth,
                    scale=self.scale,
                )
            )

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                dilation=blocks[0] * dilation,
                downsample=downsample,
                BatchNorm=BatchNorm,
                stype="stage",
                baseWidth=self.baseWidth,
                scale=self.scale,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    dilation=blocks[i] * dilation,
                    BatchNorm=BatchNorm,
                    baseWidth=self.baseWidth,
                    scale=self.scale,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def res2net50_v1b(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b model.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["res2net50_v1b_26w_4s"]))
    return model


def res2net101_v1b(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["res2net101_v1b_26w_4s"]))
    return model


def res2net50_v1b_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["res2net50_v1b_26w_4s"]))
    return model


def res2net101_v1b_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["res2net101_v1b_26w_4s"]))
    return model


def res2net152_v1b_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 8, 36, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["res2net152_v1b_26w_4s"]))
    return model


class ConvReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_sz, stride=1, relu=True, pd=True, bn=False):
        super(ConvReLU, self).__init__()
        padding = int((kernel_sz - 1) / 2) if pd else 0  # same spatial size by default
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_sz, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch) if bn else None  # eps=0.001, momentum=0, affine=True
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, ft_in, ft_out, out_ch):
        super(DecoderBlock, self).__init__()

        self.ft_conv = nn.Sequential(ConvReLU(ft_in, ft_out, 1))
        self.conv = nn.Sequential(
            ConvReLU(out_ch + ft_out, (out_ch + ft_out) // 2, 3),
            nn.Conv2d((out_ch + ft_out) // 2, out_ch, 3, padding=1),
        )

    def forward(self, ft_cur, ft_pre):
        ft_cur = self.ft_conv(ft_cur)
        x = torch.cat((ft_cur, ft_pre), dim=1)
        x = self.conv(x)
        return x


class Hourglass(nn.Module):
    def __init__(self, input_ch, output_ch, ch=[32, 32, 32, 32]):
        super(Hourglass, self).__init__()
        ch = [input_ch] + ch
        self.encoder_1 = nn.Sequential(
            ConvReLU(ch[0], (ch[0] + ch[1]) // 2, 3), ConvReLU((ch[0] + ch[1]) // 2, ch[1], 3)
        )
        self.encoder_2 = nn.Sequential(
            ConvReLU(ch[1], (ch[1] + ch[2]) // 2, 3), ConvReLU((ch[1] + ch[2]) // 2, ch[2], 3)
        )
        self.encoder_3 = nn.Sequential(
            ConvReLU(ch[2], (ch[2] + ch[3]) // 2, 3), ConvReLU((ch[2] + ch[3]) // 2, ch[3], 3)
        )
        self.encoder_4 = nn.Sequential(
            ConvReLU(ch[3], (ch[3] + ch[4]) // 2, 3), ConvReLU((ch[3] + ch[4]) // 2, ch[4], 3)
        )
        self.encoder_5 = nn.Sequential(ConvReLU(ch[4], ch[4], 3), ConvReLU(ch[4], ch[4], 3))
        self.decoder_4 = nn.Sequential(
            ConvReLU(ch[4], (ch[3] + ch[4]) // 2, 3), ConvReLU((ch[3] + ch[4]) // 2, ch[3], 3)
        )
        self.decoder_3 = nn.Sequential(
            ConvReLU(ch[3], (ch[2] + ch[3]) // 2, 3), ConvReLU((ch[2] + ch[3]) // 2, ch[2], 3)
        )
        self.decoder_2 = nn.Sequential(
            ConvReLU(ch[2], (ch[1] + ch[2]) // 2, 3), ConvReLU((ch[1] + ch[2]) // 2, ch[1], 3)
        )
        self.decoder_1 = nn.Sequential(
            ConvReLU(ch[1], (ch[0] + ch[1]) // 2, 3), ConvReLU((ch[0] + ch[1]) // 2, output_ch, 3)
        )
        self.maxpool = nn.MaxPool2d(2, 2)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        encoder_1 = self.encoder_1(x)
        encoder_1_pool = self.maxpool(encoder_1)
        encoder_2 = self.encoder_2(encoder_1_pool)
        encoder_2_pool = self.maxpool(encoder_2)
        encoder_3 = self.encoder_3(encoder_2_pool)
        encoder_3_pool = self.maxpool(encoder_3)
        encoder_4 = self.encoder_4(encoder_3_pool)
        encoder_4_pool = self.maxpool(encoder_4)
        encoder_5 = self.encoder_5(encoder_4_pool)
        decoder_5_up = upsample(encoder_5, 2) + encoder_4
        decoder_4 = self.decoder_4(decoder_5_up)
        decoder_4_up = upsample(decoder_4, 2) + encoder_3
        decoder_3 = self.decoder_3(decoder_4_up)
        decoder_3_up = upsample(decoder_3, 2) + encoder_2
        decoder_2 = self.decoder_2(decoder_3_up)
        decoder_2_up = upsample(decoder_2, 2) + encoder_1
        decoder_1 = self.decoder_1(decoder_2_up)
        return decoder_1


class RPNet(nn.Module):
    def __init__(self, num_targets=4):
        super(RPNet, self).__init__()

        self.num_targets = num_targets

        self.conv_2_side = ConvReLU(256, 128, 3, 1, bn=True)
        self.conv_3_side = ConvReLU(512, 128, 3, 1, bn=True)
        self.conv_4_side = ConvReLU(1024, 128, 3, 1, bn=True)
        self.conv_5_side = ConvReLU(2048, 128, 3, 1, bn=True)
        self.conv_fuse = ConvReLU(512, 128, 3, 1, bn=True)
        self.avgpool4 = nn.AvgPool2d(4, 4)

        self.road_seg = nn.Sequential(ConvReLU(128, 64, 3, 1, bn=True), ConvReLU(64, 64, 1, 1, bn=True))
        self.conv_road_final = nn.Conv2d(64, 1, 1, 1, 0)

        self.junc_seg = nn.Sequential(ConvReLU(128, 64, 3, 1, bn=True), ConvReLU(64, 64, 1, 1, bn=True))
        self.conv_junc_final = nn.Conv2d(64, 1, 1, 1, 0)
        self.fuse_module = Hourglass(128 + 64 + 64 + 32 * (self.num_targets - 1) + 1, 32, [128, 128, 128, 128])  # 353

        self.ft_chs = [1024, 512, 256, 64]
        self.decoders = nn.ModuleList(
            [
                DecoderBlock(self.ft_chs[0], 32, 32),
                DecoderBlock(self.ft_chs[1], 32, 32),
                DecoderBlock(self.ft_chs[2], 32, 32),
                DecoderBlock(self.ft_chs[3], 32, 32),
            ]
        )
        self.next_step_final = nn.Conv2d(32, 1, 1, 1, 0)
        self.conv_final = nn.Conv2d(32, 1, 3, 1, 1)

        self.init_weights()
        ## first init_weights for added parts, then init res2net
        res2net = res2net50_v1b_26w_4s(pretrained=True)
        self.stage_1 = nn.Sequential(res2net.conv1, res2net.relu)
        self.maxpool = res2net.maxpool
        self.stage_2 = res2net.layer1
        self.stage_3 = res2net.layer2
        self.stage_4 = res2net.layer3
        self.stage_5 = res2net.layer4

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, aerial_image, walked_path, NUM_TARGETS=None, test=False):
        stage_1 = self.stage_1(aerial_image)  # 1/2
        stage_1_down = self.maxpool(stage_1)  # 1/4

        stage_2 = self.stage_2(stage_1_down)  # 1/4
        stage_2_side = self.conv_2_side(stage_2)  # 1/4

        stage_3 = self.stage_3(stage_2)  # 1/8
        stage_3_side = self.conv_3_side(stage_3)  # 1/8
        stage_3_side = upsample(stage_3_side, 2)  # 1/4

        stage_4 = self.stage_4(stage_3)  # 1/8
        stage_4_side = self.conv_4_side(stage_4)  # 1/8
        stage_4_side = upsample(stage_4_side, 2)  # 1/4

        stage_5 = self.stage_5(stage_4)  # 1/8
        stage_5_side = self.conv_5_side(stage_5)  # 1/8
        stage_5_side = upsample(stage_5_side, 2)  # 1/4

        stage_fuse = [stage_2_side, stage_3_side, stage_4_side, stage_5_side]
        stage_fuse = torch.cat(stage_fuse, dim=1)
        stage_fuse = self.conv_fuse(stage_fuse)

        road_fts = self.road_seg(stage_fuse)
        road_final = self.conv_road_final(road_fts)

        junc_fts = self.junc_seg(stage_fuse)
        junc_final = self.conv_junc_final(junc_fts)

        if test:
            return {"road": upsample(road_final, 4), "junc": upsample(junc_final, 4)}

        next_points_placeholder = torch.zeros(
            (stage_fuse.shape[0], 32 * (self.num_targets - 1), stage_fuse.shape[2], stage_fuse.shape[3]),
            device=stage_fuse.device,
        )
        stage_fuse = torch.cat([stage_fuse, road_fts, junc_fts, walked_path, next_points_placeholder], dim=1)
        if self.training:
            stage_fuse_list = [stage_fuse]

        anchor_fts = None
        next_points = []
        next_points_lowrs = []  # low resolution
        for i in range(NUM_TARGETS if NUM_TARGETS is not None else self.num_targets):
            if self.training:
                next_step = self.fuse_module(stage_fuse_list[i])
            else:
                next_step = self.fuse_module(stage_fuse)
            next_points_lowrs.append(upsample(self.next_step_final(next_step), 4))

            decoded_ft_4 = self.decoders[0](upsample(stage_4, 2), next_step)
            decoded_ft_3 = self.decoders[1](upsample(stage_3, 2), decoded_ft_4)
            decoded_ft_2 = self.decoders[2](upsample(stage_2, 2), upsample(decoded_ft_3, 2))
            decoded_ft_1 = self.decoders[3](upsample(stage_1, 2), upsample(decoded_ft_2, 2))

            ch_idx = -(self.num_targets - i - 1) * 32
            if i < self.num_targets - 1:
                if anchor_fts is None:
                    anchor_fts = self.avgpool4(decoded_ft_1)
                else:
                    anchor_fts += self.avgpool4(decoded_ft_1)
                if self.training:
                    stage_fuse_list.append(stage_fuse_list[i].clone())
                    stage_fuse_list[i + 1][:, ch_idx : ch_idx + 32 if ch_idx + 32 != 0 else None, :, :] = anchor_fts
                else:
                    stage_fuse[:, ch_idx : ch_idx + 32 if ch_idx + 32 != 0 else None, :, :] = anchor_fts
            decoded_ft_1 = self.conv_final(decoded_ft_1)
            next_points.append(decoded_ft_1)
        next_points = torch.cat(next_points, dim=1)  # torch.Size([4, 4, 256, 256])
        next_points_lowrs = torch.cat(next_points_lowrs, dim=1)
        return {"road": road_final, "junc": junc_final, "anchor": next_points, "anchor_lowrs": next_points_lowrs}
