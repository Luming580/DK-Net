"""DK-Net model definition.

The network uses a dual-branch encoder with local residual blocks and the
large-kernel global perception modules described in the accompanying paper.
"""

import numpy as np
import torch
import torch.nn as nn
import math
from os.path import join as pjoin
from collections import OrderedDict
import torch
import torch.nn.functional as F
from itertools import repeat
import collections.abc

try:
    from timm.layers import trunc_normal_tf_, DropPath, trunc_normal_
    from timm.models.helpers import named_apply
except ImportError:
    def trunc_normal_(tensor, std=.02):
        return nn.init.trunc_normal_(tensor, std=std)

    trunc_normal_tf_ = trunc_normal_

    def drop_path(x, drop_prob=0., training=False):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor

    class DropPath(nn.Module):
        def __init__(self, drop_prob=None):
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x):
            return drop_path(x, self.drop_prob, self.training)

    def named_apply(fn, module, name="", depth_first=True, include_root=True):
        if not depth_first and include_root:
            fn(module, name)
        for child_name, child_module in module.named_children():
            child_name = ".".join((name, child_name)) if name else child_name
            named_apply(fn, child_module, child_name, depth_first=depth_first, include_root=True)
        if depth_first and include_root:
            fn(module, name)
        return module
from torch.nn import init
from functools import partial
import torch
import torch.fft as fft

##################################################################################################################resnet
# 自定义卷积层的作用是在卷积操作之前对权重进行标准化处理，以改善模型的训练稳定性和收敛速度。
class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.bn1 = nn.BatchNorm2d(cmid)
        self.conv1 = nn.Conv2d(cin, cmid, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(cmid)
        self.conv2 = nn.Conv2d(cmid, cmid, 3, 1, 1, bias=False)  # Original code has it on conv1!!
        self.bn3 = nn.BatchNorm2d(cout)
        self.conv3 = nn.Conv2d(cmid, cout, 1, 1, 0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))

        y = residual + y

        return y


class down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.root1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(cin, cout, kernel_size=3, stride=2, bias=False, padding=1)),
            ('gn', nn.BatchNorm2d(cout)),
        ]))

    def forward(self, x):
        x1 = self.root1(x)

        return x1


class down_big(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.root1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(cin, cout, kernel_size=3, stride=2, bias=False, padding=1)),
            ('gn', nn.BatchNorm2d(cout)),
        ]))

    def forward(self, x):
        x1 = self.root1(x)

        return x1


class resnet1(nn.Module):
    def __init__(self, block_units=None, width_factor=1):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width  # width=64

        self.root = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, 96, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.BatchNorm2d(96)),
        ]))

        self.body = nn.Sequential(OrderedDict([  # 这里结构和resnet不太一样
            ('block1/', nn.Sequential(OrderedDict(
                [(f'unit{i:d}/', PreActBottleneck(cin=96, cout=96, cmid=24)) for i in
                 range(3)],  # 1+2 256
            )))]))

    def forward(self, x):
        x = self.root(x)  # 统一过一个7x7
        x = self.body[0](x)  #

        return x


class resnet2(nn.Module):
    def __init__(self, block_units=None, width_factor=1):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width  # width=64
        self.root = down(96, 192)
        self.body = nn.Sequential(OrderedDict([  # 这里结构和resnet不太一样
            ('block2/', nn.Sequential(OrderedDict(
                [(f'unit{i:d}/', PreActBottleneck(cin=192, cout=192, cmid=64)) for i in
                 range(3)],  # 1+3 512
            )))]))

    def forward(self, x):
        x = self.root(x)  # 统一过一个7x7
        x = self.body[0](x)  #

        return x


class resnet3(nn.Module):
    def __init__(self, block_units=None, width_factor=1):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width  # width=64
        self.root = down(192, 384)
        self.body = nn.Sequential(OrderedDict([  # 这里结构和resnet不太一样
            ('block3/', nn.Sequential(OrderedDict(
                [(f'unit{i:d}/', PreActBottleneck(cin=384, cout=384, cmid=96)) for i in
                 range(9)],  # 1+5 1024
            )))]))

    def forward(self, x):
        x = self.root(x)  # 统一过一个7x7
        x = self.body[0](x)  #

        return x


class resnet4(nn.Module):
    def __init__(self, block_units=None, width_factor=1):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width  # width=64
        self.root = down(384, 768)
        self.body = nn.Sequential(OrderedDict([  # 这里结构和resnet不太一样
            ('block4/', nn.Sequential(OrderedDict(
                [(f'unit{i:d}/', PreActBottleneck(cin=768, cout=768, cmid=192)) for i in
                 range(3)],  # 1+2
            )))]))

    def forward(self, x):
        x = self.root(x)  # 统一过一个7x7
        x = self.body[0](x)  #

        return x


##############################################################################################################poolformer


class Mlp(nn.Module):  # 用卷积实现
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features  # 输出层 和 隐藏层 的通道数 默认为 输入通道数
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0, in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class BIGCONV(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, dim, kernel=13, padding=6): #13直接改19 25 49
        super().__init__()
        self.dwconv1 = nn.Conv2d(dim, dim, kernel, padding=padding, groups=dim)
        self.dwconv2 = nn.Conv2d(dim, dim, kernel // 3 + 1, padding=padding, groups=dim, dilation=3)
        self.dwconv3 = nn.Conv2d(dim, dim, kernel // 2 + 1, padding=padding, groups=dim, dilation=2)
        self.conv1x1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)
        self.bn3 = nn.BatchNorm2d(dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        x1 = self.bn1(self.dwconv1(x))
        x2 = self.bn2(self.dwconv2(x))
        x3 = self.bn3(self.dwconv3(x))
        x = self.conv1x1(x1 + x2 + x3)
        x = self.gelu(x)
        return x


class BigFormerBlock(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth,
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, kernel=13, padding=6):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.token_mixer = BIGCONV(dim, kernel=kernel, padding=padding)
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  # mlp_ratio相当于隐藏层的膨胀系数
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale

    def forward(self, x):
        x = x + self.drop_path(self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def big_blocks(dim, index, layers,  # 用于生成 PoolFormer 模型中每个阶段的基本块  #这个也可拆
               ):
    """
    generate PoolFormer blocks for a stage
    return: PoolFormer blocks
    """
    blocks = []
    for block_idx in range(layers[index]):
        blocks.append(BigFormerBlock(
            dim,
        ))
    blocks = nn.Sequential(*blocks)

    return blocks


class bigconv1(nn.Module):  # 参数一起传比较方便
    def __init__(self, ):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size=7, stride=2, padding=3, in_chans=3, embed_dim=96)
        network = []

        stage = big_blocks(dim=96, index=0, layers=[3, 3, 9, 3])
        network.append(stage)

        self.network = nn.ModuleList(network)

    def forward_tokens(self, x):
        for idx, block in enumerate(self.network):
            x = block(x)
        return x

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.forward_tokens(x)

        return x


class bigconv2(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.patch_embed = down_big(96, 192)
        network = []

        stage = big_blocks(dim=192, index=1, layers=[3, 3, 9, 3])
        network.append(stage)

        self.network = nn.ModuleList(network)

    def forward_tokens(self, x):
        for idx, block in enumerate(self.network):
            x = block(x)

        return x

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.forward_tokens(x)
        return x


class bigconv3(nn.Module):
    def __init__(self, ):
        super().__init__()

        network = []
        self.patch_embed = down_big(192, 384)

        stage = big_blocks(dim=384, index=2, layers=[3, 3, 9, 3])
        network.append(stage)

        self.network = nn.ModuleList(network)

    def forward_tokens(self, x):
        for idx, block in enumerate(self.network):
            x = block(x)

        return x

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.forward_tokens(x)
        return x


class bigconv4(nn.Module):
    def __init__(self, ):
        super().__init__()

        network = []
        self.patch_embed = down_big(384, 768)

        stage = big_blocks(dim=768, index=3, layers=[3, 3, 9, 3])
        network.append(stage)

        self.network = nn.ModuleList(network)

    def forward_tokens(self, x):
        for idx, block in enumerate(self.network):
            x = block(x)

        return x

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.forward_tokens(x)
        return x


#######################################################################################################################
class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv3 = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, xs, xl):
        xs = self.conv1(xs)
        xl = self.conv2(xl)
        attn = torch.cat([xs, xl], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = xs * sig[:, 0, :, :].unsqueeze(1) + xl * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv3(attn)
        return attn


class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.gap(x)  # 在空间方向执行全局平均池化: (B,C,H,W)-->(B,C,1,1)
        y = y.squeeze(-1).permute(0, 2, 1)  # 将通道描述符去掉一维,便于在通道上执行卷积操作:(B,C,1,1)-->(B,C,1)-->(B,1,C)
        y = self.conv(y)  # 在通道维度上执行1D卷积操作,建模局部通道之间的相关性: (B,1,C)-->(B,1,C)
        y = self.sigmoid(y)  # 生成权重表示: (B,1,C)
        y = y.permute(0, 2, 1).unsqueeze(-1)  # 重塑shape: (B,1,C)-->(B,C,1)-->(B,C,1,1)
        return x * y.expand_as(x)  # 权重对输入的通道进行重新加权: (B,C,H,W) * (B,C,1,1) = (B,C,H,W)


##########################################################################################################shenchaoblock
class Mixblock1(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.cnn = resnet1()
        self.transfomer = bigconv1()
        # self.conv1x1 = nn.Conv2d(192, 192, 1, 1, 0)
        self.lsk = LSKblock(96)

    def forward(self, x):
        x1 = self.cnn(x)
        x2 = self.transfomer(x)
        x = self.lsk(x1, x2)
        # x = torch.cat([x1, x2], dim=1)
        # x = self.conv1x1(x)
        return x


class Mixblock2(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.cnn = resnet2()
        self.transfomer = bigconv2()
        # self.conv1x1 = nn.Conv2d(384, 384, 1, 1, 0)
        self.lsk = LSKblock(192)

    def forward(self, x):
        x1 = self.cnn(x)
        x2 = self.transfomer(x)
        x = self.lsk(x1, x2)
        # x = torch.cat([x1, x2], dim=1)
        # x = self.conv1x1(x)
        return x


class Mixblock3(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.cnn = resnet3()
        self.transfomer = bigconv3()
        self.lsk = LSKblock(384)
        # self.conv1x1 = nn.Conv2d(768, 768, 1, 1, 0)

    def forward(self, x):
        x1 = self.cnn(x)
        x2 = self.transfomer(x)
        x = self.lsk(x1, x2)
        # x = torch.cat([x1, x2], dim=1)
        # x = self.conv1x1(x)
        return x


class Mixblock4(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.cnn = resnet4()
        self.transfomer = bigconv4()
        # self.conv1x1 = nn.Conv2d(1536, 1536, 1, 1, 0)
        self.lsk = LSKblock(768)

    def forward(self, x):
        x1 = self.cnn(x)
        x2 = self.transfomer(x)
        x = self.lsk(x1, x2)
        # x = torch.cat([x1, x2], dim=1)
        # x = self.conv1x1(x)
        return x


##############################################################################################################upsampling


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True, ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,  # 有BN无需偏执

        )

        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)
        '''最后调用是因为它用于将传入的模块按顺序添加到 nn.Sequential 类中。
           在 nn.Sequential 的初始化方法中，传入的参数会按顺序成为该序列的子模块'''


class DecoderBlock(nn.Module):  # 一个上采样块
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
            num_classes=9,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(  # 又提取了一次信息
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        # bn,relu的写法
        # self.bn = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU(inplace=True)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv2d256 = nn.Conv2d(256, 1, 1, 1, 0)
        self.up8 = nn.UpsamplingBilinear2d(scale_factor=8)

        self.conv2d128 = nn.Conv2d(128, 1, 1, 1, 0)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)

        self.conv2d64 = nn.Conv2d(64, 1, 1, 1, 0)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv2d32 = nn.Conv2d(32, num_classes, 1, 1, 0)

    def forward(self, x, i, skip=None):

        if skip is not None:
            x = torch.cat([x, skip], dim=1)  # 若有跳跃连接则在c上，先连接，然后过两个卷积
            x = self.conv1(x)
            x = self.conv2(x)  #
            if i == 0:
                y = self.conv2d256(x)
                y = self.up8(y)
            if i == 1:
                y = self.conv2d128(x)
                y = self.up4(y)
            if i == 2:
                y = self.conv2d64(x)
                y = (self.up2(y))
        else:
            x = self.up(x)
            x = self.conv1(x)
            x = self.conv2(x)

            y = (self.conv2d32(x))

        return x, y


class DecoderCup(nn.Module):  # 解码器
    def __init__(self, config=None, num_classes=9):
        super().__init__()
        # self.config = config
        head_channels = 512

        skip_channels = [384, 192, 96, 0]  # 加注意力

        decoder_channels = (256, 128, 64, 32)  # 1/8 1/4/ 1/2 1

        in_channels = [head_channels] + list(decoder_channels[:-1])  # 上采样输入为512、256、128、64、
        out_channels = decoder_channels  # 上采样输出为256，128，64，16

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch, num_classes=num_classes) for in_ch, out_ch, sk_ch, in
            zip(in_channels, out_channels, skip_channels, )
        ]

        self.blocks = nn.ModuleList(blocks)
        self.lgag1 = LGAG(512, 384, 512, 3)
        self.lgag2 = LGAG(256, 192, 256, 3)
        self.lgag3 = LGAG(128, 96, 128, 3)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, features=None):  # 后面传了x, features
        x = x.contiguous()
        features_deep = []
        for i, decoder_block in enumerate(self.blocks):  # 四个循环，最后一次循环没有跳跃连接，只有上采样
            if features is not None and i == 0:  # 若有features传入,则进行跳跃连接
                skip = features[i]
                x = self.up(x)
                skip = self.lgag1(x, skip) + skip
            if features is not None and i == 1:  # 若有features传入,则进行跳跃连接
                skip = features[i]
                x = self.up(x)
                skip = self.lgag2(x, skip) + skip
            if features is not None and i == 2:  # 若有features传入,则进行跳跃连接
                skip = features[i]
                x = self.up(x)
                skip = self.lgag3(x, skip) + skip
            if features is not None and i == 3:
                skip = None

            x, y = decoder_block(x, i, skip=skip)
            features_deep.append(y)
            features_deep = features_deep[::-1]  # 第0个是x1，第1个是x2，第2个是x4，第3个是x8
        return features_deep
        # return x, x1, x2, x3


################################################################################################################各种注意力
class LGAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):  # F_g=解码器 F_l=跳跃连接 F_int=同一映射空间
        super(LGAG, self).__init__()

        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        # self.psi2 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.BatchNorm2d(F_l),
        #     nn.Sigmoid()
        # )
        self.activation = nn.ReLU(True)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return x * psi


def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels is None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out)

    #   Spatial attention block (SAB)


class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


###################################################################################################################全流程


class DKNet(nn.Module):
    """Dual-Kernel Network for 2-D medical image segmentation.

    The implementation accepts one-channel images directly and internally
    replicates them to three channels. Spatial dimensions must be divisible by
    16 because of the four encoder downsampling stages.
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.a = Mixblock1()
        self.b = Mixblock2()
        self.c = Mixblock3()
        self.d = Mixblock4()
        # self.segmentation = SegmentationHead_small()
        self.decoder = DecoderCup(num_classes=num_classes)
        self.conv_more = Conv2dReLU(  # 升至512维的卷积
            in_channels=768,
            out_channels=512,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.conv_more2 = Conv2dReLU(  # 升至512维的卷积
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected [batch, channels, height, width], got {tuple(x.shape)}.")
        if x.shape[1] not in (1, 3):
            raise ValueError(f"DKNet expects one or three input channels, got {x.shape[1]}.")
        if x.shape[-2] % 16 or x.shape[-1] % 16:
            raise ValueError(
                "Input height and width must be divisible by 16, "
                f"got {tuple(x.shape[-2:])}."
            )
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        features = []

        x = self.a(x)
        features.append(x)

        x = self.b(x)
        features.append(x)

        x = self.c(x)
        features.append(x)

        features = features[::-1]
        x = self.d(x)

        x = self.conv_more(x)
        x = self.conv_more2(x)

        features_deep = self.decoder(x, features)

        return features_deep[0]


