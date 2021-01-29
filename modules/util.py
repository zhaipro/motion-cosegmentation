import imageio
import numpy as np

from torch import nn
import torch.nn.functional as F
import torch

from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d


def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['shift']

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out

def kp2gaussian_new(mean, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out

def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding, use_att=False):
        super(ResBlock2d, self).__init__()
        self.use_att = use_att
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding, bias=not use_att)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding, bias=not use_att)
        self.norm1 = BatchNorm2d(in_features, affine=True)
        self.norm2 = BatchNorm2d(in_features, affine=True)
        if self.use_att:
            ratio = 8
            self._channel_att = nn.Sequential(nn.Conv2d(in_features, in_features//ratio, kernel_size=1, bias=False), nn.ReLU(inplace=True),
                                            nn.Conv2d(in_features//ratio, in_features, kernel_size=1, bias=False))
            self._space_att = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False), nn.Sigmoid())


    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        if self.use_att:
            avgout = self._channel_att(nn.AdaptiveAvgPool2d(1)(out))
            maxout = self._channel_att(nn.AdaptiveMaxPool2d(1)(out))
            out = F.sigmoid(avgout+maxout) * out
            avgout = torch.mean(out, dim=1, keepdim=True)
            maxout, _ = torch.max(out, dim=1, keepdim=True)
            att_x = torch.cat([avgout, maxout], dim=1)
            out = self._space_att(att_x)*out
        out += x
        return out


class UpBlock2d(nn.Module):
    """
    Simple block for processing video (decoder).
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1, use_att=False, use_inter=True):
        super(UpBlock2d, self).__init__()
        self.use_att = use_att
        self.use_inter = use_inter
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        if use_att:
            self.conv1 = nn.Sequential(nn.Conv2d(out_features, out_features, kernel_size=kernel_size, padding=padding, groups=groups, bias=False),
                                    BatchNorm2d(out_features, affine=True), nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(nn.Conv2d(out_features, out_features, kernel_size=kernel_size, padding=padding, groups=groups, bias=False),
                                    BatchNorm2d(out_features, affine=True), nn.ReLU(inplace=True))
            ratio = 8
            self._channel_att = nn.Sequential(nn.Conv2d(out_features, out_features//ratio, kernel_size=1, bias=False), nn.ReLU(inplace=True),
                                    nn.Conv2d(out_features//ratio, out_features, kernel_size=1, bias=False))
            self._space_att = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False), nn.Sigmoid())
        if not use_inter:
            self.up = nn.Sequential(nn.ConvTranspose2d(in_features, in_features, kernel_size=7, padding=3, output_padding=1, stride=2, bias=False),
                                BatchNorm2d(in_features, affine=True), nn.ReLU(inplace=True))

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        if not self.use_inter:  # 在插值法的基础上加一些反卷积的细节，两者关注的尺度不一样
            # print('='*15)
            # tmp = self.up(x)
            # print('x.shape: ', x.shape)
            # print('out.shape: ', out.shape)
            # print('tmp.shape: ', tmp.shape)
            out += self.up(x)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        if self.use_att:
            residual = out
            x = self.conv1(out)
            x = self.conv2(x)
            avgout = self._channel_att(nn.AdaptiveAvgPool2d(1)(x))
            maxout = self._channel_att(nn.AdaptiveMaxPool2d(1)(x))
            x = F.sigmoid(avgout + maxout) * x
            avgout = torch.mean(x, dim=1, keepdim=True)
            maxout, _ = torch.max(x, dim=1, keepdim=True)
            att_x = torch.cat([avgout, maxout], dim=1)
            x = self._space_att(att_x) * x
            x += residual
            out = F.relu(x, inplace=True)
        return out


class DownBlock2d(nn.Module):
    """
    Simple block for processinGg video (encoder).
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1, use_att=False, use_inter=True):
        super(DownBlock2d, self).__init__()
        self.use_att = use_att
        self.use_inter = use_inter
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                            padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))
        if use_att:
            self.conv1 = nn.Sequential(nn.Conv2d(out_features, out_features, kernel_size=kernel_size, padding=padding, groups=groups, bias=False),
                                    BatchNorm2d(out_features, affine=True), nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(nn.Conv2d(out_features, out_features, kernel_size=kernel_size, padding=padding, groups=groups, bias=False),
                                    BatchNorm2d(out_features, affine=True), nn.ReLU(inplace=True))
            ratio = 8
            self._channel_att = nn.Sequential(nn.Conv2d(out_features, out_features//ratio, kernel_size=1, bias=False), nn.ReLU(inplace=True),
                                            nn.Conv2d(out_features//ratio, out_features, kernel_size=1, bias=False))
            self._space_att = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False), nn.Sigmoid())
        if not use_inter:
            self.down = nn.Sequential(nn.Conv2d(out_features, out_features, kernel_size=7, padding=3, stride=2, bias=False),
                                    BatchNorm2d(out_features, affine=True), nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        if self.use_att:
            # Channel Att
            residual = out
            x = self.conv1(out)
            x = self.conv2(x)
            avgout = self._channel_att(nn.AdaptiveAvgPool2d(1)(x))
            maxout = self._channel_att(nn.AdaptiveMaxPool2d(1)(x))
            x = F.sigmoid(avgout + maxout) * x
            # Space Att
            avgout = torch.mean(x, dim=1, keepdim=True)
            maxout, _ = torch.max(x, dim=1, keepdim=True)
            att_x = torch.cat([avgout, maxout], dim=1)
            x = self._space_att(att_x) * x
            x += residual
            out = F.relu(x, inplace=True)
        res = self.pool(out)
        if not self.use_inter:  # 在池化的基础上加一些卷积细节，两者关注的尺度不一样
            res += self.down(out)
        return res


class SameBlock2d(nn.Module):
    """
    Simple block with group convolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256, use_att=False, use_inter=True):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1, use_att=use_att, use_inter=use_inter))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256, use_att=False, use_inter=True):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1, use_att=use_att, use_inter=use_inter))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        return out


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256, use_att=False, use_inter=True):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features, use_att, use_inter)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features, use_att, use_inter)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AntiAliasInterpolation2d(nn.Module):
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = F.interpolate(out, scale_factor=(self.scale, self.scale))

        return out


def save_img(img, save_path):
    img = img[0].data.cpu().numpy().transpose([1,2,0])
    img = (img*255).astype(np.uint8)
    imageio.imwrite(save_path, img)
