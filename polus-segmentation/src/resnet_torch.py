# Code sourced from https://github.com/MouseLand/cellpose/tree/master/cellpose

import torch
import torch.nn as nn
import torch.nn.functional as F

sz = 3


def convbatchrelu(in_channels, out_channels, sz):
    """ Performs 2d convolution,normalisation and relu activation serially
    Args:
       in_channels(int): Number of channels in input image
       out_channels(int): Number of channels in output image
       sz(int): Kernel size
    Returns:
       _ : Module that performs 2d convolution,normalisation and relu activation

    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, sz, padding=sz // 2),
        nn.BatchNorm2d(out_channels, eps=1e-5),
        nn.ReLU(inplace=True),
    )


def batchconv(in_channels, out_channels, sz):
    """ Performs normalisation ,relu activation and 2d convolution serially
    Args:
        in_channels(int): Number of channels in input image
        out_channels(int): Number of channels in output image
        sz(int): Kernel size
    Returns:
        _ :Module that performs normalisation ,relu activation and 2d convolution

    """
    return nn.Sequential(
        nn.BatchNorm2d(in_channels, eps=1e-5),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, sz, padding=sz // 2),
    )


def batchconv0(in_channels, out_channels, sz):
    """ Performs normalisation and 2d convolution serially
    Args:
        in_channels(int): Number of channels in input image
        out_channels(int): Number of channels in output image
        sz(int): Kernel size
    Returns:
        _ : Module that performs normalisation and 2d convolution

    """
    return nn.Sequential(
        nn.BatchNorm2d(in_channels, eps=1e-5),
        nn.Conv2d(in_channels, out_channels, sz, padding=sz // 2),
    )


class resdown(nn.Module):
    """  Class for residual blocks
    Args:
        in_channels(int): Number of channels in input image
        out_channels(int): Number of channels in output image
        sz(int): Kernel size

    """

    def __init__(self, in_channels, out_channels, sz):
        super().__init__()
        self.conv = nn.Sequential()
        self.proj = batchconv0(in_channels, out_channels, 1)
        for t in range(4):
            if t == 0:
                self.conv.add_module('conv_%d' % t, batchconv(in_channels, out_channels, sz))
            else:
                self.conv.add_module('conv_%d' % t, batchconv(out_channels, out_channels, sz))

    def forward(self, x):
        """ Forward pass for residual blocks
        Args:
            x(array[float32]): Input array
        Returns:
            x(array[float32]): Array after performing series of normalisation and convolution
                               operations

        """
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        x = x + self.conv[3](self.conv[2](x))
        return x


class convdown(nn.Module):
    """ Class for down convolution
    Args:
        in_channels(int): Number of channels in input image
        out_channels(int): Number of channels in output image
        sz(int): Kernel size

    """

    def __init__(self, in_channels, out_channels, sz):
        super().__init__()
        self.conv = nn.Sequential()
        for t in range(2):
            if t == 0:
                self.conv.add_module('conv_%d' % t, batchconv(in_channels, out_channels, sz))
            else:
                self.conv.add_module('conv_%d' % t, batchconv(out_channels, out_channels, sz))

    def forward(self, x):
        """ Forward pass for down convolution
        Args:
            x(array): Array module before down convolution
        Returns:
            x(array): Array module after down convolution

        """
        x = self.conv[0](x)
        x = self.conv[1](x)
        return x


class downsample(nn.Module):
    """ Class for downsampling
    Args:
        nbase(list): Number of in channels
        sz(int): Kernel size
        residual_on(bool): Unet parameter

    """

    def __init__(self, nbase, sz, residual_on=True):
        super().__init__()
        self.down = nn.Sequential()
        self.maxpool = nn.MaxPool2d(2, 2)
        for n in range(len(nbase) - 1):
            if residual_on:
                self.down.add_module('res_down_%d' % n, resdown(nbase[n], nbase[n + 1], sz))
            else:
                self.down.add_module('conv_down_%d' % n, convdown(nbase[n], nbase[n + 1], sz))

    def forward(self, x):
        """ Forward pass of downsampling  performing maxpooling.
        Args:
            x(array): Array module before down sample
        Returns:
            x(array): Array module after down sample

        """
        xd = []
        for n in range(len(self.down)):
            if n > 0:
                y = self.maxpool(xd[n - 1])
            else:
                y = x
            xd.append(self.down[n](y))
        return xd


class batchconvstyle(nn.Module):
    """ Class handles batch convolution
        Args:
            in_channels(int): Number of channels in input image
            out_channels(int): Number of channels in output image
            sz(int): Kernel size
            style_channels(int): Number of classes
            concatenation(bool): Unet parameter

    """
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False):
        super().__init__()
        self.concatenation = concatenation
        self.conv = batchconv(in_channels, out_channels, sz)
        if concatenation:
            self.full = nn.Linear(style_channels, out_channels * 2)
        else:
            self.full = nn.Linear(style_channels, out_channels)

    def forward(self, style, x, mkldnn=False):
        """ Forward pass for batchconvstyle
        Args:
            style(array): Vector after downsampling
            x(array): Array to perform batchconvstyle
            mkldnn(bool): Check for mkldnn
        Returns:
            y(array): Output after convolution

        """
        feat = self.full(style)
        if mkldnn:
            x = x.to_dense()
            y = (x + feat.unsqueeze(-1).unsqueeze(-1)).to_mkldnn()
        else:
            y = x + feat.unsqueeze(-1).unsqueeze(-1)
        y = self.conv(y)
        return y


class resup(nn.Module):
    """ Class to build residual blocks
    Args:
        in_channels(int): Number of channels in input image
        out_channels(int): Number of channels in output image
        sz(int): Kernel size
        style_channels(int): Number of classes
        concatenation(bool): Unet parameter

    """
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', batchconv(in_channels, out_channels, sz))
        self.conv.add_module('conv_1',
                             batchconvstyle(out_channels, out_channels, style_channels, sz,
                                            concatenation=concatenation))
        self.conv.add_module('conv_2',
                             batchconvstyle(out_channels, out_channels, style_channels, sz))
        self.conv.add_module('conv_3',
                             batchconvstyle(out_channels, out_channels, style_channels, sz))
        self.proj = batchconv0(in_channels, out_channels, 1)

    def forward(self, x, y, style, mkldnn=False):
        """ Function handling forward pass
        Args:
            x(array): Input array to residual block
            y(array): Kernel size
            style(array): Vector after downsampling
            mkldnn(bool): True if mkldnn is available
        Returns:
            x(array): Output after passing through residual blocks

        """
        x = self.proj(x) + self.conv[1](style, self.conv[0](x) + y, mkldnn=mkldnn)
        x = x + self.conv[3](style, self.conv[2](style, x, mkldnn=mkldnn), mkldnn=mkldnn)
        return x


class convup(nn.Module):
    """ Up Convolution class
    Args:
        in_channels(int): Number of channels in input image
        out_channels(int): Number of channels in output image
        sz(int): Kernel size
        style_channels(int): Number of classes
        concatenation(bool): Unet parameter

    """
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', batchconv(in_channels, out_channels, sz))
        self.conv.add_module('conv_1',
                             batchconvstyle(out_channels, out_channels, style_channels, sz,
                                            concatenation=concatenation))

    def forward(self, x, y, style):
        """ Function handling forward pass
        Args:
            x(array): Array to perform upconvolution on
            y(array): Kernel size
            style(array): Vector after downsampling
        Returns:
            x(array):  Array after upconvolution

        """
        x = self.conv[1](style, self.conv[0](x) + y)
        return x


class make_style(nn.Module):
    """ Class handling style vector between downsampling and upsampling pass

    """
    def __init__(self):
        super().__init__()
        # self.pool_all = nn.AvgPool2d(28)
        self.flatten = nn.Flatten()

    def forward(self, x0):
        """ Forward pass
        Args:
            x0(array): Final output array after downsampling
        Returns:
            style(array): Flattened output array after downsampling

        """
        # style = self.pool_all(x0)
        style = F.avg_pool2d(x0, kernel_size=(x0.shape[-2], x0.shape[-1]))
        style = self.flatten(style)
        style = style / torch.sum(style ** 2, axis=1, keepdim=True) ** .5

        return style


class upsample(nn.Module):
    """ Upsampling part of neural network
    Args:
        nbase(list): Number of in channels
        sz(int): kernel size
        residual_on(bool): Unet parameter
        concatenation(bool): Unet parameter

    """
    def __init__(self, nbase, sz, residual_on=True, concatenation=False):
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.up = nn.Sequential()
        for n in range(1, len(nbase)):
            if residual_on:
                self.up.add_module('res_up_%d' % (n - 1),
                                   resup(nbase[n], nbase[n - 1], nbase[-1], sz, concatenation))
            else:
                self.up.add_module('conv_up_%d' % (n - 1),
                                   convup(nbase[n], nbase[n - 1], nbase[-1], sz, concatenation))

    def forward(self, style, xd, mkldnn=False):
        """ Forward pass for upsampling
        Args:
            style(array): Downsampled final vector
            xd(array): Concatenation array
            mkldnn(bool): True if mkldnn is available

        """
        x = self.up[-1](xd[-1], xd[-1], style, mkldnn=mkldnn)
        for n in range(len(self.up) - 2, -1, -1):
            if mkldnn:
                x = self.upsampling(x.to_dense()).to_mkldnn()
            else:
                x = self.upsampling(x)
            x = self.up[n](x, xd[n], style, mkldnn=mkldnn)
        return x


class CPnet(nn.Module):
    """ Main class for NN
    Args:
        nout(int): Number of classes
        nbase(list): Number of in channels
        sz(int): Kernel size
        residual_on(bool): Unet parameter
        concatenation(bool): Unet parameter
        mkldnn(bool): True if mkldnn is available
        style_on(bool): True if styles is enabled

    """
    def __init__(self, nbase, nout, sz, residual_on=True,
                 style_on=True, concatenation=False, mkldnn=False):
        super(CPnet, self).__init__()
        self.nbase = nbase
        self.nout = nout
        self.sz = sz
        self.residual_on = residual_on
        self.style_on = style_on
        self.concatenation = concatenation
        self.mkldnn = mkldnn if mkldnn is not None else False
        self.downsample = downsample(nbase, sz, residual_on=residual_on)
        nbaseup = nbase[1:]
        nbaseup.append(nbaseup[-1])
        self.upsample = upsample(nbaseup, sz, residual_on=residual_on, concatenation=concatenation)
        self.make_style = make_style()
        self.output = batchconv(nbaseup[0], nout, 1)
        self.style_on = style_on

    def forward(self, data):
        """ Forward pass for network
        Args:
            data(array): Batch of unlabelled images for prediction
        Returns:
            T0(array): Final Array after downsampled
            style0(array): Flattened downsampled array

        """
        if self.mkldnn:
            data = data.to_mkldnn()
        T0 = self.downsample(data)
        if self.mkldnn:
            style = self.make_style(T0[-1].to_dense())
        else:
            style = self.make_style(T0[-1])
        style0 = style
        if not self.style_on:
            style = style * 0
        T0 = self.upsample(style, T0, self.mkldnn)
        T0 = self.output(T0)
        if self.mkldnn:
            T0 = T0.to_dense()
        return T0, style0

    def save_model(self, filename):
        """ Saves pretrained model
        Args:
            filename(str): Filename

        """
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, cpu=False):
        """ Loads pretrained model
        Args:
            filename(str): Filename
            cpu(bool) : Loading using cpu/gpu

        """
        if not cpu:
            self.load_state_dict(torch.load(filename))
        else:
            self.__init__(self.nbase,
                          self.nout,
                          self.sz,
                          self.residual_on,
                          self.style_on,
                          self.concatenation,
                          self.mkldnn)
            self.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
