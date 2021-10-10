import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter
from torch.autograd import Variable
import numpy as np

# -----------------------------------------------
#                Normal ConvBlock
# -----------------------------------------------
class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='reflect',
                 activation='relu', norm='in', sn=False):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = torch.nn.utils.spectral_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


# class TransposeConv2dLayer(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='zero',
#                  activation='lrelu', norm='none', sn=False, scale_factor=2):
#         super(TransposeConv2dLayer, self).__init__()
#         # Initialize the conv scheme
#         self.scale_factor = scale_factor
#         self.conv2d = Conv2dLayer(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type,
#                                   activation, norm, sn)
#
#     def forward(self, x):
#         x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
#         x = self.conv2d(x)
#         return x


class TransposeConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='reflect',
                 activation='relu', norm='in', sn=False):
        super(TransposeConv2dLayer, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2dTranspose = SpectralNorm(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=kernel_size, stride=stride,
                                   padding=padding))
        else:
            self.conv2dTranspose = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                                    kernel_size=kernel_size, stride=stride,
                                                    padding=padding)


    def forward(self, x):
        x = self.conv2dTranspose(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResConv2dLayer(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1, dilation=1, pad_type='zero', activation='lrelu',
                 norm='none', sn=False, scale_factor=2):
        super(ResConv2dLayer, self).__init__()
        # Initialize the conv scheme
        self.conv2d = nn.Sequential(
            Conv2dLayer(in_channels, in_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm,
                        sn),
            Conv2dLayer(in_channels, in_channels, kernel_size, stride, padding, dilation, pad_type, activation='none',
                        norm=norm, sn=sn)
        )

    def forward(self, x):
        residual = x
        out = self.conv2d(x)
        out = 0.1 * out + residual
        return out


# ----------------------------------------
#               Layer Norm
# ----------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps = 1e-8, affine = True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = Parameter(torch.Tensor(num_features).uniform_())
            self.beta = Parameter(torch.zeros(num_features))

    def forward(self, x):
        # layer norm
        shape = [-1] + [1] * (x.dim() - 1)                                  # for 4d input: [-1, 1, 1, 1]
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        # if it is learnable
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)                          # for 4d input: [1, -1, 1, 1]
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

#-----------------------------------------------
#                  SpectralNorm
#-----------------------------------------------
def l2normalize(v, eps = 1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

#-----------------------------------------------
#                  ResnetBlock
#-----------------------------------------------
class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, affine=True, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, affine=True, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

#-----------------------------------------------
#                  Attention
#-----------------------------------------------
class attention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 4, stride=2, neighbors=1, use_bias=True, sn=False,
                 down_scale = 2, pool_scale=2, name='attention_pooling', training=True, padding='REFLECT', reuse=False):
        super(attention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.neighbors = neighbors
        self.use_bias = use_bias
        self.sn =sn
        self.down_scale = down_scale
        self.pool_scale = pool_scale
        self.name = name
        self.training = training
        self.padding = padding
        self.reuse = reuse

        self.attention_down_sample = Conv2dLayer(in_channels=in_channels, out_channels=out_channels,
                                                 kernel_size=kernel_size, stride=down_scale, padding=1,
                                                 activation='relu')
        self.attention_down_upsample = TransposeConv2dLayer(in_channels=out_channels // 16, out_channels=in_channels,
                                                            kernel_size=kernel_size, stride=down_scale, padding=1,
                                                            activation='relu')


        self.f_conv = Conv2dLayer(in_channels=in_channels, out_channels=out_channels // 16, kernel_size=1, stride=1,
                                  pad_type='reflect', padding=0, activation='none', norm='none')
        self.g_conv = Conv2dLayer(in_channels=in_channels, out_channels=out_channels // 16, kernel_size=1, stride=1,
                                  pad_type='reflect', padding=0, activation='none', norm='none')
        self.h_conv = Conv2dLayer(in_channels=in_channels, out_channels=out_channels // 16, kernel_size=1, stride=1,
                                  pad_type='reflect', padding=0, activation='none', norm='none')
        self.softmax = nn.Softmax(dim=-1)

        self.gamma = Parameter(torch.FloatTensor([0.0]),requires_grad=True)
        # self.gamma = Variable(torch.FloatTensor([0.0]), requires_grad=True).to('cuda:0')


    def extract_patches(self, x, kernel=3, stride=1, padding='zero'):
        if kernel != 1:
            # Initialize the padding scheme
            if padding == 'reflect':
                x = nn.ReflectionPad2d(x)
            elif padding == 'replicate':
                x = nn.ReplicationPad2d(x)
            elif padding == 'zero':
                x = nn.ZeroPad2d(1)(x)
            else:
                assert 0, "Unsupported padding type: {}".format(padding)
        x = x.permute(0, 2, 3, 1)
        all_patches = x.unfold(1, kernel, stride).unfold(2, kernel, stride)
        return all_patches

    def l2_norm(self, v, eps=1e-12):
        return v / ((v ** 2).sum() ** 0.5 + eps)

    def hw_flatten(self, x):
        return torch.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

    def max_pooling(self, x, pool_size=2):
        m_p = torch.nn.MaxPool2d(kernel_size=pool_size, stride=pool_size, padding=0)
        x = m_p(x)
        return x

    def avg_pooling(self, x, pool_size=2):
        a_p = torch.nn.AvgPool2d(kernel_size=pool_size, stride=pool_size, padding=0)
        x = a_p(x)
        return x

    def pt2tf(self, x):
        if (len(x.size()) == 3):
            return x.permute(1, 2, 0)
        if (len(x.size()) == 4):
            return x.permute(0, 2, 3, 1)

    def tf2pt(self, x):
        if (len(x.size()) == 3):
            return x.permute(2, 0, 1)
        if (len(x.size()) == 4):
            return x.permute(0, 3, 1, 2)

    def forward(self, x):
        if self.neighbors > 1:
            pass
        else:
            x_origin = x
            # down sampling
            if self.down_scale > 1:
                x = self.attention_down_sample(x)

            # attention
            f = self.f_conv(x)
            f = self.max_pooling(f, self.pool_scale)

            g = self.g_conv(x)

            h = self.h_conv(x)
            h = self.max_pooling(h, self.pool_scale)

            f = self.pt2tf(f)
            g = self.pt2tf(g)
            h = self.pt2tf(h)

            # N = h * w
            s = torch.matmul(self.hw_flatten(g), self.hw_flatten(f).transpose(1, 2))  # # [bs, N, N]
            beta = self.softmax(s)

            o = torch.matmul(beta, self.hw_flatten(h))  # [bs, C, N]
            o = torch.reshape(o, shape=[x.shape[0], x.shape[2], x.shape[3], self.out_channels // 16])
            o = self.tf2pt(o)

            if self.down_scale > 1:
                o = self.attention_down_upsample(o)

            x = self.gamma * o + x_origin

        return x


#tf.sobel_edge
def load_grad_tensor(img):
    kernels = np.array([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
               [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]).astype(np.float32)

    img_t = img.to(torch.device("cuda"))

    img_t = torch.cat((img_t, img_t), dim=1)
    k1 = torch.from_numpy(kernels[0])
    k2 = torch.from_numpy(kernels[1])
    k1 = k1.unsqueeze(0)
    k2 = k2.unsqueeze(0)
    # print(img_t[0, 250, 0:10])
    k = torch.cat((k1, k1, k1, k2, k2, k2), dim=0)
    k = k.unsqueeze(1)
    w = torch.nn.Parameter(data=k, requires_grad = False).to(torch.device("cuda"))

    img_t = F.pad(input=img_t, pad=[1, 1, 1, 1], mode='reflect')
    grad = F.conv2d(img_t, w, stride=1, padding=0, groups=6)

    return grad[:, [0, 3, 1, 4, 2, 5], ...]