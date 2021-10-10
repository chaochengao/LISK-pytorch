import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from .ops import Conv2dLayer, TransposeConv2dLayer, ResnetBlock, attention

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

class SIGeneratorNet(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True):
        super(SIGeneratorNet, self).__init__()
        in_c = 11
        out_c = 64
        # Encoder
        # scale 256
        self.en_conv1 = Conv2dLayer(in_channels=in_c, out_channels=out_c, kernel_size=7, stride=1, padding=3, activation='relu')

        # scale 128
        self.en_conv2 = Conv2dLayer(in_channels=out_c, out_channels=2*out_c, kernel_size=4, stride=2, padding=1, activation='relu')

        # scale 64
        self.en_conv3 = Conv2dLayer(in_channels=2*out_c, out_channels=4*out_c, kernel_size=4, stride=2, padding=1, activation='relu')

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(4*out_c, 2)
            blocks.append(block)

        self.en_64_8 = nn.Sequential(*blocks)

        # Decoder
        # TODO: output scale 64  Down scale = 2 (origin) pool scale = 2 (origin)
        # share attention
        self.attention_pooling_64 = attention(in_channels=4*out_c, out_channels=4*out_c, pool_scale=2, kernel_size=4, stride=2, padding=0)

        # out of predict grad map
        self.out64_grad_out = Conv2dLayer(in_channels=4*out_c, out_channels=4*out_c, kernel_size=5, stride=1, padding=2, activation='relu')
        self.grad64 = Conv2dLayer(in_channels=4*out_c, out_channels=6, kernel_size=1, stride=1, padding=0, activation='none')
        self.out64 = Conv2dLayer(in_channels=4*out_c, out_channels=3, kernel_size=1, stride=1, padding=0, activation='tanh')

        # scale 64 - 128
        self.de128_conv4_upsample = TransposeConv2dLayer(in_channels=8*out_c, out_channels=2*out_c, kernel_size=4, stride=2, padding=1, activation='relu')

        # TODO: output scale 128
        # share attention
        self.attention_pooling_128 = attention(in_channels=2*out_c, out_channels=2*out_c, pool_scale=2, kernel_size=4, stride=2, padding=0)

        # out of predict grad map
        self.out128_grad_out = Conv2dLayer(in_channels=2*out_c, out_channels=2*out_c, kernel_size=5, stride=1, padding=2, activation='relu')
        self.grad128 = Conv2dLayer(in_channels=2*out_c, out_channels=6, kernel_size=1, stride=1, padding=0, activation='none')
        self.out128 = Conv2dLayer(in_channels=2*out_c, out_channels=3, kernel_size=1, stride=1, padding=0, activation='tanh')

        # scale 128 - 256
        self.de256_conv4_upsample = TransposeConv2dLayer(in_channels=4*out_c, out_channels=out_c, kernel_size=4, stride=2, padding=1, activation='relu')

        # TODO: output scale 256
        # share attention
        self.attention_pooling_256 = attention(in_channels=out_c, out_channels=out_c, pool_scale=2, kernel_size=4, stride=2, padding=0)

        # out of predict grad map
        self.out256_grad_out = Conv2dLayer(in_channels=out_c, out_channels=out_c, kernel_size=5, stride=1, padding=2, activation='relu')
        self.grad256 = Conv2dLayer(in_channels=out_c, out_channels=6, kernel_size=1, stride=1, padding=0, activation='none')
        self.out256 = Conv2dLayer(in_channels=out_c, out_channels=3, kernel_size=1, stride=1, padding=0, activation='tanh')

        if init_weights:
            self.init_weights()


    def forward(self, x):
        # Encoder
        # scale 256
        # torch.set_printoptions(profile="full", precision=8)

        x = self.en_conv1(x)
        # scale 128
        x = self.en_conv2(x)
        # scale 64
        x = self.en_conv3(x)
        # res block
        x = self.en_64_8(x)

        # Decoder
        # TODO: output scale 64  Down scale = 2 (origin) pool scale = 2 (origin)
        # share attention

        x = self.attention_pooling_64(x)

        # out of predict grad map
        x_64 = self.out64_grad_out(x)
        x_grad_out_64 = self.grad64(x_64)
        x_out_64 = self.out64(x_64)

        # scale 64 - 128
        x = torch.cat([x, x_64], dim=1)
        x = self.de128_conv4_upsample(x)

        # TODO: output scale 128
        # share attention
        x = self.attention_pooling_128(x)

        # out of predict grad map
        x_128 = self.out128_grad_out(x)
        x_grad_out_128 = self.grad128(x_128)
        x_out_128 = self.out128(x_128)

        # scale 128 - 256
        x = torch.cat([x, x_128], dim=1)
        x = self.de256_conv4_upsample(x)

        # out of predict grad map
        x = self.out256_grad_out(x)
        x_grad = self.grad256(x)
        x = self.out256(x)

        return x, x_out_64, x_out_128, x_grad, x_grad_out_64, x_grad_out_128

class SIDiscriminatorNet(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True):
        super(SIDiscriminatorNet, self).__init__()

        in_c = 3
        out_c = 64


        self.conv1 = Conv2dLayer(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1, norm='none', activation='lrelu', sn=True)
        self.conv2 = Conv2dLayer(in_channels=out_c, out_channels=2*out_c, kernel_size=4, stride=2, padding=1, norm='none', activation='lrelu', sn=True)
        self.conv3 = Conv2dLayer(in_channels=2*out_c, out_channels=4*out_c, kernel_size=4, stride=2, padding=1, norm='none', activation='lrelu', sn=True)
        self.conv4 = Conv2dLayer(in_channels=4*out_c, out_channels=8*out_c, kernel_size=4, stride=1, padding=0, norm='none', activation='lrelu', sn=True)
        self.conv5 = Conv2dLayer(in_channels=8*out_c, out_channels=1, kernel_size=4, stride=1, padding=0, norm='none', activation='none', sn=True)

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x


from .resnet import ResNet

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_relu(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.ReLU(inplace=True))


class DulaNetBranch(BaseNetwork):
    def __init__(self, pretrained=False):
        super(DulaNetBranch, self).__init__()

        self.encoder = ResNet([2, 2, 2, 2], pretrained)

        self.decoder = nn.ModuleList([
            conv3x3_relu(512, 256),
            conv3x3_relu(256, 128),
            conv3x3_relu(128, 64),
            conv3x3_relu(64, 32),
            conv3x3_relu(32, 16),
        ])
        self.last = conv3x3(16, 1)

        for parm in self.parameters():
            parm.requires_grad = False


    def forward(self, x):
        feat = self.encoder(x)
        x = feat
        for conv in self.decoder:
            x = F.interpolate(x, scale_factor=(2, 2), mode='nearest')
            x = conv(x)
        # fp = F.sigmoid(self.last(x))
        fp = self.last(x)

        return fp

    def forward_1(self, x):
        feat = self.encoder(x)
        return feat

    def forward_2(self, feat):
        x = feat
        for conv in self.decoder:
            x = F.interpolate(x, scale_factor=(2,2), mode='nearest')
            x = conv(x)
        #fp = F.sigmoid(self.last(x))
        fp = self.last(x)
        return fp

    def forward_get_feats(self, x):
        feat = self.encoder(x)
        x = feat
        lst = [x]
        for conv in self.decoder:
            x = F.interpolate(x, scale_factor=(2,2), mode='nearest')
            x = conv(x)
            lst.append(x)
        #fp = F.sigmoid(self.last(x))
        fp = self.last(x)
        return fp, lst

    def forward_from_feats(self, feat, lst):
        x = feat
        for i, conv in enumerate(self.decoder):
            x = x + lst[i]
            x = F.interpolate(x, scale_factor=(2,2), mode='nearest')
            x = conv(x)
        fp = self.last(x)
        return fp



