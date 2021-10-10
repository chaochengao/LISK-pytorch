import torch
import torch.nn as nn
import torchvision.models as models


# class AdversarialLoss(nn.Module):
#     r"""
#     Adversarial loss
#     https://arxiv.org/abs/1711.10337
#     """
#
#     def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
#         r"""
#         type = nsgan | lsgan | hinge
#         """
#         super(AdversarialLoss, self).__init__()
#
#         self.type = type
#         self.register_buffer('real_label', torch.tensor(target_real_label))
#         self.register_buffer('fake_label', torch.tensor(target_fake_label))
#
#         if type == 'nsgan':
#             self.criterion = nn.BCELoss()
#
#         elif type == 'lsgan':
#             self.criterion = nn.MSELoss()
#
#         elif type == 'hinge':
#             self.criterion = nn.ReLU()
#
#     def __call__(self, outputs, is_real, is_disc=None):
#         if self.type == 'hinge':
#             if is_disc:
#                 if is_real:
#                     outputs = -outputs
#                 return self.criterion(1 + outputs).mean()
#             else:
#                 return (-outputs).mean()
#
#         else:
#             labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
#             loss = self.criterion(outputs, labels)
#             return loss

# import lpips
#
# class LPIPSLoss(nn.Module):
#     def __init__(self):
#         super(LPIPSLoss, self).__init__()
#         self.loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
#         self.loss_fn_vgg = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization
#
#     def __call__(self, pos, neg):
#         loss = self.loss_fn_alex(pos, neg)
#
#         return loss

class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='hinge', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type

        if type == 'nsgan':
            self.criterion = nn.BCEWithLogitsLoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

        self.real_label = target_real_label
        self.fake_label = target_fake_label

    def __call__(self, pos, neg):
        if self.type == 'hinge':
            d_loss_real = torch.mean(self.criterion(1.0 - pos))
            d_loss_fake = torch.mean(self.criterion(1.0 + neg))
            d_loss = d_loss_real + d_loss_fake

            g_loss = -torch.mean(neg)

        else:
            g_loss = torch.mean(
                self.criterion(neg, self.real_label.expand_as(neg)))
            d_loss_fake = torch.mean(
                self.criterion(neg, self.fake_label.expand_as(neg)))
            d_loss_real = torch.mean(
                self.criterion(pos, self.real_label.expand_as(pos)))
            d_loss = d_loss_fake + d_loss_real  # 鍒ゅ埆鍣╨oss

        return g_loss, d_loss, d_loss_real, d_loss_fake


class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[0.2, 0.2, 0.2, 0.2, 0.2]):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.MSELoss()
        self.weights = weights

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        # G = f.bmm(f_T) / (h * w * ch)
        G = f.bmm(f_T)

        return G

    def feature_size(self, vgg_l):
        n, c, h, w = vgg_l.size()
        feature_size = n * c * h * w
        return feature_size

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        # Compute loss
        style_loss = 0.0
        style_loss += self.weights[0] * (self.criterion(self.compute_gram(x_vgg['relu1_1']),
                                                        self.compute_gram(y_vgg['relu1_1'])) / self.feature_size(
            x_vgg['relu1_1']))
        style_loss += self.weights[0] * (self.criterion(self.compute_gram(x_vgg['relu2_1']),
                                                        self.compute_gram(y_vgg['relu2_1'])) / self.feature_size(
            x_vgg['relu2_1']))
        style_loss += self.weights[0] * (self.criterion(self.compute_gram(x_vgg['relu3_1']),
                                                        self.compute_gram(y_vgg['relu3_1'])) / self.feature_size(
            x_vgg['relu3_1']))
        style_loss += self.weights[0] * (self.criterion(self.compute_gram(x_vgg['relu4_1']),
                                                        self.compute_gram(y_vgg['relu4_1'])) / self.feature_size(
            x_vgg['relu4_1']))
        style_loss += self.weights[0] * (self.criterion(self.compute_gram(x_vgg['relu5_1']),
                                                        self.compute_gram(y_vgg['relu5_1'])) / self.feature_size(
            x_vgg['relu5_1']))

        return style_loss


class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[0.2, 0.2, 0.2, 0.2, 0.2]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.MSELoss()
        self.weights = weights

    def feature_size(self, vgg_l):
        n, c, h, w = vgg_l.size()
        feature_size = n * c * h * w
        return feature_size

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        content_loss = 0.0
        content_loss += self.weights[0] * (
                    self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1']) / self.feature_size(x_vgg['relu1_1']))
        content_loss += self.weights[1] * (
                    self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1']) / self.feature_size(x_vgg['relu2_1']))
        content_loss += self.weights[2] * (
                    self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1']) / self.feature_size(x_vgg['relu3_1']))
        content_loss += self.weights[3] * (
                    self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1']) / self.feature_size(x_vgg['relu4_1']))
        content_loss += self.weights[4] * (
                    self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1']) / self.feature_size(x_vgg['relu5_1']))

        return content_loss


class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out


"""
Salient Edge
"""
import cv2
import numpy as np
import torch.nn.functional as F


class PriorityLoss(torch.nn.Module):
    def __init__(self):
        super(PriorityLoss, self).__init__()

    def gaussian_kernel_2d_opencv(self, kernel_size=3, sigma=0):
        """
        ref: https://blog.csdn.net/qq_16013649/article/details/78784791
        ref: tensorflow
            (1) https://stackoverflow.com/questions/52012657/how-to-make-a-2d-gaussian-filter-in-tensorflow
            (2) https://github.com/tensorflow/tensorflow/issues/2826
        """
        kx = cv2.getGaussianKernel(kernel_size, sigma)
        ky = cv2.getGaussianKernel(kernel_size, sigma)
        return np.multiply(kx, np.transpose(ky))

    def __call__(self, mask, ksize=5, sigma=1, iteration=2):
        gaussian_kernel = self.gaussian_kernel_2d_opencv(kernel_size=ksize, sigma=sigma)
        gaussian_kernel = np.reshape(gaussian_kernel, (1, 1, ksize, ksize))
        gaussian_kernel = torch.from_numpy(gaussian_kernel).float().to(torch.device('cuda'))
        # mask_priority = torch.from_numpy(mask)
        mask_priority = mask

        for i in range(iteration):
            mask_priority = F.conv2d(mask_priority, gaussian_kernel, stride=1, padding=2)

        return mask_priority


class L1Loss(torch.nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def __call__(self, image, predict, mask, type='foreground'):
        error = torch.abs(predict - image)
        if type == 'foreground':
            loss = torch.sum(mask * error) / torch.sum(mask)  # * tf.reduce_sum(1. - mask) for balance?
        elif type == 'background':
            loss = torch.sum((1. - mask) * error) / torch.sum(1. - mask)
        else:
            loss = torch.sum(mask * torch.abs(predict - image)) / torch.sum(mask)
        return loss

