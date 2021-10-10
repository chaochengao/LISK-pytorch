import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .networks import SIGeneratorNet, SIDiscriminatorNet
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss, PriorityLoss, L1Loss
from .ops import load_grad_tensor

from tensorboardX import SummaryWriter

class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()
        self.writer = SummaryWriter(os.path.join(config.PATH, 'logs', 'log_SINet'))

        self.name = name
        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict(),
            'generator_optimizer': self.gen_optimizer.state_dict()
        }, self.gen_weights_path)

        torch.save({
            'discriminator': self.discriminator.state_dict(),
            'discriminator_optimizer': self.dis_optimizer.state_dict()
        }, self.dis_weights_path)


class SIInpaintingModel(BaseModel):
    def __init__(self, config):
        super(SIInpaintingModel, self).__init__('SIInpaintingModel', config)

        # generator input: [RGB(3) + Gradient(6) + mask(1)? + edge(1)?]
        # discriminator input: (grayscale(1) + edge(1))
        generator = SIGeneratorNet()
        discriminator = SIDiscriminatorNet()
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        l1_loss = L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)
        priority_loss = PriorityLoss()

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)
        self.add_module('priority_loss', priority_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.G_LR),
            betas=(0, 0.9)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.D_LR),
            betas=(0, 0.9)
        )

    def process(self, images, edges, masks, grads):
        self.iteration += 1
        # zero optimizers
        self.gen_optimizer.zero_grad()
        # self.dis_optimizer.zero_grad()

        out_256, out_64, out_128, out_grad_256, out_grad_64, out_grad_128 = self(images, edges,  masks, grads)
        outputs = out_256
        gen_loss = 0
        dis_loss = 0


        # incomplete image at full scale
        x_incomplete = images * (1. - masks)  # mask: 0 for valid pixel, 1 (white) for hole

        # incomplete edge at full scale
        input_edge = 1 - edges                 # 0 (black) for edge when save and input, 1 (white) for non edge
        edge_incomplete = input_edge * (1 - masks) + masks

        # incomplete grad
        grad_incomplete = (1. - masks) * grads

        """##### Losses #####"""
        losses = {}  # use a dict to collect losses

        # TODO: scale 64
        # complete image
        re_size = int(self.config.INPUT_SIZE / 4)
        edge_64 = F.interpolate(edges, size=re_size, mode='nearest')
        mask_64 = F.interpolate(masks, size=re_size, mode='nearest')
        x_pos_64 = F.interpolate(images, size=re_size, mode='nearest')

        x_incomplete_64 = x_pos_64 * (1. - mask_64)
        x_complete_64 = out_64 * mask_64 + x_incomplete_64
        x_neg_64 = x_complete_64 # neg input (fake)

        # Auxilary task: edge and grad loss
        grad_64 = load_grad_tensor(x_pos_64)  # normalization?
        grad_incomplete_64 = (1. - mask_64) * grad_64
        grad_complete_64 = out_grad_64 * mask_64 + grad_incomplete_64


        # more weight for edges?
        edge_mask_64 = edge_64                                       # 1 for edge, 0 for grad, when using feature.canny()
        mask_priority_64 = self.priority_loss(edge_mask_64, ksize=5, sigma=1, iteration=2)
        edge_weight_64 = self.config.EDGE_ALPHA * mask_priority_64    # salient edge

        grad_weight_64 = self.config.GRAD_ALPHA                       # equaled grad

        # error
        grad_error_64 = torch.abs(out_grad_64 - grad_64)

        losses['edge_l1_loss_64'] = torch.sum(edge_weight_64 * grad_error_64) / torch.sum(edge_weight_64) / 6.

        # grad pixel level reconstruction loss
        if self.config.GRAD_ALPHA > 0:
            losses['grad_l1_loss_64'] = torch.mean(grad_weight_64 * grad_error_64)
        else:
            losses['grad_l1_loss_64'] = 0.

        # Main task: compute losses
        # l1 loss
        # if args.L1_SCALE > 0.:
        losses['l1_loss_fore_64'] = self.config.L1_SCALE * self.config.L1_FORE_ALPHA * self.l1_loss(x_pos_64, out_64, mask_64,
                                                                                 type='foreground')
        losses['l1_loss_back_64'] = self.config.L1_SCALE * self.config.L1_BACK_ALPHA * self.l1_loss(x_pos_64, out_64, mask_64,
                                                                                 type='background')

        # self.losses_64 = [losses['l1_loss_fore_64'],
        #                   losses['l1_loss_back_64'],
        #                   losses['grad_l1_loss_64'],
        #                   losses['edge_l1_loss_64'],
        #                   ]
        # Summary
        viz_img_64 = [x_pos_64, x_incomplete_64, x_complete_64, out_64]
        viz_grad_64 = [grad_64[:, 0:1, :, :], grad_incomplete_64[:, 0:1, :, :], grad_complete_64[:, 0:1, :, :], out_grad_64[:, 0:1, :, :]]
        img_64 = torch.cat(viz_img_64, dim=3)
        grad_64 = torch.cat(viz_grad_64, dim=3)

        # TODO: scale 128
        # complete image
        re_size = int(self.config.INPUT_SIZE / 2)
        edge_128 = F.interpolate(edges, size=re_size, mode='nearest')
        mask_128 = F.interpolate(masks, size=re_size, mode='nearest')
        x_pos_128 = F.interpolate(images, size=re_size, mode='nearest')

        x_incomplete_128 = x_pos_128 * (1. - mask_128)
        x_complete_128 = out_128 * mask_128 + x_incomplete_128
        x_neg_128 = x_complete_128  # neg input (fake)

        # Auxilary task: edge and grad loss
        grad_128 = load_grad_tensor(x_pos_128)  # normalization?
        grad_incomplete_128 = (1. - mask_128) * grad_128
        grad_complete_128 = out_grad_128 * mask_128 + grad_incomplete_128

        # more weight for edges?
        edge_mask_128 = edge_128  # 1 for edge, 0 for grad, when using feature.canny()
        mask_priority_128 = self.priority_loss(edge_mask_128, ksize=5, sigma=1, iteration=2)
        edge_weight_128 = self.config.EDGE_ALPHA * mask_priority_128  # salient edge

        grad_weight_128 = self.config.GRAD_ALPHA  # equaled grad

        # error
        grad_error_128 = torch.abs(out_grad_128 - grad_128)

        losses['edge_l1_loss_128'] = torch.sum(edge_weight_128 * grad_error_128) / torch.sum(edge_weight_128) / 6.

        # grad pixel level reconstruction loss
        if self.config.GRAD_ALPHA > 0:
            losses['grad_l1_loss_128'] = torch.sum(grad_weight_128 * grad_error_128)
        else:
            losses['grad_l1_loss_128'] = 0.

        # Main task: compute losses
        # l1 loss
        # if args.L1_SCALE > 0.:
        losses['l1_loss_fore_128'] = self.config.L1_SCALE * self.config.L1_FORE_ALPHA * self.l1_loss(x_pos_128, out_128,
                                                                                                    mask_128,
                                                                                                    type='foreground')
        losses['l1_loss_back_128'] = self.config.L1_SCALE * self.config.L1_BACK_ALPHA * self.l1_loss(x_pos_128, out_128,
                                                                                                    mask_128,
                                                                                                    type='background')
        # Summary
        viz_img_128 = [x_pos_128, x_incomplete_128, x_complete_128, out_128]
        viz_grad_128 = [grad_128[:, 0:1, :, :], grad_incomplete_128[:, 0:1, :, :], grad_complete_128[:, 0:1, :, :], out_grad_128[:, 0:1, :, :]]
        img_128 = torch.cat(viz_img_128, dim=3)
        grad_128 = torch.cat(viz_grad_128, dim=3)

        # TODO: scale 256
        # apply mask and complete image
        edge_256 = edges
        mask_256 = masks
        x_pos_256 = images

        x_incomplete_256 = x_pos_256 * (1. - mask_256)
        x_complete_256 = out_256 * mask_256 + x_incomplete_256

        # Auxilary task: edge and grad loss
        grad_256 = load_grad_tensor(x_pos_256)  # normalization?
        grad_incomplete_256 = (1. - mask_256) * grad_256
        grad_complete_256 = out_grad_256 * mask_256 + grad_incomplete_256

        grad_256 = load_grad_tensor(x_pos_256)  # normalization?
        grad_incomplete_256 = (1. - mask_256) * grad_256
        grad_complete_256 = out_grad_256 * mask_256 + grad_incomplete_256

        # more weight for edges?
        edge_mask_256 = edge_256  # 1 for edge, 0 for grad, when using feature.canny()
        mask_priority_256 = self.priority_loss(edge_mask_256, ksize=5, sigma=1, iteration=2)
        edge_weight_256 = self.config.EDGE_ALPHA * mask_priority_256  # salient edge

        grad_weight_256 = self.config.GRAD_ALPHA  # equaled grad

        # error
        grad_error_256 = torch.abs(out_grad_256 - grad_256)

        losses['edge_l1_loss_256'] = torch.sum(edge_weight_256 * grad_error_256) / torch.sum(edge_weight_256) / 6.

        # grad pixel level reconstruction loss
        if self.config.GRAD_ALPHA > 0:
            losses['grad_l1_loss_256'] = torch.mean(grad_weight_256 * grad_error_256)
        else:
            losses['grad_l1_loss_256'] = 0.

        # compute losses
        x_neg_256 = x_complete_256  # neg input (fake)
        x_pos_256 = images  # pos input (real)

        # Main task: compute losses
        # l1 loss
        # if args.L1_SCALE > 0.:
        losses['l1_loss_fore_256'] = self.config.L1_FORE_ALPHA * self.l1_loss(x_pos_256, out_256, mask_256, type='foreground')
        losses['l1_loss_back_256'] = self.config.L1_BACK_ALPHA * self.l1_loss(x_pos_256, out_256, mask_256, type='background')

        # content loss, style loss
        layers = {'relu1_1': 0.2, 'relu2_1': 0.2, 'relu3_1': 0.2, 'relu4_1': 0.2, 'relu5_1': 0.2}
        if self.config.CONTENT_FORE_ALPHA > 0.:
            losses['content_loss_256'] = self.config.CONTENT_FORE_ALPHA * self.perceptual_loss(x_pos_256, x_neg_256)
        else:
            losses['content_loss_256'] = 0.
        if self.config.STYLE_FORE_ALPHA > 0.:
            # layers = {'pool1': 0.33, 'pool2': 0.34, 'pool3': 0.33}
            losses['style_loss_256'] = self.config.STYLE_FORE_ALPHA * self.style_loss(x_pos_256, x_neg_256)
        else:
            losses['style_loss_256'] = 0.

        if self.config.BACKGROUND_LOSS:
            layers = {'relu1_1': 0.2, 'relu2_1': 0.2, 'relu3_1': 0.2, 'relu4_1': 0.2, 'relu5_1': 0.2}
            losses['content_loss_256'] += self.config.CONTENT_BACK_ALPHA * self.perceptual_loss(x_pos_256, out_256)
            # layers = {'pool1': 0.33, 'pool2': 0.34, 'pool3': 0.33}
            losses['style_loss_256'] += self.config.STYLE_BACK_ALPHA * self.style_loss(x_pos_256, out_256)


        # patch-gan-loss
        x_pos_neg_256 = torch.cat([x_pos_256, x_neg_256], dim=0)  # input as pos-neg to global discriminator
        pos_neg_256 = self.discriminator(x_pos_neg_256)
        pos_256, neg_256 = torch.split(pos_neg_256, self.config.BATCH_SIZE, dim=0)

        g_loss_256, d_loss_256, d_loss_real_256, d_loss_fake_256 = self.adversarial_loss(pos_256, neg_256)
        losses['g_loss_256'] = g_loss_256
        losses['d_loss_256'] = d_loss_256

        # Summary
        viz_img_256 = [x_pos_256, x_incomplete_256, x_complete_256, out_256]
        viz_grad_256 = [grad_256[:, 0:1, :, :], grad_incomplete_256[:, 0:1, :, :], grad_complete_256[:, 0:1, :, :], out_grad_256[:, 0:1, :, :], grads[:, 0:1, :, :]]
        img_256 = torch.cat(viz_img_256, dim=3)
        grad_256 = torch.cat(viz_grad_256, dim=3)


        """##### Training Ops #####"""
        # train ops
        # Scale 64
        losses['total_g_loss_64'] = self.config.ALPHA * (losses['edge_l1_loss_64'] + losses['grad_l1_loss_64'])

        # Scale 128
        losses['total_g_loss_128'] = self.config.ALPHA * (losses['edge_l1_loss_128'] + losses['grad_l1_loss_128'])

        # losses['content_loss_256'] = 0
        # losses['style_loss_256'] = 0

        # Scale 256
        losses['total_g_loss_256'] = losses['l1_loss_fore_256'] + \
                                     losses['l1_loss_back_256'] + \
                                     losses['content_loss_256'] + \
                                     losses['style_loss_256'] + \
                                     self.config.PATCH_GAN_ALPHA * losses['g_loss_256'] + \
                                     self.config.ALPHA * (losses['edge_l1_loss_256'] +
                                                   losses['grad_l1_loss_256'])

        losses['total_d_loss_256'] = losses['d_loss_256']

        gen_loss = losses['total_g_loss_256'] + losses['total_g_loss_128'] + losses['total_g_loss_64']
        # self.g_loss = losses['total_g_loss_256']    # without deep structure supervision
        dis_loss = losses['total_d_loss_256']


        logs = [
            # ("l_gen", gen_loss.item()),
            # ("l_dis", dis_loss.item()),
        ]


        return outputs, gen_loss, dis_loss, losses, logs, img_64, grad_64, img_128, grad_128, img_256, grad_256

    def forward(self, images, edges, masks, grads):
        x_incomplete = images* (1. - masks)
        edges = 1 - edges
        edge_incomplete = edges * (1 - masks) + masks
        grad_incomplete = (1. - masks) * grads

        ones_x = torch.ones_like(images)[:, 0:1, :, :]
        # print(x_incomplete.size(), edge_incomplete.size(), grad_incomplete.size(), ones_x.size())

        inputs = torch.cat((x_incomplete, ones_x * edge_incomplete, ones_x * masks, grad_incomplete), dim=1)
        x, x_out_64, x_out_128, x_grad, x_grad_out_64, x_grad_out_128 = self.generator(inputs)
        return x, x_out_64, x_out_128, x_grad, x_grad_out_64, x_grad_out_128

    def backward(self, gen_loss=None, dis_loss=None):
        # if dis_loss is not None:
        #     dis_loss.backward(retain_graph=True)
        # self.dis_optimizer.step()

        if gen_loss is not None:
            gen_loss.backward()
        self.gen_optimizer.step()




