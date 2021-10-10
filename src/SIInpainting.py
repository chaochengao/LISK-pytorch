import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from .dataset import Dataset
from .models import SIInpaintingModel
from .networks import SIGeneratorNet, SIDiscriminatorNet
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR, EdgeAccuracy

from tensorboardX import SummaryWriter

import time

class SIInpainting():
    def __init__(self, config):
        self.config = config

        model_name = 'SIInpainting'

        self.writer = SummaryWriter(os.path.join(config.PATH, 'logs', model_name))

        self.debug = False
        self.model_name = model_name
        self.model = SIInpaintingModel(config).to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.edgeacc = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)

        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_EDGE_FLIST, config.TEST_MASK_FLIST,
                                        augment=False, training=False)
        else:
            self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_EDGE_FLIST, config.TRAIN_MASK_FLIST,
                                         augment=True, training=True)
            self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_EDGE_FLIST, config.VAL_MASK_FLIST,
                                       augment=False, training=True)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

    def load(self):
        self.model.load()

    def save(self):
        self.model.save()

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return
        torch.backends.cudnn.benchmark = True

        num_params = 0
        for param in self.model.parameters():
            if param.requires_grad:
                num_params += param.numel()
        print('Parameter numbers: ', num_params / 1e6, 'milions')

        while (keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)
            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])
            for items in train_loader:
                self.model.train()

                images, images_gray, edges, masks, grads = self.cuda(*items)
                outputs, gen_loss, dis_loss, losses, logs, img_64, grad_64, img_128, grad_128, img_256, grad_256 = self.model.process(images, edges, masks, grads)
                outputs_merged = (outputs * masks) + (images * (1 - masks))
                # metrics
                precision, recall = self.edgeacc(edges * masks, outputs * masks)
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))

                # backward
                self.model.backward(gen_loss, dis_loss)
                iteration = self.model.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break

                progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs])


                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0 or iteration == 1:
                    self.sample()

                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                    print('\nstart eval...\n')
                    self.eval()

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.writer.add_scalar("l1_loss_fore_64", losses['l1_loss_fore_64'], iteration)
                    self.writer.add_scalar("l1_loss_back_64", losses['l1_loss_back_64'], iteration)
                    self.writer.add_scalar('grad_l1_loss_64', losses['grad_l1_loss_64'], iteration)
                    self.writer.add_scalar('edge_l1_loss_64', losses['edge_l1_loss_64'], iteration)

                    self.writer.add_scalar("l1_loss_fore_128", losses['l1_loss_fore_128'], iteration)
                    self.writer.add_scalar("l1_loss_back_128", losses['l1_loss_back_128'], iteration)
                    self.writer.add_scalar('grad_l1_loss_128', losses['grad_l1_loss_128'], iteration)
                    self.writer.add_scalar('edge_l1_loss_128', losses['edge_l1_loss_128'], iteration)

                    self.writer.add_scalar('l1_loss_fore_256', losses['l1_loss_fore_256'], iteration)
                    self.writer.add_scalar("l1_loss_back_256", losses['l1_loss_back_256'], iteration)
                    self.writer.add_scalar('grad_l1_loss_256', losses['grad_l1_loss_256'], iteration)
                    self.writer.add_scalar('edge_l1_loss_256', losses['edge_l1_loss_256'], iteration)
                    self.writer.add_scalar('content_loss_256', losses['content_loss_256'], iteration)
                    self.writer.add_scalar('style_loss_256', losses['style_loss_256'], iteration)

                    self.writer.add_scalar('LPIPS_loss_256', losses['LPIPS_loss_256'], iteration)
                    self.writer.add_scalar('L1_loss_256', losses['L1_loss_256'], iteration)

                    self.writer.add_scalar('g_loss_256', losses['g_loss_256'], iteration)
                    self.writer.add_scalar('d_loss_256', losses['d_loss_256'], iteration)

                    self.writer.add_image('raw_incomplete_predicted_complete_64', (img_64[0]+ 1.) /2., iteration)
                    self.writer.add_image('raw_incomplete_predicted_completed_grad_64', (grad_64[0]+ 1.) /2., iteration)
                    self.writer.add_image('raw_incomplete_predicted_complete_128', (img_128[0]+ 1.) /2., iteration)
                    self.writer.add_image('raw_incomplete_predicted_completed_grad_128', (grad_128[0]+ 1.) /2., iteration)
                    self.writer.add_image('raw_incomplete_predicted_complete_256', (img_256[0]+ 1.) /2., iteration)
                    self.writer.add_image('raw_incomplete_predicted_completed_grad_256', (grad_256[0]+ 1.) /2., iteration)


                    self.writer.add_scalar('edge_similarity/precision', precision.item(), iteration)
                    self.writer.add_scalar('edge_similarity/recall', recall.item(), iteration)

                    self.writer.add_scalar('texture_similarity/psnr', psnr.item(), iteration)



    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            drop_last=True,
            shuffle=True
        )

        model = self.config.MODEL
        total = len(self.val_dataset)

        self.model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0

        for items in val_loader:
            iteration += 1
            images, images_gray, edges, masks, grads = self.cuda(*items)

            # eval
            outputs, gen_loss, dis_loss, losses, logs, img_64, grad_64, img_128, grad_128, img_256, grad_256 = self.model.process(images, edges, masks, grads)

            # metrics
            precision, recall = self.edgeacc(edges * masks, outputs * masks)
            outputs_merged = (outputs[0] * masks) + (images * (1 - masks))
            psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
            mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
            self.writer.add_scalar('edge_similarity/precision', precision.item(), self.model.iteration)
            self.writer.add_scalar('edge_similarity/recall', recall.item(), self.model.iteration)
            self.writer.add_scalar('texture_similarity/psnr', psnr.item(), self.model.iteration)
            self.writer.add_scalar('texture_similarity/mae', mae.item(), self.model.iteration)

            logs = [("it", iteration), ] + logs
            progbar.add(len(images), values=logs)

    def test(self):
        self.model.eval()

        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0

        time_start = time.time()

        for items in test_loader:
            name = self.test_dataset.load_name(index)
            images, images_gray, edges, masks, grads = self.cuda(*items)

            # grads_input = images.cpu().detach().numpy()
            # grads_input = np.squeeze(grads_input, axis=0)
            # grads = self.test_dataset.load_grad(np.transpose(grads_input, (1, 2, 0)))
            # grads = F.to_tensor(grads).float().unsqueeze(0).cuda()

            edges_GT = self.postprocess(edges)[0]
            index += 1


            outputs, x_out_64, x_out_128, x_grad, x_grad_out_64, x_grad_out_128 = self.model(images, edges, masks, grads)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

            output = self.postprocess(outputs_merged)[0]
            name = str(index).zfill(5) + '.png'
            path = os.path.join(self.results_path, name)
            print(index, name)

            imsave(output, path)

            if self.debug:
                edges = self.postprocess(edges)[0]
                masked = self.postprocess(images * (1 - masks) + masks)[0]
                masks = self.postprocess(masks)[0]
                fname, fext = name.split('.')

                imsave(edges, os.path.join(self.results_path, fname + '_edge.' + fext))
                imsave(masks, os.path.join(self.results_path, fname + '_mask.' + fext))
                imsave(masked, os.path.join(self.results_path, fname + '_masked.' + fext))
                imsave(edges_GT, os.path.join(self.results_path, fname + '_edges_GT.' + fext))

        time_end = time.time()
        print('time cost', time_end - time_start, 's')
        print('\nEnd test....')

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.model.eval()

        items = next(self.sample_iterator)
        images, images_gray, edges, masks, grads= self.cuda(*items)

        iteration = self.model.iteration
        inputs = (images * (1 - masks)) + masks
        outputs, x_out_64, x_out_128, x_grad, x_grad_out_64, x_grad_out_128 = self.model(images, edges, masks, grads)
        outputs_merged = (outputs * masks) + (images * (1 - masks))
        # grad_256_merged = (x_grad * masks) + (grads * (1 - masks))

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1


        images = stitch_images(
            self.postprocess(images),
            self.postprocess(inputs),
            self.postprocess(outputs),
            self.postprocess(outputs_merged),
            img_per_row = image_per_row
        )


        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)


    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = (img + 1.) * 127.5
        img = img.permute(0, 2, 3, 1)
        return img.int()