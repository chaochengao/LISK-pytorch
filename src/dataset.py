import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torch.nn import functional as F
from PIL import Image
from scipy.misc import imread
from scipy import signal
import skimage
from skimage import util, filters
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from .utils import create_mask


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, edge_flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.edge_data = self.load_flist(edge_flist)
        self.mask_data = self.load_flist(mask_flist)

        self.input_size = config.INPUT_SIZE
        self.sigma = config.SIGMA
        self.edge = config.EDGE
        self.mask = config.MASK

        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random
        if config.MODE == 2:
            self.mask = 6

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size

        # load image
        img = imread(self.data[index])

        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)

        # resize/crop if needed
        if size != 0:
            img = self.resize(img, size, size)

        # create grayscale image
        img_gray = rgb2gray(img)

        # load mask
        mask = self.load_mask(img, index)

        # load edge
        edge = self.load_edge(img_gray, index, mask)

        # load grad
        img_norm = img.astype(np.float32) / 127.5 - 1
        grad = self.load_grad(img_norm)

        # # augment data
        # if self.augment and np.random.binomial(1, 0.5) > 0:
        #     img = img[:, ::-1, ...]
        #     img_gray = img_gray[:, ::-1, ...]
        #     edge = edge[:, ::-1, ...]
        #     mask = mask[:, ::-1, ...]

        # print(type(img_norm), type(img_gray), type(edge), type(mask), type(grad))
        return self.to_tensor(img_norm), self.to_tensor(img_gray), self.to_tensor(edge), self.to_tensor(mask), \
               grad

    def load_edge(self, img, index, mask):
        sigma = self.sigma

        # in test mode images are masked (with masked regions),
        # using 'mask' parameter prevents canny to detect edges for the masked regions
        mask = None if self.training else (1 - mask / 255).astype(np.bool)

        # canny
        if self.edge == 1:
            # no edge
            if sigma == -1:
                return np.zeros(img.shape).astype(np.float32)

            # random sigma
            if sigma == 0:
                sigma = random.randint(1, 4)


            return canny(img, sigma=sigma).astype(np.float32)

        # external
        else:
            imgh, imgw = img.shape[0:2]
            edge = imread(self.edge_data[index])
            edge = rgb2gray(edge)
            edge = self.resize(edge, imgh, imgw)

            return edge

    def load_grad(self, img):
        kernels = np.array([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                   [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]).astype(np.float32)

        img_t = self.to_tensor(img)

        img_t = torch.cat((img_t, img_t), dim=0)
        k1 = torch.from_numpy(kernels[0])
        k2 = torch.from_numpy(kernels[1])
        k1 = k1.unsqueeze(0)
        k2 = k2.unsqueeze(0)
        # print(img_t[0, 250, 0:10])
        k = torch.cat((k1, k1, k1, k2, k2, k2), dim=0)
        k = k.unsqueeze(1)
        img_t = img_t.unsqueeze(0)
        w = torch.nn.Parameter(data=k, requires_grad = False)

        img_t = F.pad(input=img_t, pad=[1, 1, 1, 1], mode='reflect')
        grad = F.conv2d(img_t, w, stride=1, padding=0, groups=6)
        grad = grad.squeeze(0)

        return grad[[0, 3, 1, 4, 2, 5], ...]


    # def load_grad(self, img):
    #     kernels = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    #                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
    #
    #     grad_1 = self.convolve2D(img[:, :, 0], kernel=np.asarray(kernels)[0], padding=1)
    #     grad_2 = self.convolve2D(img[:, :, 1], kernel=np.asarray(kernels)[0], padding=1)
    #     grad_3 = self.convolve2D(img[:, :, 2], kernel=np.asarray(kernels)[0], padding=1)
    #     grad_4 = self.convolve2D(img[:, :, 0], kernel=np.asarray(kernels)[1], padding=1)
    #     grad_5 = self.convolve2D(img[:, :, 1], kernel=np.asarray(kernels)[1], padding=1)
    #     grad_6 = self.convolve2D(img[:, :, 2], kernel=np.asarray(kernels)[1], padding=1)
    #     grad = np.dstack((grad_1, grad_4, grad_2, grad_5, grad_3, grad_6))
    #
    #     return grad

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask

        # external + random block
        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # external + random block + half
        elif mask_type == 5:
            mask_type = np.random.randint(1, 4)

        # random block
        if mask_type == 1:
            return create_mask(imgw, imgh, imgw // 3, imgh // 3)

        # half
        if mask_type == 2:
            # randomly choose right or left
            return create_mask(imgw, imgh, imgw // 2, imgh, 0 if random.random() < 0.5 else imgw // 2, 0)

        # external
        if mask_type == 3:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imread(self.mask_data[mask_index])
            mask = self.resize(mask, imgh, imgw)
            mask = (mask > 0).astype(np.uint8) * 255       # threshold due to interpolation
            return mask

        # test mode: load mask non random
        if mask_type == 6:
            mask = imread(self.mask_data[index])
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            mask = rgb2gray(mask)
            # mask = (mask > 0).astype(np.uint8) * 255
            return mask

    def to_tensor(self, img):
        if img.ndim == 2:
            img = img[np.newaxis, :]
            img_t = torch.from_numpy(img).float()
        else:
            img_t = torch.from_numpy(img).float().permute(2, 0, 1)
        # print(img_t.size())
        return img_t

    def to_tensor_norm(self, img):
        img = img.astype(np.float32) / 127.5 - 1  # scale to [-1, 1]
        img_t = torch.from_numpy(img)
        if img_t.dim() == 3:
            img_t = img_t.float().permute(2, 0, 1)
        elif img_t.dim() == 2:
            img_t = img_t.float().unsqueeze(0)

        return img_t


    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = scipy.misc.imresize(img, [height, width])

        return img

    def convolve2D(self, image, kernel, padding=1, strides=1):
        # Cross Correlation
        kernel = np.flipud(np.fliplr(kernel))

        # Gather Shapes of Kernel + Image + Padding
        xKernShape = kernel.shape[0]
        yKernShape = kernel.shape[1]
        xImgShape = image.shape[0]
        yImgShape = image.shape[1]

        # Shape of Output Convolution
        xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
        yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
        output = np.zeros((xOutput, yOutput))

        # Apply Equal Padding to All Sides
        if padding != 0:
            imagePadded = np.pad(image, ((padding, padding), (padding, padding)), mode='reflect')
            # imagePadded = np.zeros((image.shape[0] + padding * 2, image.shape[1] + padding * 2))
            # imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
            # print(imagePadded)
        else:
            imagePadded = image

        # Iterate through image
        for y in range(image.shape[1]):
            # Exit Convolution
            if y > image.shape[1] - yKernShape:
                break
            # Only Convolve if y has gone down by the specified Strides
            if y % strides == 0:
                for x in range(image.shape[0]):
                    # Go to next row once kernel is out of bounds
                    if x > image.shape[0] - xKernShape:
                        break
                    try:
                        # Only Convolve if x has moved by the specified Strides
                        if x % strides == 0:
                            output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                    except:
                        break

        return -1*output

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item
