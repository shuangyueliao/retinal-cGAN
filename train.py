import os, sys
import time
import argparse
import torch.optim as optim
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from dataset import VOC2012Dataset
from cgan import discriminator, Generator
import logging
import cv2
from skimage.transform import rotate

VOC2012_PATH = './VOC2012'

""" Usage
python -m visdom.server
python train.py --model=fcn32s --nb_epoch=100 --batch_size=2 --nb_worker=2 --l_rate=1e-4
python train.py --model=segnet --nb_epoch=100 --batch_size=5 --nb_worker=4 --l_rate=6e-4
"""

""" references
https://meetshah1995.github.io/semantic-segmentation/deep-learning/pytorch/visdom/2017/06/01/semantic-segmentation-over-the-years.html
"""


def hiimage(img, opts=None):
    """
    This function draws an img. It takes as input an `CxHxW` or `HxW` tensor
    `img` that contains the image. The array values can be float in [0,1] or
    uint8 in [0, 255].
    """
    opts = {} if opts is None else opts
    opts['jpgquality'] = opts.get('jpgquality', 75)
    opts['width'] = opts.get('width', img.shape[img.ndim - 1])
    opts['height'] = opts.get('height', img.shape[img.ndim - 2])

    nchannels = img.shape[0] if img.ndim == 3 else 1
    if nchannels == 1:
        img = np.squeeze(img)
        img = img[np.newaxis, :, :].repeat(3, axis=0)

    if 'float' in str(img.dtype):
        if img.max() <= 1:
            img = img * 255.
        img = np.uint8(img)

    img = np.transpose(img, (1, 2, 0))
    im = Image.fromarray(img)
    im.save('./tmp.jpg')


def train(args):
    print(args)

    ds_train = VOC2012Dataset(VOC2012_PATH, set='train')

    loader_train = data_utils.DataLoader(ds_train,
                                         batch_size=args.batch_size,
                                         num_workers=1, shuffle=True)
    G = Generator(64)
    D = discriminator(64)
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
    G.cuda()
    D.cuda()
    G.train()
    D.train()
    BCE_Loss = nn.BCELoss().cuda()
    L1_Loss = nn.L1Loss().cuda()

    print("init_params done.")

    if not os.path.exists("./models"):
        os.mkdir("./models")

    G_optimizer = optim.Adam(G.parameters(), lr=opt.lrG, betas=(opt.beta1, opt.beta2))
    D_optimizer = optim.Adam(D.parameters(), lr=opt.lrD, betas=(opt.beta1, opt.beta2))

    real_a = torch.FloatTensor(1, 3, 256, 256)
    real_b = torch.FloatTensor(1, 3, 256, 256)

    real_a = real_a.cuda()
    real_b = real_b.cuda()
    real_a = Variable(real_a)
    real_b = Variable(real_b)

    for epoch in range(opt.train_epoch):
        D_losses = []
        G_losses = []
        epoch_start_time = time.time()
        print('training epoch {}'.format(epoch + 1))
        for i, (realA, realB) in enumerate(loader_train):
            real_a_cpu, real_b_cpu = realA, realB
            real_a.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)
            real_b.data.resize_(real_b_cpu.size()).copy_(real_b_cpu)

            D.zero_grad()
            D_result = D(real_a, real_b).squeeze()
            D_real_loss = BCE_Loss(D_result, torch.Tensor(torch.ones(D_result.size())).cuda())

            G_result = G(real_a)
            D_result = D(real_a, G_result).squeeze()
            D_fake_loss = BCE_Loss(D_result, torch.Tensor(torch.zeros(D_result.size())).cuda())

            D_train_loss = (D_real_loss + D_fake_loss) * 0.5
            D_train_loss.backward()
            D_optimizer.step()

            G.zero_grad()
            G_result = G(real_a)
            D_result = D(real_a, G_result).squeeze()
            plt.imsave('result.png', (G_result[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
            G_train_loss = BCE_Loss(D_result, torch.Tensor(torch.ones(D_result.size())).cuda()) + opt.L1_lambda * L1_Loss( G_result, real_b)
            G_train_loss.backward()
            G_optimizer.step()
        # if (epoch % 50 == 0):
        #     torch.save(model, "./models/fcn8s{}.pkl".format(epoch))
        #     torch.save(netG, "./models/G{}.pkl".format(epoch))
        #     torch.save(netD, './models/D{}.pkl'.format(epoch))

    torch.save(G, "./models/G.pkl")
    torch.save(D, './models/D.pkl')


def xcross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss


if __name__ == '__main__':
    logging.basicConfig(filename='loss.log', level=logging.INFO, filemode='w')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=False, default='train_img', help='')
    parser.add_argument('--train_subfolder', required=False, default='combine', help='')
    parser.add_argument('--test_subfolder', required=False, default='test', help='')
    parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
    parser.add_argument('--test_batch_size', type=int, default=5, help='test batch size')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    # parser.add_argument('--input_size', type=int, default=256, help='input size')
    # parser.add_argument('--crop_size', type=int, default=256, help='crop size (0 is false)')
    # parser.add_argument('--resize_scale', type=int, default=286, help='resize scale (0 is false)')
    # parser.add_argument('--fliplr', type=bool, default=True, help='random fliplr True or False')
    parser.add_argument('--train_epoch', type=int, default=100, help='number of train epochs')
    parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--L1_lambda', type=float, default=100, help='lambda for L1 loss')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--save_root', required=False, default='results', help='results save path')
    parser.add_argument('--inverse_order', type=bool, default=True, help='0: [input, target], 1 - [target, input]')
    opt = parser.parse_args()

    train(opt)

### EOF ###
