import os, sys
import argparse
import torch.optim as optim
import numpy as np
from PIL import Image
import torch, visdom
import torch.nn.functional as F
import torch.utils.data as data_utils
import torch.nn as nn
from torch.autograd import Variable
from util import is_image_file, load_img, save_img
import torchvision.models as models
import torchvision.transforms as transforms
from dataset import VOC2012Dataset
from model import FCN8s, FCN16s, FCN32s
from networks import define_G, define_D, GANLoss, print_network
import logging
import cv2
from skimage.transform import rotate

VOC2012_PATH = './VOC2012'
# VOC2012_PATH='/home/masao/voc2012/VOC2012'

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
                                         num_workers=args.nb_worker, shuffle=True)
    vis = visdom.Visdom()
    netG = define_G(3, 3, 64, 'batch', False, [0])
    netD = define_D('noReLU', 6, 64, 'batch', False, [0])
    criterionGAN = GANLoss()
    criterionL1 = nn.L1Loss()
    # criterionMSE = nn.MSELoss()

    if args.model == 'fcn8s':
        model = FCN8s()
    elif args.model == 'fcn16s':
        model = FCN16s()
    elif args.model == 'fcn32s':
        model = FCN32s()

    print(netD)

    vgg16 = models.vgg16(pretrained=True)

    model.init_params(vgg16)

    print("init_params done.")

    if torch.cuda.is_available():
        #        model=torch.load('./models/fcn8s.pkl')
        model.cuda(0)

        img_test, segmap_test, _, _, _ = ds_train[1]
        img_test = Variable(img_test.unsqueeze(0).cuda(0))
    else:
        print("cuda required.")
        sys.exit(0)

    if not os.path.exists("./models"):
        os.mkdir("./models")

    optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

    real_a = torch.FloatTensor(1, 3, 256, 256)
    real_b = torch.FloatTensor(1, 3, 256, 256)
    #    netD=torch.load('./models/D.pkl')
    #    netG=torch.load('./models/G.pkl')
    netD = netD.cuda()
    netG = netG.cuda()
    criterionGAN = criterionGAN.cuda()
    criterionL1 = criterionL1.cuda()
    # criterionMSE = criterionMSE.cuda()
    real_a = real_a.cuda()
    real_b = real_b.cuda()
    real_a = Variable(real_a)
    real_b = Variable(real_b)
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform_list)
    for epoch in range(args.nb_epoch):
        FCN_losses = []
        D_losses = []
        G_losses = []
        for i, (images, gts, realA, realB, status) in enumerate(loader_train):
            if status[0] == 1:
                images = Variable(images.cuda(0))
                gts = Variable(gts.cuda(0))
                real_a_cpu, real_b_cpu = realA, realB
                real_a.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)
                real_b.data.resize_(real_b_cpu.size()).copy_(real_b_cpu)
                outputs = model(images)
                predicted = ds_train.decode_segmap(outputs[0].cpu().data.numpy().argmax(0))
                tmp = np.transpose(predicted, [2, 0, 1])
                hiimage(tmp)
                fake_b = load_img('./tmp.jpg')
                fake_b = transform(fake_b)
                fake_b = Variable(fake_b.cuda(0))
                fake_b = torch.unsqueeze(fake_b, 0)
                fake_a = netG(real_b)
                ###########################
                optimizerD.zero_grad()
                # train with fake
                fake_ab = torch.cat((real_a, fake_b), 1)  #
                pred_fake = netD.forward(fake_ab.detach())  #
                loss_d_fake = criterionGAN(pred_fake, False)  #
                # train with fake
                fake_ab1 = torch.cat((fake_a, real_b), 1)
                pred_fake1 = netD.forward(fake_ab1.detach())
                loss_d_fake1 = criterionGAN(pred_fake1, False)
                # train with real
                real_ab = torch.cat((real_a, real_b), 1)
                pred_real = netD.forward(real_ab)
                loss_d_real = criterionGAN(pred_real, True)
                # Combined loss
                loss_d = (loss_d_fake + loss_d_real + loss_d_fake1) / 3
                D_losses.append(loss_d.data[0])
                loss_d.backward()
                optimizerD.step()
                ############################
                optimizer.zero_grad()
                realA_fakeB = torch.cat((real_a, fake_b), 1)  #
                realA_fakeB_result = netD.forward(realA_fakeB)  #
                loss1 = torch.nn.CrossEntropyLoss()(outputs, gts)
                loss2 = criterionGAN(realA_fakeB_result, True)  #
                loss = loss1 + loss2
                FCN_losses.append(loss.data[0])
                loss.backward()
                optimizer.step()
                ##################################
                optimizerG.zero_grad()
                fake_ab = torch.cat((fake_a, real_b), 1)
                pred_fake = netD.forward(fake_ab)
                loss_g_gan = criterionGAN(pred_fake, True)
                loss_g_l1 = criterionL1(fake_a, real_a) * 10
                loss_g = loss_g_gan + loss_g_l1
                loss_g.backward()
                G_losses.append(loss_g.data[0])
                optimizerG.step()
                out = fake_a.cpu()
                out_img = out.data[0]
                save_img(out_img, "result_a.jpg")
                ######################################
            else:
                images = Variable(images.cuda(0))
                real_a_cpu = realA
                real_a.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)
                outputs = model(images)
                predicted = ds_train.decode_segmap(outputs[0].cpu().data.numpy().argmax(0))
                tmp = np.transpose(predicted, [2, 0, 1])
                hiimage(tmp)
                fake_b = load_img('./tmp.jpg')
                fake_b = transform(fake_b)
                fake_b = Variable(fake_b.cuda(0))
                fake_b = torch.unsqueeze(fake_b, 0)
                ###########################
                optimizerD.zero_grad()
                # train with fake
                fake_ab = torch.cat((real_a, fake_b), 1)  #
                pred_fake = netD.forward(fake_ab.detach())  #
                loss_d_fake = criterionGAN(pred_fake, False)  #
                loss_d = loss_d_fake
                loss_d.backward()
                optimizerD.step()

        print(epoch)

        if False:
            output_test = model(img_test)
            predicted = ds_train.decode_segmap(output_test[0].cpu().data.numpy().argmax(0))
            # predicted = cv2.linearPolar(rotate(predicted, 90), (256 / 2, 256 / 2), 256 / 2,cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)  # 极坐标转直角坐标
            target = ds_train.decode_segmap(segmap_test.numpy())
            # target = cv2.linearPolar(rotate(target, 90), (256 / 2, 256 / 2), 256 / 2,cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)  # 极坐标转直角坐标

            vis.image(img_test[0].cpu().data.numpy(), opts=dict(title='Train Input - e' + str(epoch)))
            vis.image(np.transpose(target, [2, 0, 1]), opts=dict(title='Train GT - e' + str(epoch)))
            vis.image(np.transpose(predicted, [2, 0, 1]), opts=dict(title='Train Predicted - e' + str(epoch)))

        # if (epoch % 50 == 0):
        #     torch.save(model, "./models/fcn8s{}.pkl".format(epoch))
        #     torch.save(netG, "./models/G{}.pkl".format(epoch))
        #     torch.save(netD, './models/D{}.pkl'.format(epoch))

    torch.save(model, "./models/{}.pkl".format(args.model))
    torch.save(netG, "./models/G.pkl")
    torch.save(netD, './models/D.pkl')


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
    """
    Adamax default : lr=2e-3 weight_decay=0
    FCN : --lr=1e-4 --weight_decay=5.0e-7
    """
    logging.basicConfig(filename='loss.log', level=logging.INFO, filemode='w')
    parser = argparse.ArgumentParser(description='FCN for VOC2012')
    parser.add_argument('--model', type=str, default='fcn8s',
                        help='model to use: fcn8s, fcn16s, fcn32s')
    parser.add_argument('--nb_epoch', type=int, default=200,
                        help='# of epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch_size')
    parser.add_argument('--nb_worker', type=int, default=1,
                        help='# of workers')
    parser.add_argument('--lr', type=float, default=1e-4,  # 5e-5,
                        help='Learning Rate')
    parser.add_argument('--weight_decay', type=float, default=5.0e-7,
                        help='weight decay')
    args = parser.parse_args()

    train(args)

### EOF ###

