import os,sys
import argparse

import numpy as np

import torch, visdom
import torch.nn.functional as F
import torch.utils.data as data_utils

from torch.autograd import Variable

import torchvision.models as models

from dataset import VOC2012Dataset
from model import FCN8s, FCN16s, FCN32s

VOC2012_PATH='./VOC2012'
#VOC2012_PATH='/home/masao/voc2012/VOC2012'

""" Usage
python -m visdom.server
python train.py --model=fcn32s --nb_epoch=100 --batch_size=2 --nb_worker=2 --l_rate=1e-4
python train.py --model=segnet --nb_epoch=100 --batch_size=5 --nb_worker=4 --l_rate=6e-4
"""

""" references
https://meetshah1995.github.io/semantic-segmentation/deep-learning/pytorch/visdom/2017/06/01/semantic-segmentation-over-the-years.html
"""

def train(args):
    print(args)

    ds_train = VOC2012Dataset(VOC2012_PATH, set='train')
    ds_val   = VOC2012Dataset(VOC2012_PATH, set='val')

    loader_train = data_utils.DataLoader(ds_train,
                                            batch_size=args.batch_size,
                                            num_workers=args.nb_worker)

    vis = visdom.Visdom()

    if args.model == 'fcn8s':
        model = FCN8s()
    elif args.model == 'fcn16s':
        model = FCN16s()
    elif args.model == 'fcn32s':
        model = FCN32s()

    vgg16 = models.vgg16(pretrained=True)
    model.init_params(vgg16)
    print ("init_params done.")

    if torch.cuda.is_available():
        model.cuda(0)

        img_test, segmap_test = ds_train[1]
        img_test = Variable(img_test.unsqueeze(0).cuda(0))

        img_val, segmap_val = ds_val[0]
        img_val = Variable(img_val.unsqueeze(0).cuda(0))
    else:
        print ("cuda required.")
        sys.exit(0)

    if not os.path.exists("./models"):
        os.mkdir ("./models")

    optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.nb_epoch):


        for i, (images, gts) in enumerate(loader_train):
            if torch.cuda.is_available():
                images = Variable(images.cuda(0))
                gts    = Variable(gts.cuda(0))
            else:
                images = Variable(images)
                gts    = Variable(gts)
            print(images.data.size())
            print(gts.data.size())
            optimizer.zero_grad()
            outputs = model(images)

            loss = torch.nn.CrossEntropyLoss()(outputs, gts)
            #loss = cross_entropy2d(outputs, gts)

            loss.backward()
            optimizer.step()



            if (i+1) % 50 == 0:
                print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, args.nb_epoch, loss.data[0]))

        if True:
        #if (epoch % 10 == 0) or (epoch == 99):
            output_test = model(img_test)
            predicted   = ds_train.decode_segmap(output_test[0].cpu().data.numpy().argmax(0))
            target      = ds_train.decode_segmap(segmap_test.numpy())

            vis.image(img_test[0].cpu().data.numpy(),   opts=dict(title='Train Input - e' + str(epoch)))
            vis.image(np.transpose(target, [2,0,1]),    opts=dict(title='Train GT - e' + str(epoch)))
            vis.image(np.transpose(predicted, [2,0,1]), opts=dict(title='Train Predicted - e' + str(epoch)))

            output_val    = model(img_val)
            predicted_val = ds_val.decode_segmap(output_val[0].cpu().data.numpy().argmax(0))
            target_val    = ds_val.decode_segmap(segmap_val.numpy())

            vis.image(img_val[0].cpu().data.numpy(),        opts=dict(title='Val Input - e' + str(epoch)))
            vis.image(np.transpose(target_val, [2,0,1]),    opts=dict(title='Val GT - e' + str(epoch)))
            vis.image(np.transpose(predicted_val, [2,0,1]), opts=dict(title='Val Predicted - e' + str(epoch)))

            torch.save(model, "./models/{}.e{}.pkl".format(args.model, epoch))

    torch.save(model, "./models/{}.pkl".format(args.model))


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
    parser = argparse.ArgumentParser(description='FCN for VOC2012')
    parser.add_argument('--model', type=str, default='fcn16s',
                        help='model to use: fcn8s, fcn16s, fcn32s')
    parser.add_argument('--nb_epoch', type=int, default=100,
                        help='# of epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch_size')
    parser.add_argument('--nb_worker', type=int, default=4,
                        help='# of workers')
    parser.add_argument('--lr', type=float, default=1e-4, #5e-5,
                        help='Learning Rate')
    parser.add_argument('--weight_decay', type=float, default=5.0e-7,
                        help='weight decay')
    args = parser.parse_args()

    train(args)


### EOF ###