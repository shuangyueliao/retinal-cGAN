import os,sys
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
import cv2
from skimage.transform import rotate
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
def hiimage(img,filename,opts=None):
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
    im.save('./test/'+filename)
def test(args):
    print(args)

    ds_test = VOC2012Dataset(VOC2012_PATH, set='val')

    loader_test = data_utils.DataLoader(ds_test,
                                            batch_size=args.batch_size,
                                            num_workers=args.nb_worker)

    print ("init_params done.")

    if torch.cuda.is_available():
        #model=torch.load("./models/{}.pkl".format(args.model))
        model = torch.load("./models/fcn8s.pkl")
        model.cuda(0)
    else:
        print ("cuda required.")
        sys.exit(0)

    if not os.path.exists("./test"):
        os.mkdir ("./test")

    for i, (images,realA,filename) in enumerate(loader_test):
        images = Variable(images.cuda(0))
        outputs = model(images)
        predicted = ds_test.decode_segmap(outputs[0].cpu().data.numpy().argmax(0))
        #predicted = cv2.linearPolar(rotate(predicted, 90), (256 / 2, 256 / 2), 256 / 2,cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)  # 极坐标转直角坐标
        tmp = np.transpose(predicted, [2, 0, 1])
        print(''.join(filename))
        hiimage(tmp,''.join(filename))



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
    parser.add_argument('--model', type=str, default='fcn8s',
                        help='model to use: fcn8s, fcn16s, fcn32s')
    parser.add_argument('--nb_epoch', type=int, default=200,
                        help='# of epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch_size')
    parser.add_argument('--nb_worker', type=int, default=1,
                        help='# of workers')
    parser.add_argument('--lr', type=float, default=1e-4, #5e-5,
                        help='Learning Rate')
    parser.add_argument('--weight_decay', type=float, default=5.0e-7,
                        help='weight decay')
    args = parser.parse_args()

    test(args)


### EOF ###

