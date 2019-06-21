import argparse
import torch
import torch.utils.data as data_utils
from torch.autograd import Variable
import matplotlib.pyplot as plt
from dataset import VOC2012Dataset
from cgan import Generator
import logging
import os

VOC2012_PATH = './VOC2012'


def train(args):
    print(args)

    ds_val = VOC2012Dataset(VOC2012_PATH, set='val')

    loader_val = data_utils.DataLoader(ds_val,
                                         batch_size=args.batch_size,
                                         num_workers=1, shuffle=False)
    G = torch.load("./models/G.pkl")
    G.cuda()

    print("init_params done.")
    real_a = torch.FloatTensor(1, 3, 256, 256)

    real_a = real_a.cuda()
    real_a = Variable(real_a)

    if not os.path.exists("./test"):
        os.mkdir("./test")

    for i, (base_name, realA) in enumerate(loader_val):
        real_a_cpu = realA
        real_a.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)
        G_result = G(real_a)
        plt.imsave('./test/' + base_name[0], (G_result[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

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
    parser.add_argument('--train_epoch', type=int, default=50, help='number of train epochs')
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
