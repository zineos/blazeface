from __future__ import print_function

import sys
sys.path.append("/data/face_detections/blazefacev3/config")
sys.path.append("/data/face_detections/blazefacev3/blazeface")

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data.dataset.wider_face import WiderFaceDetection, detection_collate
from data.transform.data_augment import preproc
from config import cfg_mnet, cfg_slim, cfg_rfb, cfg_blaze
from models.loss import MultiBoxLoss
from models.loss.custom_loss import MultiBoxLoss as CustomLoss
from models.loss.custom_loss2 import MultiBoxLoss as CustomLoss2
from models.module.prior_box import PriorBox
from apex import amp
import time
import datetime
import math
from models.retinaface import RetinaFace
from models.net_slim import Slim
from models.net_rfb import RFB
from models.net_blaze import Blaze
from utils.data_loader import data_prefetcher, FastDataLoader
import logging

LOG_FORMAT = "%(asctime)s %(levelname)s : %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--training_dataset', default='/data/face_detections/face_dataset/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--network', default='Blaze', help='Backbone network mobile0.25 or slim or RFB')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='/data/face_detections/blazefacev3/weights/train/', help='Location to save checkpoint models')

args = parser.parse_args()


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
cfg = None
net = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
    net = RetinaFace(cfg=cfg)
elif args.network == "slim":
    cfg = cfg_slim
    net = Slim(cfg=cfg)
elif args.network == "RFB":
    cfg = cfg_rfb
    net = RFB(cfg=cfg)
elif args.network == "Blaze":
    cfg = cfg_blaze
    net = Blaze(cfg=cfg)
else:
    print("Don't support network!")
    exit(0)

print("Printing net...")
print(net)

rgb_mean = (104, 117, 123) # bgr order
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
save_folder = args.save_folder

if args.resume_net is not None:
    logging.info('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if num_gpu >= 1 and gpu_train:
    net = torch.nn.DataParallel(net).cuda()

else:
    net = net.cuda()

cudnn.benchmark = True


# optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
optimizer = optim.RMSprop(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
# criterion = MultiBoxLoss(cfg, 0.35, True, 0, True, 7, 0.35, False)
criterion = CustomLoss2(cfg, 0.35, True, 0, True, 7, 0.35, False)


# fp16 training
# net, optimizer = amp.initialize(net, optimizer, opt_level="O2")


priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()

def train():
    net.train()
    epoch = 0 + args.resume_epoch
    logging.info('Loading Dataset...')

    dataset = WiderFaceDetection(training_dataset,preproc(img_dim, rgb_mean))

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, 
                  cfg['decay2'] * epoch_size, 
                  cfg['decay3'] * epoch_size,
                  cfg['decay4'] * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    device = 0

    loader = data.DataLoader(dataset, batch_size, 
                            shuffle=True, num_workers=num_workers, 
                            collate_fn=detection_collate, pin_memory=True)

    # loader = FastDataLoader(dataset, batch_size, 
    #                         shuffle=True, num_workers=num_workers, 
    #                         collate_fn=detection_collate, pin_memory=True)

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = data_prefetcher(loader, device)
            # batch_iterator = iter(loader)
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                torch.save(net.state_dict(), save_folder + cfg['name']+ '_epoch_' + str(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = batch_iterator.next()

        # images, targets = next(batch_iterator)
        # images = images.cuda()
        # targets = [anno.cuda() for anno in targets]

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss =  loss_l + loss_c + loss_landm

        ## fp32 training
        loss.backward()

        ## fp16 training
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        
        optimizer.step()

        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration)) 
        logging.info('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
              .format(epoch, max_epoch, (iteration % epoch_size) + 1,
              epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))

    torch.save(net.state_dict(), save_folder + cfg['name'] + '_Final.pth')

def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = 1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    train()
