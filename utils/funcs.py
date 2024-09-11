from __future__ import print_function

import os
import random

import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def save_config(args, path):
    dict = vars(args)
    if not os.path.isdir(path):
        os.mkdir(os.path.abspath(path))
    with open(path + '/params.csv', 'w') as f:
        for key in dict.keys():
            f.write("%s\t%s\n" % (key, dict[key]))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
