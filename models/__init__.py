import torch.nn as nn
from torchvision.models import vgg19_bn, resnet50

from .inceptionresnetv2 import *
from .preresnet import *


def get_model(dataset):
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, 0, 0.001)
            nn.init.constant_(m.bias, 0)

    if dataset == 'clothing1m':
        encoder = resnet50(pretrained=True)
        classifier = nn.Linear(encoder.fc.in_features, 14)
        encoder.fc = nn.Identity()
        classifier.apply(init_weights)
        encoder.cuda()
        classifier.cuda()
        num_classes = 14
    elif dataset == 'cifar10' or dataset == 'cifar10-IND':
        encoder = PreResNet18(10)
        classifier = nn.Linear(encoder.fc.in_features, 10)
        encoder.fc = nn.Identity()
        encoder.cuda()
        classifier.cuda()
        num_classes = 10
    elif dataset == 'cifar100':
        encoder = PreResNet18(100)
        classifier = nn.Linear(encoder.fc.in_features, 100)
        encoder.fc = nn.Identity()
        encoder.cuda()
        classifier.cuda()
        num_classes = 100
    elif dataset == 'animal10n':
        encoder = vgg19_bn(pretrained=False)
        encoder.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
        )
        encoder.classifier.apply(init_weights)
        classifier = nn.Linear(4096, 10)
        encoder.cuda()
        classifier.cuda()
        num_classes = 10
    elif dataset == 'webvision':
        encoder = InceptionResNetV2()
        # encoder = resnet18(num_classes=args.num_classes)
        classifier = nn.Linear(encoder.fc.in_features, 50)
        encoder.fc = nn.Identity()
        encoder.cuda()
        classifier.cuda()
        num_classes = 50
    else:  # dataset == 'redimagenet
        # pass  # tobe done
        encoder = PreResNet18(100)
        classifier = nn.Linear(encoder.fc.in_features, 100)
        encoder.fc = nn.Identity()
        encoder.cuda()
        classifier.cuda()
        num_classes = 100

    return encoder, classifier, num_classes
