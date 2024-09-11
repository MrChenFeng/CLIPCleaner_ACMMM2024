'''
Simplified main file for CIFAR/Animal10n/RedImageNet.
- wandb removed
- hyperparameters default to CIFAR dataset

Chen Feng
2024.09.10
'''
import argparse
import os.path

from torch.optim import SGD
from torch.utils.data import Subset
from tqdm import tqdm
import torch.nn.functional as F

from clipcleaner import combined_selection
from datasets import *
from models import *
from utils import *

# dataset settings
parser = argparse.ArgumentParser('Main trainer')
parser.add_argument('--dataset', default='cifar10',
                    choices=['animal10n', 'clothing1m', 'red_imagenet', 'cifar10', 'cifar100', 'webvision',
                             'cifar10-IND'], help='dataset')
parser.add_argument('--dataset_path', default='./datasets/CIFAR', help='dataset path')

#######################################################################################################################
# synthetic noise modes: for some datasets only
parser.add_argument('--noise_mode', type=str, default='sym', help='noise mode')
parser.add_argument('--noise_ratio', type=float, default=0.5, help='noise ratio')
parser.add_argument('--type', type=str, default='red', help='noise mode of mini-imagenet')
#######################################################################################################################

# sample selection threshold
parser.add_argument('--theta_r', type=float, default=0.8, help='absorbing threshold for mixfix')
parser.add_argument('--theta_r2', type=float, default=0.8, help='relabelling threshold for mixfix')
parser.add_argument('--theta_gmm', type=float, default=0.5, help='selection threshold for GMM')
parser.add_argument('--theta_cons', type=float, default=0.8, help='selection threshold for Consistency')

# miscellaneous settings
parser.add_argument('--beta', default=1, type=float, help='beta of mixup interpolation (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run (default: 300)')
parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--lr', default=0.02, type=float, help='initial learning rate (default: 0.02)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver (default: 0.9)')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--gamma', type=float, default=0.1, help='Multiplicative factor of learning rate decay')
parser.add_argument('--milestones', type=int, nargs='+', default=[200, 250],
                    help='List of epoch indices. Must be increasing.')
parser.add_argument('--seed', default=3047, type=int, help='seed for initializing training. (default: 3047)')
parser.add_argument('--gpuid', default='0', type=str, help='Selected GPU (default: "0")')
parser.add_argument('--run_path', type=str, help='run path containing all results')


def train(labeled_trainloader, updated_label, encoder, classifier, optimizer, epoch, args, num_iters):
    encoder.train()
    classifier.train()
    xlosses = AverageMeter('xloss')
    all_bar = tqdm(range(num_iters))
    labeled_train_iter = iter(labeled_trainloader)
    for batch_idx in enumerate(all_bar):
        try:
            inputs_x, labels_x, index = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, labels_x, index = next(labeled_train_iter)
        inputs = inputs_x.cuda()
        labels = updated_label[index].cuda()
        batch_size = inputs.shape[0]

        targets = torch.zeros(batch_size, args.num_classes, device=inputs.device).scatter_(1, labels.view(-1, 1), 1)
        l = np.random.beta(args.beta, args.beta)
        l = max(l, 1 - l)
        idx = torch.randperm(batch_size)
        input_a, input_b = inputs, inputs[idx]
        target_a, target_b = targets, targets[idx]
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # print(mixed_target.shape, mixed_input.shape)
        logits = classifier(encoder(mixed_input))
        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))

        xlosses.update(Lx.item())
        all_bar.set_description(
            f'Train epoch {epoch} LR:{optimizer.param_groups[0]["lr"]} Labeled loss: {xlosses.avg:.4f}')

        optimizer.zero_grad()
        Lx.backward()
        optimizer.step()
    return xlosses.avg


def test(testloader, encoder, classifier, epoch):
    encoder.eval()
    classifier.eval()
    accuracy = AverageMeter('accuracy')
    data_bar = tqdm(testloader)
    with torch.no_grad():
        for i, (data, label, _) in enumerate(data_bar):
            data, label = data.cuda(), label.cuda()
            feat = encoder(data)
            res = classifier(feat)
            pred = torch.argmax(res, dim=1)
            acc = torch.sum(pred == label) / float(data.size(0))
            accuracy.update(acc.item(), data.size(0))
            data_bar.set_description(f'Test epoch {epoch}: Accuracy#{accuracy.avg:.4f}')
    return accuracy.avg


def evaluate(dataloader, encoder, classifier, args, noisy_label, clean_label, i, initial_id):
    '''

    :param dataloader:
    :param encoder:
    :param classifier:
    :param args:
    :param noisy_label:
    :param clean_label: groundtruth labels
    :param i:
    :param stat_logs:
    :param initial_id: initial selected samples by CLIP?
    :return:
    '''
    encoder.eval()
    classifier.eval()
    feature_bank = []
    prediction = []

    ################################### feature extraction ###################################
    with torch.no_grad():
        # generate feature bank
        for (data, target, index) in tqdm(dataloader, desc='Feature extracting'):
            data = data.cuda()
            feature = encoder(data)
            feature_bank.append(feature)
            res = classifier(feature)
            prediction.append(res)

        ################################### sample relabelling ###################################
        prediction_cls = torch.softmax(torch.cat(prediction, dim=0), dim=1)
        his_score, his_label = prediction_cls.max(1)
        print(f'Prediction track: mean: {his_score.mean()} max: {his_score.max()} min: {his_score.min()}')
        same = his_label == noisy_label
        different = his_label != noisy_label
        mask1 = his_score.ge(args.theta_r).float() * same
        mask2 = his_score.ge(args.theta_r2).float() * different
        # mask = mask1 + mask2
        conf_id_same = torch.where(mask1 != 0)[0]
        conf_id_different = torch.where(mask2 != 0)[0]

        conf_id = torch.cat([conf_id_same, conf_id_different]).unique()

        modified_label = torch.clone(noisy_label).detach()
        modified_label[conf_id] = his_label[conf_id]

        select_id = torch.cat([conf_id, torch.tensor(initial_id).cuda()]).unique()
        correct = torch.sum(modified_label[select_id] == clean_label[select_id])
        orginal = torch.sum(noisy_label[select_id] == clean_label[select_id])
        all = len(select_id)
        print(
            f'Epoch [{i}/{args.epochs}] conf: {len(conf_id)} relabelling:  correct: {correct} original: {orginal} total: {all}')

    return modified_label, select_id


def main():
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True

    if args.run_path is None:
        args.run_path = f'{args.dataset}'
        print('Results saved in: ', args.run_path)

    # 1. load dataset and model
    none_aug, weak_aug, strong_aug = get_augment(args.dataset)
    encoder, classifier, num_classes = get_model(args.dataset)

    encoder = torch.nn.DataParallel(encoder).cuda()
    classifier = torch.nn.DataParallel(classifier).cuda()

    test_data = get_specific_dataset(args, aug=none_aug, mode='test', noise_mode=args.noise_mode,
                                     noise_ratio=args.noise_ratio, type=args.type)
    eval_data = get_specific_dataset(args, aug=none_aug, mode='train', noise_mode=args.noise_mode,
                                     noise_ratio=args.noise_ratio, type=args.type)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                              pin_memory=True)
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=args.batch_size * 4, shuffle=False, num_workers=4,
                                              pin_memory=True)

    train_data = get_specific_dataset(args, aug=strong_aug, mode='train', noise_mode=args.noise_mode,
                                      noise_ratio=args.noise_ratio, type=args.type)
    args.num_classes = train_data.num_classes

    args.model = 'small'
    initial_ids, selected_labels = combined_selection(args)

    # 3. training
    optimizer = SGD([{'params': encoder.parameters()}, {'params': classifier.parameters()}], lr=args.lr,
                    weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    if not os.path.isdir(f'{args.dataset}'):
        os.mkdir(f'{args.dataset}')
    if not os.path.isdir(f'{args.dataset}/{args.run_path}'):
        os.mkdir(f'{args.dataset}/{args.run_path}')
    acc_logs = open(f'{args.dataset}/{args.run_path}/acc.txt', 'w')
    save_config(args, f'{args.dataset}/{args.run_path}')
    print('Train args: \n', args)

    best_acc = 0
    noisy_label = torch.tensor(train_data.label).cuda()
    modified_label = torch.clone(noisy_label)
    selected_ids = torch.tensor(initial_ids).cuda()
    clean_label = train_data.clean_label if args.dataset.find('cifar') != -1 else train_data.label
    clean_label = torch.tensor(clean_label).cuda()

    for i in range(args.epochs):
        selected_labels = modified_label[selected_ids]
        num_per = torch.tensor([torch.sum(selected_labels == i) for i in range(num_classes)])
        # print(num_per.min(), num_per.max())
        sampler = ClassBalancedSampler(labels=selected_labels, num_classes=num_classes)
        # balancing the selected subset
        labeled_data = Subset(train_data, selected_ids.cpu())

        labeled_loader = torch.utils.data.DataLoader(labeled_data, batch_size=args.batch_size, sampler=sampler,
                                                     num_workers=4, pin_memory=True)
        num_iters = int(len(train_data) / float(args.batch_size))
        train(labeled_loader, modified_label, encoder, classifier, optimizer, i, args,
              num_iters)  # not always have clean label
        cur_acc = test(test_loader, encoder, classifier, i)
        scheduler.step()
        if cur_acc > best_acc:
            best_acc = cur_acc
            save_checkpoint({
                'cur_epoch': i,
                'classifier': classifier.state_dict(),
                'encoder': encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=f'{args.dataset}/{args.run_path}/best_acc.pth.tar')
        acc_logs.write(f'Epoch [{i}/{args.epochs}]: Best accuracy@{best_acc}! Current accuracy@{cur_acc} \n')
        acc_logs.flush()
        print(f'Epoch [{i}/{args.epochs}]: Best accuracy@{best_acc}! Current accuracy@{cur_acc} \n')

        # mixfix update
        modified_label, selected_ids = evaluate(eval_loader, encoder, classifier, args, noisy_label, clean_label, i,
                                                initial_ids)

    save_checkpoint({
        'cur_epoch': args.epochs,
        'classifier': classifier.state_dict(),
        'encoder': encoder.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, filename=f'{args.dataset}/{args.run_path}/last.pth.tar')


if __name__ == '__main__':
    main()
