from .dataloader_animal10n import animal_dataset
from .dataloader_cifar import cifar10_dataset, cifar100_dataset
from .dataloader_clothing1m import clothing_dataset
from .dataloader_miniwebvision import miniwebvision_dataset
from .dataloader_redimagenet import red_mini_imagenet_dataset

__all__ = ['MixTransform', 'get_specific_dataset']


class MixTransform:
    def __init__(self, strong_transform, weak_transform, K=2):
        self.strong_transform = strong_transform
        self.weak_transform = weak_transform
        self.K = K

    def __call__(self, x):
        res = [self.weak_transform(x) for i in range(self.K)] + [self.strong_transform(x) for i in range(self.K)]
        return res


def get_specific_dataset(args, aug, mode, noise_mode='sym', noise_ratio=0.5, type='blue'):
    if args.dataset == 'clothing1m':
        dataset = clothing_dataset(root_dir=args.dataset_path, transform=aug, mode=mode,
                                   num_samples=args.batch_size * 1000)

    elif args.dataset == 'cifar10' or args.dataset == 'cifar10-IND':
        dataset = cifar10_dataset(dataset=args.dataset, root_dir=args.dataset_path, transform=aug, mode=mode,
                                  noise_mode=noise_mode, noise_ratio=noise_ratio)

    elif args.dataset == 'cifar100':
        dataset = cifar100_dataset(dataset='cifar100', root_dir=args.dataset_path, transform=aug, mode=mode,
                                   noise_mode=noise_mode, noise_ratio=noise_ratio)

    elif args.dataset == 'animal10n':
        dataset = animal_dataset(root_dir=args.dataset_path, transform=aug, mode=mode)

    elif args.dataset == 'webvision':
        dataset = miniwebvision_dataset(root_dir=args.dataset_path, transform=aug, mode=mode)

    else:  # red_imagenet
        dataset = red_mini_imagenet_dataset(root_dir=args.dataset_path, transform=aug, mode=mode,
                                            noise_ratio=noise_ratio, type=type)

    return dataset
