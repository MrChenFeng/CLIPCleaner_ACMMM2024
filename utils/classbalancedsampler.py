## class-balanced data sampler
## Chen Feng
## 2021.09

import torch
from torch.utils.data.sampler import *


class ClassBalancedSampler(Sampler[int]):
    def __init__(self, labels, num_classes, num_samples=None, mode='max'):
        # self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.mode = mode
        self.labels = torch.as_tensor(labels, dtype=torch.int)
        self.classes = torch.arange(num_classes)
        self.num_classes = torch.as_tensor([torch.sum(self.labels == i) for i in self.classes], dtype=torch.int)
        # for each class, get K fold samples
        # print(self.num_classes)
        # print(self.num_classes.max(), self.num_classes.min())
        if num_samples is not None:  # and num_fold is None:
            self.num_samples = num_samples
        else:
            if self.mode == 'max':
                self.num_samples = self.num_classes.max()
            elif self.mode == 'min':
                self.num_samples = self.num_classes.min()
            elif self.mode == 'mean':
                self.num_samples = torch.ceil(self.num_classes.float().mean()).int()
            else:
                raise ValueError(f'mode should be max/min/mean other than {self.mode}!')

        ids = []
        # print(self.num_samples)
        # print(self.max_num)
        for i, cid in enumerate(self.classes):
            if self.num_classes[i] == 0:
                continue
            else:
                fold_i = torch.ceil(self.num_samples / self.num_classes[i]).to(torch.int)
            tmp_i = torch.where(self.labels == cid)[0].repeat(fold_i)  # [:self.max_num]
            start = 0 if self.num_classes[i] * (fold_i - 1) < 0 else self.num_classes[i] * (fold_i - 1)
            rand = torch.randperm(self.num_samples - start)
            tmp_i[-(self.num_samples - start):] = tmp_i[-(self.num_samples - self.num_classes[i] * (fold_i - 1)):][rand]
            ids.append(tmp_i[:self.num_samples])
        self.ids = torch.cat(ids)

    def __iter__(self):
        rand = torch.randperm(len(self.ids))
        ids = self.ids[rand]
        # rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement)
        return iter(ids.tolist())

    def __len__(self):
        return len(self.ids)

# class ClassBalancedSampler(Sampler[int]):
#     def __init__(self, labels, num_classes, num_samples=None, num_fold=1):
#         # self.weights = torch.as_tensor(weights, dtype=torch.double)
#         self.num_fold = num_fold
#         self.labels = torch.as_tensor(labels, dtype=torch.int)
#         self.classes = torch.arange(num_classes)
#         self.num_classes = torch.as_tensor([torch.sum(self.labels == i) for i in self.classes], dtype=torch.int)
#         if num_samples is not None and num_fold is None:
#             self.num_fold = torch.floor(torch.tensor(num_samples / len(labels)))
#         self.max_num = self.num_classes.max() * self.num_fold
#         ids = []
#         # print(self.max_num)
#         for i, cid in enumerate(self.classes):
#             if self.num_classes[i] == 0:
#                 continue
#             else:
#                 fold_i = torch.ceil(self.max_num / self.num_classes[i]).to(torch.int)
#             tmp_i = torch.where(self.labels == cid)[0].repeat(fold_i)  # [:self.max_num]
#             rand = torch.randperm(self.num_classes.max())
#             tmp_i[-self.num_classes.max():] = tmp_i[-self.num_classes.max():][rand]
#             ids.append(tmp_i[:self.max_num])
#         self.ids = torch.cat(ids)
#
#     def __iter__(self):
#         rand = torch.randperm(len(self.ids))
#         ids = self.ids[rand]
#         # rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement)
#         return iter(ids.tolist())
#
#     def __len__(self):
#         return len(self.ids)
