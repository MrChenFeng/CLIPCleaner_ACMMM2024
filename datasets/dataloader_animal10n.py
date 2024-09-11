import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class animal_dataset(Dataset):
    # class_names = ['Cat', 'Lynx', 'Wolf', 'Coyote', 'Cheetah', 'Jaguar', 'Chimpanzee', 'Orangutan', 'Hamster', 'Guinea pig']
    # class_names = ['Cat', 'Lynx', 'Wolf', 'Coyote', 'Cheetah', 'jaguar', 'Chimpanzee', 'Orangutan', 'Hamster', 'Guinea pig']
    #################################################### An error in annotation: jaguar should be 4 while Cheetah should be 5 #####################################################################
    class_names = ['Cat', 'Lynx', 'Wolf', 'Coyote', 'jaguar', 'Cheetah', 'Chimpanzee', 'Orangutan', 'Hamster',
                   'Guinea pig']
    # we get detailed features from chatgpt
    detailed_features = [
        [
            'which has retractable claws',
            'which is a carnivorous mammal',
            'which is a common house pet',
            'which has excellent night vision',
            'which is agile',
            'which is a solitary hunter',
            'which has a keen sense of hearing',
            'which is a member of the Felidae family'
        ],
        [
            'which has tufted ears',
            'which has distinctive facial ruffs',
            'which is a skilled hunter',
            'which is found in forested areas',
            'which is a solitary animal',
            'which has sharp claws',
            'which is a member of the Lynx genus',
            'which has a short tail'
        ],
        [
            'which hunts in packs',
            'which is a carnivorous mammal',
            'which has a howling vocalization',
            'which has excellent sense of smell',
            'which has a strong bite',
            'which is a member of the Canidae family',
            'which has a bushy tail',
            'which is a social animal'
        ],
        [
            'which is found in North America',
            'which is a scavenger',
            'which has a yipping vocalization',
            'which is known for its adaptability',
            'which has sharp teeth',
            'which is a member of the Canis genus',
            'which has a pointed snout',
            'which has a dense fur coat'
        ],
        [
            'which is a skilled hunter',
            'which is found in the Americas',
            'which has distinctive rosette markings',
            'which has a powerful jaw',
            'which is a solitary animal',
            'which is a member of the Panthera genus',
            'which has a muscular body',
            'which has a long tail'
        ],
        [
            'which is the fastest land animal',
            'which is a carnivorous mammal',
            'which has distinctive spots',
            'which has a slender body',
            'which is a solitary animal',
            'which is a member of the Acinonyx genus',
            'which has non-retractable claws',
            'which has a flexible spine'
        ],
        [
            'which is a highly intelligent primate',
            'which is found in Africa',
            'which has opposable thumbs',
            'which has complex social structures',
            'which has a hairless face',
            'which is a member of the Hominidae family',
            'which has a diverse diet',
            'which is capable of using tools'
        ],
        [
            'which is a highly intelligent primate',
            'which is found in Asia',
            'which has distinctive red fur',
            'which is arboreal',
            'which has a long tail',
            'which is a member of the Pongo genus',
            'which has a varied diet',
            'which builds sleeping nests'
        ],
        [
            'which is a small rodent',
            'which is a popular pet',
            'which is known for storing food in its cheeks',
            'which is active primarily at night',
            'which has incisor teeth that continuously grow',
            'which is a member of the Cricetinae subfamily',
            'which has a short tail',
            'which has poor eyesight'
        ],
        [
            'which is a small rodent',
            'which is a popular pet',
            'which is known for its vocalizations',
            'which has a docile temperament',
            'which has a sensitive sense of smell',
            'which is a member of the Cavia genus',
            'which has short legs',
            'which has a compact body'
        ]
    ]
    suffix = ', a type of animal.'
    num_classes = 10

    def __init__(self, root_dir, transform=None, mode='train'):
        train_path = os.listdir(os.path.abspath(root_dir) + '/training')
        test_path = os.listdir(os.path.abspath(root_dir) + '/testing')
        print('Please be patient for image loading!')
        if mode == 'train':
            dir_path = os.path.abspath(root_dir) + '/training'
            self.label = [int(i.split('_')[0]) for i in train_path]
            self.data = [np.asarray(Image.open(dir_path + '/' + i)) for i in train_path]
        else:
            dir_path = os.path.abspath(root_dir) + '/testing'
            self.label = [int(i.split('_')[0]) for i in test_path]
            self.data = [np.asarray(Image.open(dir_path + '/' + i)) for i in test_path]
        print('Loading finished!')

        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index

    def update_labels(self, new_label):
        self.label = new_label.cpu()

    def __len__(self):
        return len(self.label)
