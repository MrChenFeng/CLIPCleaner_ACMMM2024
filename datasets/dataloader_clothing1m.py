import random

import torch
from PIL import Image
from torch.utils.data import Dataset


class clothing_dataset(Dataset):
    '''
    Select 14 instead of using mini-batch
    '''

    # class_names = ['T-shirt', 'Shirt', 'Knitwear', 'Chiffon', 'Sweater', 'Hoodie', 'Windbreaker', 'Jacket', 'Down Coat', 'Suit', 'Shawl', 'Dress', 'Vest', 'Underwear']
    class_names = ['T-shirt', 'Shirt', 'Knitwear', 'Chiffon', 'Sweater', 'Hoodie', 'Windbreaker', 'Jacket', 'Down Coat',
                   'Suit', 'Shawl', 'Dress', 'Vest', 'Underwear']
    detailed_features = [
        [
            'which is a casual, short-sleeved garment',
            'which is made of cotton or polyester',
            'which is often worn as an undershirt or on its own',
            'which is characterized by a round neckline and a simple design',
            'which is versatile and can be dressed up or down',
            'which is popular in warm weather',
            'which comes in many colors and patterns',
            'which is often worn by athletes and teenagers'
        ],
        [
            'which is a button-up garment',
            'which is typically made of cotton or silk',
            'which is often worn with dress pants or a suit',
            'which comes in a variety of colors and patterns',
            'which can be long-sleeved or short-sleeved',
            'which can be formal or casual',
            'which is a staple of men\'s and women\'s wardrobes',
            'which can be tucked in or left untucked'
        ],
        [
            'which is a type of clothing made from knitted fabric',
            'which is often made of wool, cotton, or synthetic fibers',
            'which is characterized by its stretchiness and softness',
            'which can be long-sleeved or short-sleeved',
            'which comes in a variety of styles and designs',
            'which is warm and comfortable to wear',
            'which is often worn in colder weather',
            'which can be dressed up or down'
        ],
        [
            'which is a lightweight, sheer fabric',
            'which is often made of silk or polyester',
            'which is characterized by its draping quality',
            'which is often used in formal or dressy clothing',
            'which comes in a variety of colors and patterns',
            'which can be used for blouses, dresses, and skirts',
            'which is cool and comfortable to wear in warm weather',
            'which requires special care in washing and ironing'
        ],
        [
            'which is a warm, knitted garment',
            'which is often made of wool or acrylic',
            'which is characterized by its texture and pattern',
            'which can be crewneck or V-neck',
            'which can be long-sleeved or short-sleeved',
            'which is often worn in colder weather',
            'which can be dressed up or down',
            'which comes in many colors and patterns'
        ],
        [
            'which is a casual, hooded sweatshirt',
            'which is often made of cotton or polyester',
            'which is characterized by its hood and front pocket',
            'which is often worn for exercise or leisure',
            'which can be long-sleeved or short-sleeved',
            'which comes in a variety of colors and designs',
            'which is warm and comfortable to wear',
            'which is popular with teenagers and young adults'
        ],
        [
            'which is a lightweight, water-resistant jacket',
            'which is often made of nylon or polyester',
            'which is characterized by its windproof and breathable properties',
            'which is often used for outdoor activities',
            'which can be zip-up or button-up',
            'which comes in a variety of colors and styles',
            'which is versatile and can be worn in different weather conditions',
            'which is popular with hikers, campers, and travelers'
        ],
        [
            'which is a type of outerwear',
            'which typically has a front zipper or buttons',
            'which often has pockets and a collar',
            'which can be made from a variety of materials',
            'which can be worn as a casual or formal garment',
            'which can be tailored or fitted for a specific look',
            'which can be worn in different seasons and weather',
            'which can come in different styles and colors'
        ],
        [
            'which is a type of coat',
            'which is filled with feathers or down insulation',
            'which is designed for warmth in cold weather',
            'which often has a hood and zippered pockets',
            'which can be made from water-resistant materials',
            'which can come in different lengths and styles',
            'which is often worn for outdoor activities',
            'which can be paired with winter accessories'
        ],
        [
            'which is a type of formal attire',
            'which typically includes a jacket and trousers',
            'which is often made from wool or other fine materials',
            'which can come in different colors and patterns',
            'which is usually worn for special occasions',
            'which can be tailored or fitted for a specific look',
            'which can be accessorized with a tie or pocket square',
            'which is a classic and timeless wardrobe staple'
        ],
        [
            'which is a type of scarf',
            'which is typically made from soft, woven fabric',
            'which can be worn around the neck or shoulders',
            'which can come in different colors and patterns',
            'which is often used for warmth or fashion',
            'which can be draped or wrapped in different styles',
            'which can be paired with different outfits and accessories',
            'which is a versatile and stylish accessory'
        ],
        [
            'which is a type of garment',
            'which is typically worn by women',
            'which can come in different lengths and styles',
            'which can be made from different materials',
            'which is often designed to accentuate the figure',
            'which can be worn for different occasions',
            'which can be paired with different shoes and accessories',
            'which is a classic and feminine wardrobe staple'
        ],
        [
            'which is a type of garment',
            'which is typically sleeveless',
            'which can come in different styles and materials',
            'which can be worn as an outer or inner layer',
            'which can be worn for different occasions',
            'which can be paired with different outfits and accessories',
            'which is a versatile and functional wardrobe staple',
            'which can provide warmth or added style to an outfit'
        ],
        [
            'which is a type of garment',
            'which is worn closest to the skin',
            'which can come in different styles and materials',
            'which can be designed for warmth or comfort',
            'which can come in different levels of support',
            'which can be paired with different underwear and clothing',
            'which is an essential and functional wardrobe item',
            'which can be made from natural or synthetic fibers'
        ]
    ]
    suffix = ', a type of clothes.'
    num_classes = 14

    def __init__(self, root_dir, transform=None, mode='train', num_samples=0, num_classes=14):
        super(clothing_dataset, self).__init__()
        # super(Clothing1M, self).__init__(root_dir, transform, mode)
        self.root = root_dir
        self.transform = transform
        self.mode = mode

        # started random selected evaluate samples
        if self.mode == 'train':  # select part of samples for subsequent maintrain [clothing only]
            label = {}
            with open('%s/noisy_label_kv.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for line in lines:
                    entry = line.split()
                    img_path = '%s/' % self.root + entry[0]  # [7:]
                    label[img_path] = int(entry[1])

                data = []
                with open('%s/noisy_train_key_list.txt' % self.root, 'r') as f2:
                    lines = f2.read().splitlines()
                    for line in lines:
                        img_path = '%s/' % self.root + line  # [7:]
                        data.append(img_path)
                # select same samples always
                random.shuffle(data)

                class_num = torch.zeros(num_classes)
                self.data = []
                self.label = []
                for path in data:
                    cur_label = label[path]
                    if class_num[cur_label] < (num_samples / num_classes) and len(self.data) < num_samples:
                        self.data.append(path)
                        self.label.append(cur_label)
                        class_num[cur_label] += 1
                # random.shuffle(self.data)

        else:  # self.mode == 'evaluation':
            # self.label = {}
            label = {}
            with open('%s/clean_label_kv.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for line in lines:
                    entry = line.split()
                    img_path = '%s/' % self.root + entry[0]  # [7:]
                    # self.label[img_path] = int(entry[1])
                    label[img_path] = int(entry[1])

            self.data = []
            self.label = []
            with open('%s/clean_test_key_list.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for line in lines:
                    img_path = '%s/' % self.root + line  # [7:]
                    self.data.append(img_path)
                    self.label.append(label[img_path])
        # self.label = np.array(self.label)
        # self.data = np.array(self.data)

    def set_augment(self, transform):  # transform: strong, weak, none
        self.transform = transform

    def get_subset(self, subset_id):
        self.data = self.data[subset_id]
        self.label = self.label[subset_id]

    def __getitem__(self, index):
        img_path = self.data[index]
        # target = self.label[img_path]
        target = self.label[index]
        image = Image.open(img_path).convert('RGB')
        img = self.transform(image)
        return img, target, index
        # return img, target, img_path
        # if self.mode == 'train':
        #     return img, target, target, index
        # else:
        #     return img, target, index

    def __len__(self):
        return len(self.label)
