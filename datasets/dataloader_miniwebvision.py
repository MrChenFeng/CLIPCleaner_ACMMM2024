import os

from PIL import Image
from torch.utils.data import Dataset


class imagenet_dataset(Dataset):
    def __init__(self, transform, num_class=50, root_dir='./'):
        # self.root_dir = root_dir + '/val/'
        self.root = root_dir + '/imagenet_val/'
        self.transform = transform
        self.val_data = []
        classes = os.listdir(self.root)
        classes.sort()
        for c in range(num_class):
            imgs = os.listdir(self.root + classes[c])
            for img in imgs:
                self.val_data.append([c, os.path.join(self.root, classes[c], img)])

    def __getitem__(self, index):
        data = self.val_data[index]
        target = data[0]
        image = Image.open(data[1]).convert('RGB')
        img = self.transform(image)
        return img, target, index

    def __len__(self):
        return len(self.val_data)


class miniwebvision_dataset(Dataset):
    class_names = [['tench', 'Tinca tinca'],
                   ['goldfish', 'Carassius auratus'],
                   ['great white shark', 'white shark', 'Carcharodon carcharias'],
                   ['tiger shark', 'Galeocerdo cuvieri'],
                   ['hammerhead shark'],
                   ['electric ray', 'crampfish', 'numbfish'],
                   ['stingray'],
                   ['cock animal', 'cock bird'],
                   ['hen'],
                   ['ostrich', 'Struthio camelus'],
                   ['brambling', 'Fringilla montifringilla'],
                   ['goldfinch', 'Carduelis carduelis'],
                   ['house finch', 'linnet', 'Carpodacus mexicanus'],
                   ['junco', 'snowbird bird'],
                   ['indigo bunting', 'indigo finch', 'indigo bird', 'Passerina cyanea'],
                   ['robin bird', 'American robin', 'Turdus migratorius'],
                   ['bulbul'],
                   ['jay bird'],
                   ['magpie'],
                   ['chickadee'],
                   ['water ouzel', 'dipper bird'],
                   ['kite bird'],
                   ['bald eagle', 'American eagle bird', 'Haliaeetus leucocephalus'],
                   ['vulture'],
                   ['great grey owl', 'great gray owl', 'Strix nebulosa'],
                   ['European fire salamander', 'Salamandra salamandra'],
                   ['common newt', 'Triturus vulgaris'],
                   ['eft newt'],
                   ['spotted salamander', 'Ambystoma maculatum'],
                   ['axolotl', 'Ambystoma mexicanum'],
                   ['bullfrog', 'Rana catesbeiana'],
                   ['tree frog', 'tree-frog'],
                   ['tailed frog', 'bell toad', 'ribbed toad', 'tailed toad', 'Ascaphus truei'],
                   ['loggerhead turtle', 'Caretta caretta'],
                   ['leatherback turtle', 'leathery turtle', 'Dermochelys coriacea'],
                   ['mud turtle'],
                   ['terrapin'],
                   ['box turtle', 'box tortoise'],
                   ['banded gecko'],
                   ['common iguana', 'iguana', 'Iguana iguana'],
                   ['American chameleon', 'anole', 'Anolis carolinensis'],
                   ['whiptail', 'whiptail lizard'],
                   ['agama'],
                   ['frilled lizard', 'Chlamydosaurus kingi'],
                   ['alligator lizard'],
                   ['Gila monster', 'Heloderma suspectum'],
                   ['green lizard', 'Lacerta viridis'],
                   ['African chameleon', 'Chamaeleo chamaeleon'],
                   ['Komodo dragon',
                    'Komodo lizard',
                    'dragon lizard',
                    'giant lizard',
                    'Varanus komodoensis'],
                   ['African crocodile', 'Nile crocodile', 'Crocodylus niloticus']]
    detailed_features = [
        [
            'which is a type of fish',
            'which is typically found in freshwater',
            'which has an elongated body with a dorsal fin',
            'which is often used for recreational fishing',
            'which can be found in many different colors',
            'which can grow up to several feet in length',
            'which is often considered a game fish',
            'which is a popular food fish'
        ],
        [
            'which is a type of fish',
            'which is typically kept as a pet',
            'which is often kept in a fishbowl or aquarium',
            'which is known for its shiny, golden scales',
            'which is a popular species for aquaculture',
            'which is often bred for its unique appearance',
            'which is a popular subject in art and literature',
            'which is believed to bring good luck'
        ],
        [
            'which is a type of shark',
            'which is known for its large size and predatory behavior',
            'which is found in oceans around the world',
            'which has a gray or white body and pointed teeth',
            'which is often the subject of movies and documentaries',
            'which is an apex predator in its ecosystem',
            'which can grow up to several meters in length',
            'which is often hunted for its fins and meat'
        ],
        [
            'which is a type of shark',
            'which is known for its striped body pattern',
            'which is found in tropical and temperate oceans',
            'which has a pointed snout and serrated teeth',
            'which is often the subject of movies and documentaries',
            'which is a powerful predator in its ecosystem',
            'which can grow up to several meters in length',
            'which is often hunted for its fins and meat'
        ],
        [
            'which is a type of shark',
            'which is known for its hammer-shaped head',
            'which is found in warm coastal waters',
            'which has a gray or brown body with a white belly',
            'which is often the subject of movies and documentaries',
            'which uses its head to detect prey in the sand',
            'which can grow up to several meters in length',
            'which is often hunted for its fins and meat'
        ],
        [
            'which is a type of fish',
            'which is known for its ability to generate electric shocks',
            'which is found in freshwater and saltwater habitats',
            'which has a flattened body and a wide head',
            'which is often the subject of scientific research',
            'which uses its electric field to navigate and detect prey',
            'which can grow up to several feet in length',
            'which is often found in shallow water habitats'
        ],
        [
            'which is a type of fish',
            'which is known for its venomous spine',
            'which is found in tropical and subtropical waters',
            'which has a flattened body and a long tail',
            'which is often the subject of scientific research',
            'which uses its venomous spine for self-defense',
            'which can grow up to several feet in length',
            'which is often found in sandy or rocky habitats'
        ],
        [
            'which is a domesticated bird',
            'which is often raised for meat and eggs',
            'which has a distinctive red crest and wattle',
            'which is known for its crowing at dawn',
            'which is often kept on farms and in backyards',
            'which is a popular subject in art and literature',
            'which is often depicted as a farm animal',
            'which can come in many different colors'
        ],
        [
            'which is a domesticated bird',
            'which is often raised for its meat and eggs',
            'which has a smaller comb and wattles than the cock',
            'which is often kept on farms and in backyards',
            'which is a popular subject in art and literature',
            'which is often depicted as a farm animal',
            'which can come in many different colors',
            'which is sometimes kept as a pet'
        ],
        [
            'which is a flightless bird',
            'which is native to Africa',
            'which has a distinctive long neck and legs',
            'which is known for its fast running speed',
            'which is often kept on farms and in zoos',
            'which is a popular subject in art and literature',
            'which is often depicted as a comical or silly animal',
            'which can weigh up to several hundred pounds'
        ],
        [
            'which is a small passerine bird',
            'which breeds in the forests of northern Eurasia',
            'which has a distinctive orange breast and white belly',
            'which has a black head, throat, and bill',
            'which is known for its beautiful song',
            'which often travels in large flocks',
            'which feeds on seeds and insects',
            'which migrates south to Asia and Europe for the winter'
        ],
        [
            'which is a small passerine bird',
            'which is native to Europe, Asia, and North Africa',
            'which has a bright yellow and black plumage',
            'which has a distinctive red face and white cheeks',
            'which is known for its melodic song',
            'which often feeds on seeds and insects',
            'which is a common backyard bird',
            'which can be kept as a pet'
        ],
        [
            'which is a small passerine bird',
            'which is native to North America',
            'which has a brown and gray plumage',
            'which has a distinctive red forehead and breast',
            'which is known for its cheerful song',
            'which often feeds on seeds and fruits',
            'which is a common backyard bird',
            'which can be attracted to bird feeders'
        ],
        [
            'which is a small sparrow-like bird',
            'which is native to North America',
            'which has a gray head and back',
            'which has a white belly and dark wings',
            'which is known for its trilling song',
            'which often feeds on seeds and insects',
            'which is a common winter visitor to bird feeders',
            'which can be found in wooded areas and backyards'
        ],
        [
            'which is a small passerine bird',
            'which is native to North America',
            'which has a bright blue plumage',
            'which has a distinctive forked tail and wing bars',
            'which is known for its high-pitched song',
            'which often feeds on seeds and insects',
            'which is a common backyard bird',
            'which can be attracted to bird feeders'
        ],
        [
            'which is a small passerine bird',
            'which is native to Europe and North America',
            'which has a distinctive red breast and face',
            'which has a brown and gray back and wings',
            'which is known for its cheerful song',
            'which often feeds on insects, fruits, and berries',
            'which is a common backyard bird',
            'which is often depicted on Christmas cards'
        ],
        [
            'which is a medium-sized passerine bird',
            'which is native to Africa and Asia',
            'which has a brown or gray plumage',
            'which has a distinctive crest on its head',
            'which is known for its melodious song',
            'which often feeds on insects, fruits, and nectar',
            'which is a common backyard bird in some regions',
            'which is sometimes kept as a pet'
        ],
        [
            'which is a medium to large-sized bird',
            'which is native to North America',
            'which has a blue or gray plumage',
            'which has a distinctive crest on its head',
            'which is known for its loud and raucous calls',
            'which often feeds on nuts, seeds, and insects',
            'which is a common backyard bird in some regions',
            'which is sometimes considered a nuisance bird'
        ],
        [
            'which is a medium-sized bird',
            'which is native to Europe, Asia, and Africa',
            'which has a black and white plumage',
            'which has a long tail and a distinctive beak',
            'which is known for its intelligence and adaptability',
            'which often feeds on insects, fruits, and carrion',
            'which is sometimes considered a pest bird',
            'which is sometimes featured in folklore and literature'
        ],
        [
            'which is a small passerine bird',
            'which is native to North America',
            'which has a gray and black plumage',
            'which has a distinctive black cap and bib',
            'which is known for its cheerful and whistling song',
            'which often feeds on insects and seeds',
            'which is a common backyard bird in some regions',
            'which is sometimes attracted to bird feeders'
        ],
        [
            'which is a small bird',
            'which is also known as the American dipper',
            'which is native to North America',
            'which has a dark plumage and white feathers on its eyelids',
            'which is able to walk and swim underwater',
            'which is known for its beautiful song',
            'which feeds on aquatic insects and larvae',
            'which builds its nest near fast-moving streams and rivers'
        ],
        [
            'which is a bird of prey',
            'which is found in many parts of the world',
            'which has a long, pointed wingspan',
            'which has a forked tail',
            'which is known for its soaring flight',
            'which feeds on small mammals, birds, and insects',
            'which is sometimes kept as a pet or used in falconry'
        ],
        [
            'which is a bird of prey',
            'which is native to North America',
            'which has a brown body and white head and tail',
            'which has a hooked beak and powerful talons',
            'which is known for its majestic appearance',
            'which feeds on fish, small mammals, and carrion',
            'which is a national symbol of the United States',
            'which is sometimes kept in captivity for educational purposes'
        ],
        [
            'which is a bird of prey',
            'which is found on every continent except Antarctica and Australia',
            'which has a bald head and neck',
            'which has a sharp, hooked beak',
            'which is known for its ability to eat carrion',
            'which plays an important role in many ecosystems',
            'which is sometimes considered a symbol of death or decay'
        ],
        [
            'which is a large owl',
            'which is native to the Northern Hemisphere',
            'which has a gray and white plumage',
            'which has a large head and yellow eyes',
            'which is known for its excellent hearing',
            'which feeds on small mammals, birds, and fish',
            'which is sometimes considered a symbol of wisdom or intelligence',
            'which is sometimes kept in captivity for educational purposes'
        ],
        [
            'which is a large, brightly colored salamander',
            'which is native to central and southern Europe',
            'which has distinctive black and yellow stripes',
            'which has a broad head and stout body',
            'which is toxic and secretes poison from its skin',
            'which often lives in damp forests and near streams',
            'which can live up to 15 years in the wild',
            'which is an important symbol in European folklore'
        ],
        [
            'which is a small, semiaquatic salamander',
            'which is native to Europe and western Asia',
            'which has a brown or olive green coloration',
            'which has a rough, granular skin',
            'which is known for its flattened head and wide tail',
            'which often lives in ponds, lakes, and slow-moving streams',
            'which feeds on insects, worms, and small aquatic animals',
            'which can regenerate lost limbs'
        ],
        [
            'which is a juvenile or terrestrial phase of a newt',
            'which is often brightly colored, such as red or orange',
            'which has a smooth, moist skin',
            'which has a long tail and sharp claws',
            'which often lives in woodlands and grasslands',
            'which feeds on insects, snails, and small animals',
            'which can secrete toxins from its skin',
            'which can live up to 12 years in the wild'
        ],
        [
            'which is a large, stocky salamander',
            'which is native to eastern North America',
            'which has a distinctive yellow spots on its black body',
            'which has a broad, flat head and short legs',
            'which often lives in moist woodlands and near vernal pools',
            'which can secrete a toxic milky substance from its skin',
            'which feeds on insects, worms, and small animals',
            'which is a popular subject in North American folklore'
        ],
        [
            'which is a neotenic salamander',
            'which is native to Mexico',
            'which has a distinctive frilly appearance around its neck',
            'which has a wide, flat head and long tail',
            'which is known for its ability to regenerate lost limbs and organs',
            'which often lives in lakes and canals',
            'which feeds on insects, crustaceans, and small fish',
            'which is often used in scientific research due to its ability to regenerate'
        ],
        [
            'which is a large, semiaquatic frog',
            'which is native to North America',
            'which has a green or brown skin with dark spots',
            'which has a deep, resonant croak',
            'which is known for its loud and distinctive call',
            'which often inhabits ponds, lakes, and rivers',
            'which feeds on insects, fish, and small mammals',
            'which is commonly used for dissection in biology classes'
        ],
        [
            'which is a small, arboreal frog',
            'which is found in many parts of the world',
            'which has a bright green or yellow skin with markings',
            'which has large, adhesive toe pads for climbing',
            'which is known for its high-pitched chirping call',
            'which often inhabits trees and shrubs near water',
            'which feeds on insects and other small invertebrates',
            'which can change color to blend in with its surroundings'
        ],
        [
            'which is a small, semiaquatic frog',
            'which is native to North America',
            'which has a brown or gray skin with markings',
            'which has a distinctive pointed tail',
            'which is known for its high-pitched, birdlike call',
            'which often inhabits cold, clear streams and rivers',
            'which feeds on insects and other small invertebrates',
            'which is sensitive to pollution and habitat degradation'
        ],
        [
            'which is a large sea turtle',
            'which is found in many parts of the world',
            'which has a reddish-brown shell and a large head',
            'which is known for its powerful jaws and strong bite',
            'which often nests on beaches and feeds on jellyfish',
            'which is an endangered species due to habitat loss and hunting',
            'which can hold its breath underwater for up to four hours',
            'which can migrate over long distances to feed and nest'
        ],
        [
            'which is a large sea turtle',
            'which is found in many parts of the world',
            'which has a black or dark blue shell and no visible bony plates',
            'which is known for its large size and long migrations',
            'which often feeds on jellyfish and other soft-bodied organisms',
            'which is an endangered species due to habitat loss and hunting',
            'which can dive to depths of over 1,000 meters',
            'which has a flexible, leathery shell that allows it to dive deep'
        ],
        [
            'which is a small freshwater turtle',
            'which has a smooth shell and webbed feet',
            'which is native to North America',
            'which is often found in muddy areas and slow-moving water',
            'which feeds on insects, crustaceans, and small fish',
            'which can hibernate during the winter months',
            'which is a popular pet turtle',
            'which is sometimes sold in pet stores'
        ],
        [
            'which is a small to medium-sized turtle',
            'which is found in brackish or saltwater environments',
            'which has a flat shell and webbed feet',
            'which is native to North America and Central America',
            'which feeds on a variety of foods, including crustaceans and plants',
            'which is a popular food item in some cultures',
            'which is often used in turtle soup',
            'which is sometimes kept as a pet'
        ],
        [
            'which is a small to medium-sized turtle',
            'which has a high-domed shell and hinged lower shell',
            'which is native to North America',
            'which is often found in wooded areas and near water sources',
            'which feeds on a variety of foods, including insects and plants',
            'which is a popular pet turtle',
            'which can hibernate during the winter months',
            'which is sometimes sold in pet stores'
        ],
        [
            'which is a small to medium-sized lizard',
            'which has a banded pattern on its skin',
            'which is native to the southwestern United States',
            'which is often found in desert and rocky areas',
            'which feeds on insects and small invertebrates',
            'which can shed its tail as a defense mechanism',
            'which is a popular pet lizard',
            'which is sometimes sold in pet stores'
        ],
        [
            'which is a large, arboreal lizard',
            'which has a spiny crest on its back',
            'which is native to Central and South America',
            'which is often found in forests and near water sources',
            'which feeds on a variety of foods, including plants and insects',
            'which can change color in response to its environment',
            'which is a popular pet lizard',
            'which requires a large enclosure and specialized care'
        ],
        [
            'which is a reptile',
            'which is native to North and South America',
            'which is also known as an anole or iguana',
            'which has the ability to change color',
            'which has a long, sticky tongue to catch prey',
            'which has feet that can grip onto surfaces',
            'which can be kept as a pet'
        ],
        [
            'which is a lizard',
            'which is native to the Americas',
            'which has a long, slender body and tail',
            'which can run quickly on its hind legs',
            'which is often brown or gray in color',
            'which feeds on insects and small animals',
            'which is a common sight in deserts and grasslands',
            'which can reproduce through parthenogenesis'
        ],
        [
            'which is a lizard',
            'which is native to Africa, Asia, and Europe',
            'which is often brightly colored',
            'which has a triangular head and long tail',
            'which has the ability to shed its tail to escape predators',
            'which feeds on insects and small animals',
            'which is often kept as a pet',
            'which can change color to regulate its body temperature'
        ],
        [
            'which is a lizard',
            'which is native to Australia and New Guinea',
            'which has a frill of skin around its neck',
            'which can expand the frill to intimidate predators',
            'which is often brown or gray in color',
            'which feeds on insects and small animals',
            'which is a solitary animal',
            'which is often kept as a pet'
        ],
        [
            'which is a lizard',
            'which is native to North America',
            'which has a long, slender body and tail',
            'which is often brown or green in color',
            'which has a rough, scaly skin',
            'which is often found in wooded areas',
            'which feeds on insects and small animals',
            'which can drop its tail to escape predators'
        ],
        [
            'which is a venomous lizard',
            'which is native to the southwestern United States and Mexico',
            'which has a thick, bumpy skin',
            'which has a black and orange pattern',
            'which is known for its slow-moving nature',
            'which feeds on small mammals, birds, and reptiles',
            'which can survive for months without food or water',
            'which is protected by law in some areas due to its endangered status'
        ],
        [
            'which is a type of lizard',
            'which is found in many parts of the world',
            'which has a green coloration',
            'which has a long tail and sharp claws',
            'which is known for its ability to regenerate lost limbs',
            'which often lives in trees and bushes',
            'which feeds on insects and small animals',
            'which can change its color to blend in with its surroundings'
        ],
        [
            'which is a type of lizard',
            'which is native to sub-Saharan Africa',
            'which has a long, sticky tongue',
            'which has a prehensile tail',
            'which can change its color to match its surroundings',
            'which has independently moving eyes',
            'which feeds on insects and small animals',
            'which is known for its slow, deliberate movements'
        ],
        [
            'which is a type of lizard',
            'which is native to the Indonesian islands of Komodo, Rinca, Flores, and Gili Motang',
            'which is the largest living species of lizard',
            'which has a rough, scaly skin',
            'which has a long tail and sharp claws',
            'which is known for its deadly bite',
            'which feeds on deer, pigs, and water buffalo',
            'which is listed as vulnerable to extinction by the IUCN'
        ],
        [
            'which is a large reptile',
            'which is found throughout sub-Saharan Africa',
            'which has a long, powerful tail',
            'which has a scaly, armored skin',
            'which is known for its strong jaws and sharp teeth',
            'which feeds on fish, birds, and mammals',
            'which can live for over 50 years in the wild',
            'which is threatened by habitat loss and poaching'
        ]
    ]
    suffix = ', a type of animal.'
    num_classes = 50

    def __init__(self, root_dir, transform, mode, num_class=50):
        self.root = root_dir
        self.transform = transform
        self.mode = mode

        if self.mode == 'test':
            with open(self.root + '/info/val_filelist.txt') as f:
                lines = f.readlines()
            self.val_imgs = []
            self.label = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    self.val_imgs.append(img)
                    self.label[img] = target
        elif self.mode == 'train':
            with open(self.root + '/info/train_filelist_google.txt') as f:
                lines = f.readlines()
            self.train_imgs = []
            # self.label = {}
            self.label = []
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    self.train_imgs.append(img)
                    # self.label[img] = target
                    self.label.append(target)
        else:
            raise ValueError(f'dataset_mode should be train or test, rather than {self.mode}!')

        self.clean_label = self.label

    def update_labels(self, new_label_dict):
        if self.mode == 'train':
            self.label = new_label_dict.cpu()
        else:
            raise ValueError(f'Dataset mode should be train rather than {self.mode}!')

    def __getitem__(self, index):
        if self.mode == 'train':
            img_path = self.train_imgs[index]
            target = self.label[index]
            image = Image.open(self.root + '/' + img_path).convert('RGB')
            img = self.transform(image)
            image.close()

            return img, target, index
        elif self.mode == 'test':
            img_path = self.val_imgs[index]
            target = self.label[img_path]
            image = Image.open(self.root + '/val_images_256/' + img_path).convert('RGB')
            img = self.transform(image)
            image.close()

            return img, target, index

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)
