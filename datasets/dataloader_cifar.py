import json
import random

from torch.utils.data import Dataset
from torchvision.datasets.cifar import *


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class cifar10_dataset(Dataset):
    class_names = ['airplane',
                   'automobile',
                   'bird',
                   'cat',
                   'deer',
                   'dog',
                   'frog',
                   'horse',
                   'ship',
                   'truck']
    detailed_features = [
        [
            'which is a type of aircraft',
            'which is designed for air travel',
            'which is powered by one or more jet engines',
            'which is used for transportation of passengers and cargo',
            'which can travel at high speeds and altitudes',
            'which requires a runway for takeoff and landing',
            'which can be classified as commercial, military, or private',
            'which can vary in size from small single-engine planes to large jumbo jets'
        ],
        [
            'which is a motorized vehicle',
            'which is designed for transportation on roads',
            'which is powered by an internal combustion engine',
            'which has four or more wheels',
            'which can vary in size from small compact cars to large trucks',
            'which can be classified as sedans, SUVs, trucks, etc.',
            'which requires a driver\'s license to operate',
            'which can be fueled by gasoline, diesel, or electric power'
        ],
        [
            'which is a warm-blooded vertebrate',
            'which has feathers and wings',
            'which can fly or glide through the air',
            'which has a beak or bill for feeding',
            'which lays eggs instead of giving birth to live young',
            'which can be found in various habitats such as forests, grasslands, and wetlands',
            'which can vary in size from tiny hummingbirds to large ostriches',
            'which can be classified as passerines, raptors, waterfowl, etc.'
        ],
        [
            'which is a carnivorous mammal',
            'which has fur and sharp claws',
            'which is often kept as a pet',
            'which is known for its ability to hunt rodents',
            'which can be trained to perform various tricks',
            'which can purr, meow, and hiss',
            'which can vary in size from small domestic cats to large wildcats',
            'which is a popular subject in art and literature'
        ],
        [
            'which is a hoofed mammal',
            'which has antlers or horns',
            'which is known for its grace and speed',
            'which can be found in various habitats such as forests and grasslands',
            'which is a popular subject in hunting',
            'which can vary in size from small deer to large moose',
            'which can be classified as whitetails, mule deer, elk, etc.',
            'which is often depicted in art and literature'
        ],
        [
            'which is a common household pet',
            'which is known for its loyalty and companionship',
            'which comes in a wide variety of breeds and sizes',
            'which has a keen sense of smell and hearing',
            'which is often trained for various tasks and jobs',
            'which requires regular exercise and socialization',
            'which has a lifespan of 10-13 years on average',
            'which is often considered a member of the family'
        ],
        [
            'which is a cold-blooded amphibian',
            'which starts its life in water and then moves to land',
            'which has a smooth and slimy skin',
            'which has powerful hind legs for jumping and swimming',
            'which undergoes metamorphosis from a tadpole to an adult',
            'which has a lifespan ranging from 5-20 years depending on the species',
            'which plays an important ecological role as both predator and prey'
        ],
        [
            'which is a large domesticated mammal',
            'which is often used for riding, racing, and working',
            'which has a long and flowing mane and tail',
            'which comes in a wide variety of breeds and colors',
            'which has a gentle and friendly temperament',
            'which requires regular exercise, grooming, and veterinary care',
            'which has a lifespan of 25-30 years on average',
            'which has played a significant role in human history and culture'
        ],
        [
            'which is a large watercraft designed for transportation on the sea, rivers, or lakes',
            'which comes in a variety of sizes and shapes depending on the purpose',
            'which can be powered by engines or sails',
            'which has a hull that allows it to float on water',
            'which can carry passengers, cargo, or military equipment',
            'which has played an important role in exploration, trade, and warfare throughout history',
            'which can be made of various materials including wood, steel, and fiberglass'
        ],
        [
            'which is a large vehicle designed for transporting goods and cargo',
            'which comes in a variety of sizes and shapes depending on the purpose',
            'which has a cab for the driver and a cargo area for the goods',
            'which can be powered by diesel or gasoline engines',
            'which is often used in logistics and transportation industries',
            'which requires a commercial driver\'s license to operate',
            'which can be found on highways, roads, and construction sites',
            'which has played an important role in the global economy'
        ]
    ]
    suffix = '.'
    num_classes = 10

    def __init__(self, dataset, root_dir, transform, noise_mode='sym', mode='train',
                 noise_ratio=0.5):  # , noise_file=None):

        self.r = noise_ratio  # total noise ratio
        self.transform = transform
        self.mode = mode
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6,
                           8: 8}  # class transition for asymmetric noise
        self.closed_noise = None

        noise_file = f'{root_dir}/{noise_ratio}_{noise_mode}_noise.json'
        if self.mode == 'test':
            cifar_dic = unpickle('%s/cifar-10-batches-py/test_batch' % root_dir)
            self.cifar_data = cifar_dic['data']
            self.cifar_data = self.cifar_data.reshape((10000, 3, 32, 32))
            self.cifar_data = self.cifar_data.transpose((0, 2, 3, 1))
            self.label = cifar_dic['labels']

        elif self.mode == 'train':
            cifar_data = []
            cifar_label = []
            for n in range(1, 6):
                dpath = '%s/cifar-10-batches-py/data_batch_%d' % (root_dir, n)
                data_dic = unpickle(dpath)
                cifar_data.append(data_dic['data'])
                cifar_label = cifar_label + data_dic['labels']
            self.cifar_data = np.concatenate(cifar_data).reshape((50000, 3, 32, 32)).transpose((0, 2, 3, 1))

            self.clean_label = cifar_label

            if dataset == 'cifar10':
                if os.path.exists(noise_file):
                    noise = json.load(open(noise_file, "r"))
                    noise_labels = noise['noise_labels']
                    self.closed_noise = noise['closed_noise']
                    self.label = noise_labels
                else:
                    # inject noise
                    noise_labels = []  # all labels (some noisy, some clean)
                    idx = list(range(50000))  # indices of cifar dataset
                    random.shuffle(idx)
                    num_total_noise = int(self.r * 50000)  # total amount of noise
                    print('Statistics of synthetic noisy CIFAR dataset: ', 'num of clean samples: ',
                          50000 - num_total_noise,
                          ' num of closed-set noise: ', num_total_noise)
                    self.closed_noise = idx[:num_total_noise]  # closed set noise indices
                    # populate noise_labels
                    for i in range(50000):
                        if i in self.closed_noise:
                            if noise_mode == 'sym':
                                noiselabel = random.randint(0, 9)
                            else:  # if noise_mode == 'asym':
                                noiselabel = self.transition[cifar_label[i]]
                            noise_labels.append(noiselabel)
                        else:
                            noise_labels.append(cifar_label[i])

                    # write noise to a file, to re-use
                    noise = {'noise_labels': noise_labels, 'closed_noise': self.closed_noise}
                    print("save noise to %s ..." % noise_file)
                    json.dump(noise, open(noise_file, "w"))
                    self.label = noise_labels
            else:
                import csv
                def read_csv_to_numpy(file_path):
                    data = []
                    with open(file_path, 'r') as csvfile:
                        csvreader = csv.reader(csvfile)
                        header = next(csvreader)  # Skip the header row
                        for row in csvreader:
                            data.append(row)

                    data = np.array(data, dtype=int)
                    labels = data[:, 0]
                    label_noisy = data[:, 1]

                    return labels, label_noisy

                clean_label, noisy_label = read_csv_to_numpy(f'./datasets/INDnoise/dependent{noise_ratio}.csv')
                self.label = noisy_label  # replace noisy label by downloaded noise
        else:
            raise ValueError(f'Dataset mode should be train or test rather than {self.mode}!')

    def update_labels(self, new_label):
        self.label = new_label.cpu()

    def __getitem__(self, index):
        img = self.cifar_data[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        target = self.label[index]
        return img, target, index

    def __len__(self):
        return len(self.cifar_data)

    def get_noise(self):
        return self.closed_noise

    def __repr__(self):
        return f'dataset_mode: {self.mode}, dataset number: {len(self)} \n'


class cifar100_dataset(Dataset):
    class_names = ['apple',
                   'aquarium_fish',
                   'baby',
                   'bear',
                   'beaver',
                   'bed',
                   'bee',
                   'beetle',
                   'bicycle',
                   'bottle',
                   'bowl',
                   'boy',
                   'bridge',
                   'bus',
                   'butterfly',
                   'camel',
                   'can',
                   'castle',
                   'caterpillar',
                   'cattle',
                   'chair',
                   'chimpanzee',
                   'clock',
                   'cloud',
                   'cockroach',
                   'couch',
                   'crab',
                   'crocodile',
                   'cup',
                   'dinosaur',
                   'dolphin',
                   'elephant',
                   'flatfish',
                   'forest',
                   'fox',
                   'girl',
                   'hamster',
                   'house',
                   'kangaroo',
                   'keyboard',
                   'lamp',
                   'lawn_mower',
                   'leopard',
                   'lion',
                   'lizard',
                   'lobster',
                   'man',
                   'maple_tree',
                   'motorcycle',
                   'mountain',
                   'mouse',
                   'mushroom',
                   'oak_tree',
                   'orange',
                   'orchid',
                   'otter',
                   'palm_tree',
                   'pear',
                   'pickup_truck',
                   'pine_tree',
                   'plain',
                   'plate',
                   'poppy',
                   'porcupine',
                   'possum',
                   'rabbit',
                   'raccoon',
                   'ray',
                   'road',
                   'rocket',
                   'rose',
                   'sea',
                   'seal',
                   'shark',
                   'shrew',
                   'skunk',
                   'skyscraper',
                   'snail',
                   'snake',
                   'spider',
                   'squirrel',
                   'streetcar',
                   'sunflower',
                   'sweet_pepper',
                   'table',
                   'tank',
                   'telephone',
                   'television',
                   'tiger',
                   'tractor',
                   'train',
                   'trout',
                   'tulip',
                   'turtle',
                   'wardrobe',
                   'whale',
                   'willow_tree',
                   'wolf',
                   'woman',
                   'worm']
    detailed_features = [
        [
            'which is a type of fruit',
            'which has a red, green, or yellow skin',
            'which is round or oblong in shape',
            'which has a firm, juicy flesh',
            'which is sweet or tart in flavor',
            'which is high in fiber and vitamin C',
            'which is commonly used in cooking and baking',
            'which grows on trees in orchards'
        ],
        [
            'which is a type of pet fish',
            'which is kept in an aquarium',
            'which comes in many colors and patterns',
            'which requires a clean and well-maintained tank',
            'which requires a specific water temperature and pH level',
            'which requires a balanced diet of fish flakes and live or frozen food',
            'which can be peaceful or aggressive depending on the species',
            'which can be bred in captivity'
        ],
        [
            'which is a young human',
            'which is typically under 1 year old',
            'which requires special care and attention',
            'which is fed with milk or formula',
            'which sleeps for most of the day and night',
            'which needs frequent diaper changes',
            'which develops rapidly in the first year of life',
            'which learns through exploration and play'
        ],
        [
            'which is a large mammal',
            'which is found in many parts of the world',
            'which has a thick fur coat',
            'which has sharp claws and teeth',
            'which is an omnivore and eats both plants and animals',
            'which can be dangerous if threatened or provoked',
            'which hibernates during the winter in cold climates',
            'which has been hunted for its fur and meat'
        ],
        [
            'which is a semi-aquatic rodent',
            'which is found in North America and Europe',
            'which has a flat tail and webbed feet',
            'which has a thick fur coat that repels water',
            'which feeds on bark, leaves, and aquatic plants',
            'which builds dams and lodges using branches and mud',
            'which is an important keystone species in wetland ecosystems',
            'which has been hunted for its fur'
        ],
        [
            'which is a piece of furniture designed for sleeping on',
            'which consists of a mattress and a frame',
            'which comes in different sizes, such as twin, full, queen, and king',
            'which can be made of different materials, such as wood, metal, or upholstered',
            'which can have different styles, such as platform, sleigh, or four-poster'
        ],
        [
            'which is an insect that belongs to the superfamily Apoidea',
            'which is known for its role in pollination',
            'which has a hairy body and two pairs of wings',
            'which has a stinger that can be used for defense',
            'which lives in hives or nests made of wax',
            'which produces honey and beeswax',
            'which can communicate with other bees through dance',
            'which is an important part of many ecosystems'
        ],
        [
            'which is an insect that belongs to the order Coleoptera',
            'which has a hard outer shell called a carapace',
            'which has two pairs of wings, with the front pair modified into hardened covers called elytra',
            'which comes in many shapes and sizes, from tiny to very large',
            'which can be found in many habitats, from forests to deserts to water',
            'which feeds on plants or other insects, depending on the species',
            'which can be harmful as pests or beneficial as pollinators or decomposers'
        ],
        [
            'which is a human-powered vehicle with two wheels',
            'which has a frame made of metal or carbon fiber',
            'which has handlebars for steering and brakes for stopping',
            'which has two pedals for propulsion and a chain that connects them to the rear wheel',
            'which can have different styles, such as road, mountain, or BMX',
            'which can be used for transportation, exercise, or sport',
            'which is an environmentally friendly alternative to cars'
        ],
        [
            'which is a container designed for holding liquids or other substances',
            'which can be made of different materials, such as glass, plastic, or metal',
            'which comes in different shapes and sizes, from small to large',
            'which can have different types of caps or lids, such as screw-on or pop-up',
            'which can be used for storing, transporting, or serving drinks or other liquids',
            'which can also be used for other purposes, such as storing spices or cosmetics',
            'which can be recycled to reduce waste'
        ],
        [
            'which is a dish used for serving food',
            'which is typically round and open at the top',
            'which can be made from various materials such as glass, ceramic, or metal',
            'which can be used for holding food or for decoration',
            'which can come in different sizes and shapes',
            'which can have various designs or patterns',
            'which is often used for eating soup or cereal'
        ],
        [
            'which is a young male human',
            'which typically ranges in age from infancy to adolescence',
            'which can have various physical characteristics such as hair color or eye color',
            'which can have various personalities and interests',
            'which can be educated in various settings such as schools or homes',
            'which is a common subject of photographs or artwork'
        ],
        [
            'which is a structure that spans a physical obstacle',
            'which can be made from various materials such as wood, steel, or concrete',
            'which can be designed in various styles such as arch, beam, or suspension',
            'which can have various functions such as carrying vehicles or pedestrians',
            'which can have various lengths and widths',
            'which can have various shapes such as curved or straight',
            'which is a common subject of photographs or artwork'
        ],
        [
            'which is a large motor vehicle designed to carry passengers',
            'which can be powered by various fuels such as gasoline, diesel, or electricity',
            'which can come in various sizes such as minibuses or double-decker buses',
            'which can have various features such as air conditioning or Wi-Fi',
            'which can have various routes and schedules',
            'which is a common mode of transportation in many cities and countries',
            'which is often depicted in popular culture such as movies or TV shows'
        ],
        [
            'which is an insect with large, often colorful wings',
            'which has a unique life cycle that includes metamorphosis from a caterpillar',
            'which can have various wing patterns and shapes',
            'which can have various sizes from small to large',
            'which can be found in various habitats such as meadows or forests',
            'which can have various behaviors such as migration or hibernation',
            'which is a common subject of nature photography or artwork'
        ], [
            'which is a large, even-toed ungulate',
            'which is native to the deserts of Asia and Africa',
            'which has a hump on its back for storing fat',
            'which is known for its ability to travel long distances without water',
            'which has a tough and durable hide',
            'which is often used for transportation and as a source of milk and meat',
            'which has a distinctive long neck and eyelashes',
            'which can close its nostrils to keep out sand'
        ],
        [
            'which is a cylindrical container',
            'which is often made of metal or plastic',
            'which is used for storing and transporting liquids',
            'which can be opened by a tab or a lid',
            'which comes in different sizes and shapes',
            'which can be recycled and reused',
            'which can be filled with beverages, food, or other substances',
            'which is often found in households, stores, and vending machines'
        ],
        [
            'which is a large fortified building',
            'which is often made of stone or brick',
            'which has thick walls, towers, and battlements',
            'which was used for military defense and as a residence for nobility',
            'which often has a moat and a drawbridge',
            'which is associated with medieval Europe',
            'which can be found in ruins or preserved as a historical monument',
            'which is a popular subject in fantasy and adventure stories'
        ],
        [
            'which is a long, worm-like insect',
            'which has a segmented body and many legs',
            'which can be hairy or smooth',
            'which is known for its ability to transform into a butterfly or moth',
            'which often feeds on plants and leaves',
            'which can be found in gardens and forests',
            'which can be harmful to crops and plants',
            'which is a popular subject in children\'s literature'
        ],
        [
            'which is a domesticated mammal',
            'which is raised for meat, milk, and other products',
            'which is often kept on farms and ranches',
            'which can be a source of income and food',
            'which has a distinctive black and white pattern',
            'which is known for its docile and social nature',
            'which can be trained for plowing and transportation',
            'which is a popular subject in rural and pastoral art'
        ],
        [
            'which is a piece of furniture designed for sitting',
            'which typically has a backrest and armrests',
            'which can be made of various materials, such as wood, metal, or plastic',
            'which comes in various styles, such as modern, traditional, or antique',
            'which is often used in homes, offices, and public spaces',
            'which can be adjusted for height, recline, or swivel',
            'which can be upholstered with fabric, leather, or vinyl',
            'which can have additional features, such as footrests, headrests, or massage functions'
        ],
        [
            'which is a great ape native to Africa',
            'which is closely related to humans',
            'which has a highly developed brain and cognitive abilities',
            'which has a black or brown hair covering its body',
            'which has opposable thumbs and feet',
            'which uses tools and communicates with a variety of vocalizations and gestures',
            'which lives in social groups and has complex social behavior',
            'which is an endangered species due to habitat loss and hunting'
        ],
        [
            'which is a time-measuring device',
            'which typically has a circular face with hour and minute hands',
            'which can be analog or digital',
            'which can be powered by batteries or mechanical mechanisms',
            'which comes in various styles, such as wall-mounted, desk, or alarm',
            'which can have additional features, such as date display, stopwatch, or timer',
            'which can be made of various materials, such as plastic, metal, or wood',
            'which is an essential tool for daily life and work'
        ],
        [
            'which is a visible mass of water droplets or ice crystals',
            'which forms in the Earth\'s atmosphere',
            'which can take various shapes, such as cumulus, stratus, or cirrus',
            'which can produce various types of precipitation, such as rain, snow, or hail',
            'which can be affected by various factors, such as temperature, humidity, and wind',
            'which plays a crucial role in the Earth\'s climate and weather patterns',
            'which can be observed and appreciated for its beauty and complexity',
            'which can also have negative impacts, such as causing storms or blocking sunlight'
        ],
        [
            'which is a type of insect',
            'which is known for its ability to survive in various environments',
            'which has a flattened body and long antennae',
            'which can be brown or black in color',
            'which has a fast running speed and ability to climb walls',
            'which feeds on various materials, such as food, paper, or fabrics',
            'which can spread diseases and cause allergies',
            'which can be controlled through various methods, such as pesticides or sanitation'
        ],
        [
            'which is a type of furniture',
            'which is designed for sitting or reclining',
            'which has a backrest and armrests',
            'which is often upholstered',
            'which comes in various sizes and styles',
            'which is commonly found in living rooms',
            'which is made of wood, metal, or other materials'
        ],
        [
            'which is a crustacean',
            'which has a hard exoskeleton',
            'which has two claws, one larger than the other',
            'which has eight legs',
            'which has two stalked eyes',
            'which can regenerate lost limbs',
            'which can live in both saltwater and freshwater',
            'which is often used as seafood'
        ],
        [
            'which is a large reptile',
            'which is found in tropical regions',
            'which has a long snout and sharp teeth',
            'which has tough, scaly skin',
            'which can grow up to 20 feet in length',
            'which is a powerful swimmer',
            'which is known for its ability to ambush prey',
            'which is often hunted for its skin and meat'
        ],
        [
            'which is a small, handheld container',
            'which is used for drinking liquids',
            'which can be made of various materials, such as glass, ceramic, or plastic',
            'which has a handle or a rim for drinking',
            'which comes in different shapes and sizes',
            'which can be decorated with designs or logos',
            'which is often used for coffee, tea, or other beverages',
            'which can be found in homes, restaurants, and cafes'
        ],
        [
            'which is a prehistoric reptile',
            'which lived millions of years ago',
            'which is known for its large size and fierce appearance',
            'which came in various shapes and sizes',
            'which included carnivores and herbivores',
            'which had different types of teeth and jaws',
            'which is often depicted in movies and books',
            'which is studied by paleontologists'
        ],
        [
            'which is a highly intelligent marine mammal',
            'which is known for its playful behavior',
            'which has a streamlined body and dorsal fin',
            'which has a gray or blue-gray coloration',
            'which uses echolocation to navigate and communicate',
            'which can swim at speeds up to 20 miles per hour',
            'which feeds on fish and squid',
            'which is found in oceans around the world'
        ],
        [
            'which is a large land animal',
            'which has a gray skin and long trunk',
            'which has two ivory tusks',
            'which is known for its intelligence and social behavior',
            'which can weigh up to several tons',
            'which feeds on a variety of plant material',
            'which is found in Africa and Asia',
            'which is considered a keystone species'
        ],
        [
            'which is a type of fish',
            'which is flattened sideways and has both eyes on one side of its head',
            'which is found in oceans and coastal waters around the world',
            'which can vary in color from light to dark brown',
            'which is known for its ability to blend in with the seafloor',
            'which can grow up to several feet in length',
            'which feeds on a variety of small marine animals',
            'which is important to commercial and recreational fishing industries'
        ],
        [
            'which is a large area covered with trees and undergrowth',
            'which is an important habitat for many species of plants and animals',
            'which can be classified as tropical, temperate, or boreal',
            'which provides a range of ecosystem services, including carbon storage and water regulation',
            'which is threatened by deforestation and habitat fragmentation',
            'which is home to many indigenous communities',
            'which is a source of inspiration for art, literature, and spirituality',
            'which is essential for the health and well-being of the planet'
        ],
        [
            'which is a small to medium-sized carnivorous mammal',
            'which has a pointed snout, bushy tail, and triangular ears',
            'which can be found in a variety of habitats, including forests, grasslands, and deserts',
            'which is known for its intelligence and adaptability',
            'which is an important predator in many ecosystems',
            'which feeds on a variety of prey, including rodents, birds, and insects',
            'which can live in social groups or as solitary individuals',
            'which has been the subject of mythology and folklore in many cultures'
        ],
        [
            'which is a young female human',
            'which is typically under 18 years old',
            'which has long hair and feminine features',
            'which often wears dresses or skirts',
            'which is known for its social and emotional intelligence',
            'which can be interested in a variety of hobbies and activities',
            'which is an important part of human families and societies',
            'which can grow up to become women with diverse careers and aspirations'
        ],
        [
            'which is a small rodent',
            'which is native to Syria and surrounding countries',
            'which has a stocky body and short tail',
            'which has large cheek pouches for storing food',
            'which is known for its ability to run and climb quickly',
            'which often feeds on seeds and vegetables',
            'which can be kept as a popular pet',
            'which can be trained to perform tricks and tasks'
        ],
        [
            'which is a building designed for people to live in',
            'which can have different sizes and shapes',
            'which can be made of various materials',
            'which often has multiple rooms and floors',
            'which can be decorated and furnished in many different styles',
            'which provides shelter and comfort for its inhabitants',
            'which can be owned or rented by individuals or families',
            'which can be an important investment and source of wealth'
        ],
        [
            'which is a marsupial mammal',
            'which is native to Australia',
            'which has powerful hind legs and a long tail',
            'which can hop long distances at high speeds',
            'which carries its young in a pouch',
            'which is known for its unique appearance and behavior',
            'which often feeds on grass and leaves',
            'which can be found in a variety of habitats, from forests to deserts'
        ],
        [
            'which is an input device for computers',
            'which is used for typing and entering data',
            'which has a set of keys for letters, numbers, and symbols',
            'which can have different layouts and designs',
            'which is known for its role in computing and communication',
            'which often comes with other accessories like a mouse and monitor',
            'which can be used for a variety of tasks, from writing to gaming',
            'which is an essential tool for many people in modern society'
        ],
        [
            'which is a device that produces light',
            'which can come in many shapes and sizes',
            'which can be made of various materials',
            'which often has a lampshade to diffuse and direct light',
            'which is known for its ability to create ambiance and atmosphere',
            'which often comes with different light settings and features',
            'which can be powered by electricity or batteries',
            'which is a common household item and decorative element'
        ],
        [
            'which is a machine used for cutting grass',
            'which can come in different sizes and styles',
            'which can be powered by electricity, gas, or manual force',
            'which has a rotating blade for cutting grass to a uniform length',
            'which is known for its noise and efficiency',
            'which often comes with different cutting settings and features',
            'which is essential for maintaining lawns and gardens',
            'which can be a source of pride and satisfaction for homeowners'
        ],
        [
            'which is a large wild cat',
            'which has a distinctive yellow or gold coat with black spots',
            'which is a fierce predator',
            'which is known for its strength and agility',
            'which can run at speeds up to 60 km/h (37 mph)',
            'which is found in Africa and Asia',
            'which is an endangered species',
            'which is sometimes kept as a pet'
        ],
        [
            'which is a large carnivorous cat',
            'which has a distinctive mane of hair around its neck',
            'which has a tawny coat with white underparts',
            'which is a fierce predator',
            'which is known for its strength and courage',
            'which is the second-largest living cat after the tiger',
            'which is found in Africa and some parts of Asia',
            'which is sometimes kept as a pet'
        ],
        [
            'which is a reptile',
            'which has a long and slender body',
            'which has a scaly skin',
            'which can change color to match its surroundings',
            'which has a long tail and sharp claws',
            'which is a cold-blooded animal',
            'which can regenerate lost limbs',
            'which is found in many parts of the world'
        ],
        [
            'which is a marine crustacean',
            'which has a hard exoskeleton and ten legs',
            'which has two large claws for catching prey and defense',
            'which is a bottom-dweller',
            'which is found in many parts of the world',
            'which is an important seafood',
            'which is often cooked live',
            'which can live up to 100 years'
        ],
        [
            'which is a bipedal mammal',
            'which has a highly developed brain',
            'which is capable of language, culture, and technology',
            'which has opposable thumbs',
            'which is found in many parts of the world',
            'which is the only surviving member of the genus Homo',
            'which is capable of abstract thinking and self-awareness',
            'which has a complex social structure'
        ],
        [
            'which is a deciduous tree',
            'which has a characteristic five-lobed leaf',
            'which has a woody stem and branches',
            'which produces a sweet sap used for making maple syrup',
            'which is native to North America and parts of Europe',
            'which is a popular ornamental tree',
            'which turns bright red, orange, and yellow in the fall',
            'which can live up to 300 years'
        ],
        [
            'which is a two-wheeled vehicle',
            'which has an engine and a motor',
            'which is designed for speed and agility',
            'which has a frame made of metal or other materials',
            'which has handlebars for steering',
            'which is a popular mode of transportation',
            'which is often used for recreational purposes',
            'which comes in many different styles and models'
        ],
        [
            'which is a large landform that rises steeply above its surroundings',
            'which often has a rocky summit or peak',
            'which can be formed by tectonic forces or erosion',
            'which is often covered in snow and ice at high elevations',
            'which can be a popular destination for hiking and climbing',
            'which can have unique ecosystems and wildlife adapted to its altitude'
        ],
        [
            'which is a small mammal',
            'which is found in many parts of the world',
            'which has a pointed snout and large ears',
            'which has a long tail and soft fur',
            'which is known for its ability to squeeze through small openings',
            'which often feeds on seeds, fruits, and insects',
            'which can be a common household pest'
        ],
        [
            'which is a type of fungus',
            'which comes in many different shapes and sizes',
            'which can be found in many habitats, including forests and lawns',
            'which often has a cap and stem',
            'which reproduces by releasing spores',
            'which can be edible or poisonous',
            'which can be used in cooking or medicine',
            'which plays an important role in the ecosystem by breaking down organic matter'
        ],
        [
            'which is a type of tree',
            'which is found in many parts of the world',
            'which can grow to be very large and old',
            'which has a strong and durable wood',
            'which has a distinctive lobed leaf shape',
            'which produces acorns',
            'which is important for wildlife such as squirrels and deer',
            'which can be a symbol of strength and longevity'
        ],
        [
            'which is a type of citrus fruit',
            'which is native to Southeast Asia',
            'which is known for its sweet and tangy flavor',
            'which has a tough and pitted skin',
            'which is often peeled and eaten fresh',
            'which is rich in vitamin C',
            'which can be used in cooking, baking, and beverages',
            'which can be a popular flavor for candies and desserts'
        ],
        [
            'which is a type of flowering plant',
            'which comes in many different shapes and colors',
            'which is often grown for its showy and fragrant blooms',
            'which can be found in many habitats, including tropical and temperate regions',
            'which reproduces by seeds or vegetative propagation',
            'which can be a popular gift or decoration',
            'which has cultural significance in many societies',
            'which can be used in perfume and medicine'
        ],
        [
            'which is a semi-aquatic mammal',
            'which is found in many parts of the world',
            'which has a long and streamlined body',
            'which has a thick and waterproof fur',
            'which has webbed feet for swimming',
            'which feeds on fish, crustaceans, and other aquatic animals',
            'which is known for its playful behavior',
            'which can be a symbol of clean water and healthy ecosystems'
        ],
        [
            'which is a tropical tree',
            'which has a tall, slender trunk',
            'which has long, feathery leaves',
            'which often grows near beaches and oceans',
            'which produces coconuts',
            'which is a symbol of tropical vacations',
            'which is often used in landscaping'
        ],
        [
            'which is a type of fruit tree',
            'which has a round, bulbous shape',
            'which has a green or yellow skin',
            'which has a sweet, juicy flesh',
            'which is high in fiber and vitamin C',
            'which is often eaten fresh or used in cooking',
            'which is a symbol of health and abundance'
        ],
        [
            'which is a type of vehicle',
            'which has an open cargo area at the back',
            'which is designed for hauling goods and equipment',
            'which often has a powerful engine and rugged suspension',
            'which is popular in rural and industrial settings',
            'which is often used for construction and farming',
            'which is a symbol of practicality and utility'
        ],
        [
            'which is a type of coniferous tree',
            'which has a tall, straight trunk',
            'which has long, needle-like leaves',
            'which often grows in colder regions',
            'which produces cones',
            'which is often used for lumber and paper production',
            'which is a symbol of evergreen forests'
        ],
        [
            'which is a vast and flat expanse of land',
            'which has no significant changes in elevation',
            'which is often covered in grasses or crops',
            'which can be found in various regions',
            'which is an important agricultural resource',
            'which can be used for transportation and recreation',
            'which is a symbol of simplicity and openness'
        ],
        [
            'which is a type of dishware',
            'which is used for serving and eating food',
            'which can be made of various materials such as ceramic, glass, or plastic',
            'which can have different shapes and sizes',
            'which is often decorated with patterns or designs',
            'which is an essential item in dining settings',
            'which is a symbol of hospitality and socializing'
        ],
        [
            'which is a type of flowering plant',
            'which has delicate, papery petals',
            'which can be found in various colors',
            'which often grows in fields and meadows',
            'which is a symbol of beauty and remembrance',
            'which is often used in floral arrangements',
            'which has medicinal and culinary uses'
        ],
        [
            'which is a nocturnal rodent',
            'which is covered in sharp quills for defense',
            'which has a stocky body and short legs',
            'which is an excellent climber',
            'which feeds on bark, leaves, and fruits',
            'which is found in forests and deserts of North America',
            'which can be kept as a pet',
            'which is often hunted for its meat and quills'
        ],
        [
            'which is a small to medium-sized marsupial',
            'which is native to Australia and nearby islands',
            'which has a pointed snout and prehensile tail',
            'which is a skilled climber and swimmer',
            'which plays dead when threatened',
            'which feeds on fruits, insects, and small animals',
            'which can be found in urban areas and forests',
            'which can carry diseases such as leptospirosis and tuberculosis'
        ],
        [
            'which is a small mammal',
            'which is found in various habitats around the world',
            'which has long ears and powerful hind legs',
            'which is a fast runner and jumper',
            'which can have many different coat colors and patterns',
            'which feeds on grasses, hay, and vegetables',
            'which is often kept as a pet',
            'which can be hunted for its meat and fur'
        ],
        [
            'which is a medium-sized mammal',
            'which is native to North America',
            'which has a distinctive black mask around its eyes',
            'which has a bushy tail and dexterous front paws',
            'which is a skilled climber and swimmer',
            'which feeds on fruits, nuts, and small animals',
            'which is often found in urban areas and forests',
            'which can carry diseases such as rabies and leptospirosis'
        ],
        [
            'which is a type of fish',
            'which has a flattened body and wide, wing-like fins',
            'which is found in oceans and rivers around the world',
            'which can deliver electric shocks to stun prey and defend itself',
            'which feeds on small fish and crustaceans',
            'which can be a popular aquarium fish',
            'which can be caught for its meat and cartilage',
            'which can be threatened by overfishing and habitat destruction'
        ],
        [
            'which is a paved surface for vehicles',
            'which is often used for transportation and travel',
            'which can be made of asphalt, concrete, or other materials',
            'which can have many different designs and markings',
            'which can be straight, curved, or winding',
            'which can have different speed limits and traffic rules',
            'which can be congested during rush hour and accidents',
            'which can be maintained by road crews and government agencies'
        ],
        [
            'which is a type of vehicle that travels through space',
            'which is propelled by rockets or other engines',
            'which can carry people, cargo, or scientific equipment',
            'which can reach high speeds and altitudes',
            'which can be used for exploration, communication, or military purposes',
            'which can have different shapes and sizes',
            'which can be launched from the ground, air, or water',
            'which can be controlled by ground crews or on-board computers'
        ],
        [
            'which is a type of flowering plant',
            'which has fragrant petals',
            'which comes in many different colors, including red, pink, and white',
            'which is often used in gardens and as cut flowers',
            'which has thorny stems',
            'which is a symbol of love and beauty',
            'which is used in perfumes and cosmetics',
            'which has edible petals that can be used in cooking and as a garnish'
        ],
        [
            'which is a large body of saltwater',
            'which covers more than 70% of the Earth\'s surface',
            'which is home to a wide variety of marine life',
            'which has currents, tides, and waves',
            'which is used for transportation, commerce, and recreation',
            'which is affected by climate change and pollution',
            'which has many different zones, including the intertidal, pelagic, and benthic zones',
            'which is an important source of food and resources for humans'
        ],
        [
            'which is a marine mammal',
            'which is found in both the Arctic and Antarctic regions',
            'which has a thick layer of blubber for insulation',
            'which can swim and dive for long periods of time',
            'which feeds on fish and other marine life',
            'which is often hunted for its meat, fur, and oil',
            'which has a distinctive bark and is sometimes called a sea dog',
            'which can be trained to perform tricks and entertain audiences'
        ],
        [
            'which is a type of predatory fish',
            'which has a cartilaginous skeleton and five to seven gill slits',
            'which has a streamlined body and powerful jaws',
            'which comes in many different species, including the great white, tiger, and hammerhead sharks',
            'which is often feared and respected by humans',
            'which plays an important role in ocean ecosystems',
            'which is threatened by overfishing and habitat destruction',
            'which has a reputation as a dangerous predator but is not usually interested in attacking humans'
        ],
        [
            'which is a small, insect-eating mammal',
            'which has a pointed snout and tiny eyes',
            'which is found in many different habitats, including forests, fields, and gardens',
            'which has a high metabolism and must eat frequently to survive',
            'which is active both day and night',
            'which is not a rodent, but is often mistaken for one',
            'which is known for its rapid movements and high-pitched squeaks',
            'which is an important part of many ecosystems'
        ],
        [
            'which is a small mammal',
            'which is native to North and South America',
            'which has black and white fur and a distinctive odor',
            'which can spray a foul-smelling liquid as a defense mechanism',
            'which is often hunted for its fur',
            'which is sometimes kept as a pet',
            'which is an important part of many ecosystems',
            'which is sometimes associated with disease and parasites'
        ],
        [
            'which is a tall building with multiple floors',
            'which is typically found in urban areas',
            'which is made of steel and concrete',
            'which is used for office, residential or commercial purposes',
            'which often has a modern and sleek design',
            'which is typically over 100 meters in height',
            'which requires elevators to move between floors',
            'which can be a symbol of a city or country'
        ],
        [
            'which is a small mollusk with a spiral shell',
            'which moves slowly and leaves a slime trail',
            'which feeds on plants and other debris',
            'which has a soft body and a protective shell',
            'which can retract its body into the shell for protection',
            'which is hermaphroditic and can reproduce on its own',
            'which can be found in gardens, forests, and other moist habitats',
            'which is considered a delicacy in some cuisines'
        ],
        [
            'which is a long, legless reptile',
            'which is found in a variety of habitats',
            'which has a scaly skin and no eyelids',
            'which can be venomous or non-venomous',
            'which feeds on small mammals, birds, and other reptiles',
            'which can sense vibrations and heat with its tongue',
            'which can shed its skin as it grows',
            'which has a forked tongue and can flick it to sense its surroundings'
        ],
        [
            'which is an eight-legged arachnid',
            'which has a small body and long legs',
            'which can spin webs to catch prey',
            'which can be venomous or non-venomous',
            'which feeds on insects and other small animals',
            'which can regenerate lost limbs',
            'which has multiple pairs of eyes for better vision',
            'which is often associated with Halloween and horror movies'
        ],
        [
            'which is a small to medium-sized mammal',
            'which has a bushy tail and a slender body',
            'which is found in a variety of habitats',
            'which feeds on nuts, seeds, and fruits',
            'which can climb trees and run on the ground',
            'which hibernates during the winter',
            'which is known for storing food in the ground',
            'which is often seen in parks and suburban neighborhoods'
        ],
        [
            'which is a public transportation vehicle',
            'which runs on tracks or rails',
            'which is typically powered by electricity',
            'which can carry passengers or freight',
            'which has multiple cars or compartments',
            'which can operate on city streets or dedicated lines',
            'which often has a bell or horn to signal its presence',
            'which is a common sight in urban areas'
        ],
        [
            'which is a tall flowering plant',
            'which has a large head with many petals',
            'which is typically yellow in color',
            'which can grow up to 3 meters in height',
            'which is used for food, oil, and decoration',
            'which is an important crop in many countries',
            'which can be grown in gardens or fields',
            'which is often associated with summer and warmth'
        ],
        [
            'which is a type of vegetable',
            'which is often used in cooking and salads',
            'which comes in a variety of colors, such as green, red, and yellow',
            'which has a sweet taste',
            'which can be eaten raw or cooked',
            'which is a good source of vitamins and minerals'
        ],
        [
            'which is a piece of furniture',
            'which is used for eating, working, or playing games',
            'which can be made of various materials, such as wood, glass, or metal',
            'which comes in different shapes and sizes',
            'which can be a focal point of a room',
            'which can be decorated with tablecloths, centerpieces, and place settings'
        ],
        [
            'which is a large military vehicle',
            'which is used for transporting soldiers, weapons, and supplies',
            'which has thick armor and tracks for mobility',
            'which can be armed with weapons, such as guns and missiles',
            'which can be used for offensive and defensive operations',
            'which has been used in many wars and conflicts'
        ],
        [
            'which is a communication device',
            'which is used for speaking with others over a distance',
            'which can be used for making local and long-distance calls',
            'which comes in various types, such as landline, mobile, and cordless',
            'which has a keypad or touch screen for dialing',
            'which can be used for texting, emailing, and browsing the internet'
        ],
        [
            'which is an electronic device',
            'which is used for watching television shows and movies',
            'which has a screen for displaying images and videos',
            'which can receive signals from antennas, cables, or satellites',
            'which can have various features, such as built-in speakers, internet connectivity, and voice control',
            'which can be mounted on a wall or placed on a stand'
        ],
        [
            'which is a large carnivorous animal',
            'which is known for its distinctive orange and black stripes',
            'which is native to Asia',
            'which is the largest cat species in the world',
            'which is an apex predator and can hunt animals much larger than itself',
            'which is an endangered species due to habitat loss and poaching'
        ],
        [
            'which is a heavy agricultural vehicle',
            'which is used for plowing, tilling, and harvesting crops',
            'which can be powered by diesel or gasoline engines',
            'which has large wheels or tracks for mobility',
            'which has a cab for the driver and passengers',
            'which can have various attachments, such as plows, cultivators, and sprayers'
        ],
        [
            'which is a mode of transportation',
            'which runs on tracks',
            'which consists of one or more cars',
            'which can travel long distances',
            'which is powered by steam, diesel, or electricity',
            'which can carry passengers and/or cargo',
            'which is often used for commuting, travel, or shipping',
            'which has a distinct shape and sound'
        ],
        [
            'which is a type of freshwater fish',
            'which is found in rivers and streams',
            'which has a sleek, streamlined body',
            'which has a distinctive pattern of spots and/or stripes',
            'which can range in size from small to large',
            'which is a popular game fish',
            'which can be caught using bait or lures',
            'which is often eaten as food'
        ],
        [
            'which is a type of flowering plant',
            'which is native to Eurasia and North Africa',
            'which has a bulbous root and long stem',
            'which has large, showy flowers',
            'which can be red, pink, yellow, or white in color',
            'which blooms in the spring and summer',
            'which is often grown in gardens and parks',
            'which is a symbol of love and affection'
        ],
        [
            'which is a type of reptile',
            'which has a bony or cartilaginous shell',
            'which can retract its head and limbs into its shell',
            'which is cold-blooded and lays eggs',
            'which has a beak-like mouth and sharp claws',
            'which can range in size from small to large',
            'which is found in both water and on land',
            'which is often kept as a pet'
        ],
        [
            'which is a type of furniture',
            'which is used for storing clothes and accessories',
            'which can come in many styles and sizes',
            'which can be made of wood, metal, or plastic',
            'which can have drawers, shelves, or hanging space',
            'which is often found in bedrooms or dressing rooms',
            'which can be customized or built-in',
            'which is an essential piece of household furniture'
        ],
        [
            'which is a large marine mammal',
            'which is part of the cetacean family',
            'which breathes air through a blowhole on top of its head',
            'which has a streamlined body and flippers',
            'which has a wide variety of species, including the blue whale, humpback whale, and killer whale',
            'which is known for its songs and vocalizations',
            'which feeds on krill, plankton, and small fish',
            'which is highly intelligent and social'
        ],
        [
            'which is a deciduous tree',
            'which is native to temperate regions of the Northern Hemisphere',
            'which has long, narrow leaves with a distinctive silver underside',
            'which produces catkins in early spring',
            'which is often found near bodies of water, such as rivers and streams',
            'which has a long history of medicinal and cultural uses',
            'which can be used for basketry and other crafts',
            'which provides habitat for a wide variety of animals'
        ],
        [
            'which is a carnivorous mammal',
            'which is part of the Canidae family',
            'which is native to wild and remote areas of North America, Europe, and Asia',
            'which has a distinctive howl and other vocalizations',
            'which lives and hunts in packs, led by an alpha male and female',
            'which feeds on a variety of prey, including deer, elk, and small mammals',
            'which is highly intelligent and adaptable',
            'which is the ancestor of the domestic dog'
        ],
        [
            'which is a female human',
            'which is part of the Homo sapiens species',
            'which has a variety of physical features, including breasts and a wider pelvis than males',
            'which has a longer lifespan than males',
            'which has played a significant role in human history and culture',
            'which has the ability to give birth and breastfeed',
            'which has a wide range of abilities and accomplishments',
            'which has faced discrimination and inequality in many societies'
        ],
        [
            'which is an invertebrate animal',
            'which belongs to the phylum Annelida',
            'which has a long, slender body with no legs or arms',
            'which has a moist, slimy skin',
            'which burrows through soil and feeds on decaying organic matter',
            'which plays a crucial role in soil health and nutrient cycling',
            'which can also be used as a food source for birds and other animals',
            'which comes in many different species, including earthworms and marine worms'
        ]
    ]
    suffix = '.'
    num_classes = 100

    def __init__(self, dataset, root_dir, transform, noise_mode='sym', mode='train',
                 noise_ratio=0.5):  # , noise_file=None):

        self.r = noise_ratio  # total noise ratio
        self.transform = transform
        self.mode = mode
        self.closed_noise = None

        noise_file = f'{root_dir}/{noise_ratio}_{noise_mode}_noise.json'
        if self.mode == 'test':
            cifar_dic = unpickle('%s/cifar-100-python/test' % root_dir)
            self.cifar_data = cifar_dic['data'].reshape((10000, 3, 32, 32)).transpose((0, 2, 3, 1))
            self.label = cifar_dic['fine_labels']

        elif self.mode == 'train':
            cifar_dic = unpickle('%s/cifar-100-python/train' % root_dir)
            cifar_label = cifar_dic['fine_labels']
            self.cifar_data = cifar_dic['data'].reshape((50000, 3, 32, 32)).transpose((0, 2, 3, 1))
            self.clean_label = cifar_label

            if os.path.exists(noise_file):
                noise = json.load(open(noise_file, "r"))
                noise_labels = noise['noise_labels']
                self.closed_noise = noise['closed_noise']
                self.label = noise_labels
            else:
                # inject noise
                noise_labels = []  # all labels (some noisy, some clean)
                idx = list(range(50000))  # indices of cifar dataset
                random.shuffle(idx)
                num_total_noise = int(self.r * 50000)  # total amount of noise
                print('Statistics of synthetic noisy CIFAR dataset: ', 'num of clean samples: ',
                      50000 - num_total_noise,
                      ' num of closed-set noise: ', num_total_noise)
                self.closed_noise = idx[:num_total_noise]  # closed set noise indices
                # populate noise_labels
                for i in range(50000):
                    if i in self.closed_noise:
                        if noise_mode == 'sym':
                            noiselabel = random.randint(0, 99)
                        else:  # if noise_mode == 'asym':
                            raise ValueError(f'Asym noise mode is not supported for CIFAR100 in this project.')
                            # noiselabel = self.transition[cifar_label[i]]
                        noise_labels.append(noiselabel)
                    else:
                        noise_labels.append(cifar_label[i])

                # write noise to a file, to re-use
                noise = {'noise_labels': noise_labels, 'closed_noise': self.closed_noise}
                print("save noise to %s ..." % noise_file)
                json.dump(noise, open(noise_file, "w"))
                self.label = noise_labels
                # self.open_id = np.array(self.open_noise)[:, 0] if len(self.open_noise) !=0 else None

        else:
            raise ValueError(f'Dataset mode should be train or test rather than {self.mode}!')

    def update_labels(self, new_label):
        self.label = new_label.cpu()

    def __getitem__(self, index):
        img = self.cifar_data[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        target = self.label[index]
        return img, target, index

    def __len__(self):
        return len(self.cifar_data)

    def get_noise(self):
        return self.closed_noise

    def __repr__(self):
        return f'dataset_mode: {self.mode}, dataset number: {len(self)} \n'
