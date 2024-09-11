import codecs
import json
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class red_mini_imagenet_dataset(Dataset):
    class_names = [['triceratops'],
                   ['upright piano'],
                   ['Gordon setter'],
                   ['cocktail shaker'],
                   ['unicycle', 'monocycle'],
                   ['organ', 'pipe organ'],
                   ['Alaskan malamute'],
                   ['prayer rug'],
                   ['Newfoundland dog'],
                   ['tobacco shop'],
                   ['ladybug'],
                   ['combination lock'],
                   ['ashcan', 'trash can'],
                   ['American robin'],
                   ['scoreboard'],
                   ['dome'],
                   ['iPod'],
                   ['one - armed bandit'],
                   ['miniskirt'],
                   ['French bulldog'],
                   ['carton'],
                   ['Tibetan mastiff'],
                   ['pencil box'],
                   ['king crab', 'Alaska crab'],
                   ['horizontal bar', 'high bar'],
                   ['spider web'],
                   ['electric guitar'],
                   ['meerkat', 'mierkat'],
                   ['file cabinet'],
                   ['consomme'],
                   ['jellyfish'],
                   ['cuirass'],
                   ['black - footed ferret'],
                   ['school bus'],
                   ['miniature poodle'],
                   ['catamaran'],
                   ['snorkel'],
                   ['oboe'],
                   ['worm fence', 'snake fence'],
                   ['African hunting dog'],
                   ['golden retriever'],
                   ['carousel', 'carrousel'],
                   ['aircraft carrier'],
                   ['photocopier'],
                   ['Arctic fox', 'white fox'],
                   ['hair slide'],
                   ['tile roof'],
                   ['Ibizan hound', 'Ibizan Podenco'],
                   ['toucan'],
                   ['house finch'],
                   ['poncho'],
                   ['trifle'],
                   ['hourglass'],
                   ['fire screen', 'fireguard'],
                   ['white wolf'],
                   ['street sign'],
                   ['solar dish', 'solar collector'],
                   ['rock beauty'],
                   ['komondor'],
                   ['bookshop'],
                   ['crate'],
                   ['theater curtain'],
                   ['tank', 'army tank'],
                   ['dugong'],
                   ['dalmatian'],
                   ['ear', 'fruit'],
                   ['missile'],
                   ['bolete'],
                   ['orange'],
                   ['vase'],
                   ['Walker hound'],
                   ['lion'],
                   ['three - toed sloth'],
                   ['lipstick'],
                   ['coral reef'],
                   ['reel'],
                   ['beer bottle'],
                   ['green mamba'],
                   ['frying pan'],
                   ['wok'],
                   ['goose'],
                   ['rhinoceros beetle'],
                   ['yawl'],
                   ['clog'],
                   ['Saluki Hund'],
                   ['chime', 'bell', 'gong'],
                   ['stage'],
                   ['boxer'],
                   ['cliff'],
                   ['ant'],
                   ['cannon'],
                   ['harvestman'],
                   ['mixing bowl'],
                   ['nematode'],
                   ['parallel bars'],
                   ['garbage truck'],
                   ['holster'],
                   ['barrel'],
                   ['hotdog'],
                   ['dishrag']]
    detailed_features = [
        [
            'which is a genus of large herbivorous dinosaurs',
            'which lived during the Late Cretaceous period',
            'which had a distinctive frill and three horns on its head',
            'which is known for its massive size and strength',
            'which is believed to have been a social animal',
            'which likely migrated to different areas to find food',
            'which is now extinct'
        ],
        [
            'which is a type of piano',
            'which has a vertical frame and strings',
            'which is often used in classical music',
            'which can be found in homes, schools, and concert halls',
            'which has pedals that control sustain and softness',
            'which has a wide range of notes and dynamics',
            'which is played sitting down with both hands',
            'which can be a centerpiece of a room'
        ],
        [
            'which is a breed of hunting dog',
            'which originated in Scotland',
            'which has a black and tan coat',
            'which has long ears and a bushy tail',
            'which is known for its loyalty and intelligence',
            'which is often used for bird hunting and field trials',
            'which requires regular exercise and grooming',
            'which can make a good family pet'
        ],
        [
            'which is a container for mixing and serving cocktails',
            'which is typically made of metal or glass',
            'which has a lid and a strainer',
            'which can be used to make a variety of drinks',
            'which is often used in bars and restaurants',
            'which can be a decorative item for a home bar',
            'which can come in different sizes and shapes',
            'which can be used for non-alcoholic drinks as well'
        ],
        [
            'which is a single-wheeled vehicle',
            'which is propelled by pedals or by the rider\'s balance',
            'which requires skill and balance to ride',
            'which is often used for entertainment or sport',
            'which can be ridden on flat or uneven surfaces',
            'which is lightweight and portable',
            'which can be used for transportation in some situations',
            'which is a unique and fun way to get around'
        ],
        [
            'which is a musical instrument',
            'which produces sound by air flowing through pipes',
            'which can have a wide range of tones and timbres',
            'which is often used in churches and concert halls',
            'which can have multiple keyboards and pedals',
            'which requires skill and practice to play',
            'which can be a centerpiece of a room',
            'which can be found in different styles and sizes'
        ],
        [
            'which is a breed of domestic dog',
            'which originated in Alaska',
            'which has a thick fur coat and bushy tail',
            'which is bred for strength and endurance',
            'which is often used for sled pulling and racing',
            'which is known for its loyalty and affection',
            'which requires regular exercise and grooming',
            'which can make a good family pet'
        ],
        [
            'which is a type of rug used by Muslims for prayer',
            'which has a decorative design with a mihrab, or prayer niche, at one end',
            'which is made from various materials such as wool, silk, or cotton',
            'which is often colorful and intricately patterned',
            'which is used for kneeling and prostrating during prayer',
            'which is an important religious and cultural symbol'
        ],
        [
            'which is a large breed of working dog',
            'which is known for its strength, intelligence, and loyalty',
            'which has a thick waterproof coat that can be black, brown, or gray',
            'which is a strong swimmer and has webbed feet',
            'which is often used for water rescue and as a draft animal',
            'which is a gentle and patient family dog',
            'which requires regular exercise and grooming'
        ],
        [
            'which is a store that specializes in selling tobacco products',
            'which may also sell smoking accessories and related items',
            'which often has a distinctive smell and atmosphere',
            'which may also provide a place for customers to smoke',
            'which is subject to various regulations and taxes',
            'which may be a source of controversy and health concerns'
        ],
        [
            'which is a small beetle',
            'which is often brightly colored with black spots',
            'which has a hard shell that protects it from predators',
            'which has six legs and two pairs of wings',
            'which feeds on aphids and other small insects',
            'which is considered a beneficial insect for gardeners',
            'which has a distinctive round shape and small size'
        ],
        [
            'which is a type of lock that requires a specific sequence of numbers or symbols to open',
            'which is often used to secure safes, doors, or luggage',
            'which may have a rotating dial or push buttons',
            'which is more secure than a traditional key lock',
            'which can be difficult to pick or bypass',
            'which is commonly used in schools, gyms, and public spaces',
            'which can be opened with a combination that only the owner knows'
        ],
        [
            'which is a type of garbage can or container',
            'which is often made of metal or plastic',
            'which is used to hold household waste or refuse',
            'which may have a lid or pedal-operated opening mechanism',
            'which is emptied by garbage collectors or sanitation workers',
            'which can help reduce odors and keep homes and streets clean',
            'which can be decorated or painted for aesthetic purposes'
        ],
        [
            'which is a small songbird',
            'which is native to North America',
            'which has a reddish-orange breast and a gray back',
            'which has a distinctive white eye ring and black head',
            'which is known for its melodious song and cheerful personality',
            'which often feeds on insects, fruits, and seeds',
            'which is a common backyard bird that can be attracted to bird feeders'
        ],
        [
            'which is used to display scores and game information',
            'which is often found in sports arenas and stadiums',
            'which is usually electronic and has bright colors',
            'which displays the names of the teams and the current score',
            'which can also display the time remaining and other statistics',
            'which is operated by a computer or a scoreboard operator'
        ],
        [
            'which is a hemispherical roof or ceiling',
            'which is often used to cover a building or a sports stadium',
            'which can be made of various materials, such as glass or metal',
            'which is often used to provide natural light or a panoramic view',
            'which is often used to create an impressive architectural feature',
            'which can also be used to control the temperature and lighting inside the building'
        ],
        [
            'which is a portable digital media player',
            'which is designed and marketed by Apple Inc.',
            'which can play music, videos, podcasts, and games',
            'which has a touch-sensitive display and a user-friendly interface',
            'which can be synced with a computer or the internet',
            'which can store thousands of songs and other media files',
            'which can also be used to access the internet and social media'
        ],
        [
            'which is a gambling machine',
            'which is also known as a slot machine or a fruit machine',
            'which has a lever or a button to activate the spinning reels',
            'which has various symbols, such as fruits or numbers, on the reels',
            'which pays out a prize if the symbols line up in a certain way',
            'which is often found in casinos or other gambling establishments',
            'which is designed to generate revenue for the operator'
        ],
        [
            'which is a type of women\'s clothing',
            'which is characterized by its short length',
            'which usually falls above the knee',
            'which can be made of various materials, such as denim or cotton',
            'which is often worn in warm weather or for formal occasions',
            'which can be paired with a variety of tops and accessories',
            'which can also be worn with tights or leggings'
        ],
        [
            'which is a small domestic dog breed',
            'which originated in France',
            'which has a short and smooth coat',
            'which has a distinctive "bat" ears and a squished face',
            'which is known for its affectionate and playful nature',
            'which is a popular companion dog',
            'which requires minimal exercise and grooming'
        ],
        [
            'which is a type of container',
            'which is often made of cardboard or paper',
            'which is used to store or transport goods',
            'which can be folded flat when not in use',
            'which can be printed with various designs and logos',
            'which can be recycled or reused',
            'which comes in various sizes and shapes'
        ],
        [
            'which is a large and powerful dog breed',
            'which originated in Tibet',
            'which has a thick and long coat',
            'which has a heavy and muscular build',
            'which is known for its protective and loyal nature',
            'which is used for guarding livestock and homes',
            'which requires daily exercise and grooming',
            'which can weigh over 100 pounds'
        ],
        [
            'which is a rectangular container for pencils, pens, and other writing utensils',
            'which is typically made of wood, plastic, or metal',
            'which may have compartments or a single open space',
            'which may have a hinged lid or slide-out tray',
            'which is often used by students and professionals',
            'which can be personalized or decorated',
            'which can be used for storing other small items'
        ],
        [
            'which is a large and spiny crustacean',
            'which is found in the oceans around the world',
            'which has a hard exoskeleton and five pairs of legs',
            'which has large claws used for defense and feeding',
            'which is considered a delicacy in many cultures',
            'which can weigh up to 20 pounds',
            'which is typically cooked and served whole',
            'which is harvested using traps and pots'
        ],
        [
            'which is a piece of gymnastics equipment',
            'which is used for practicing and performing various skills',
            'which consists of a horizontal bar supported by two vertical bars',
            'which can be adjusted in height and width',
            'which requires strength, balance, and coordination',
            'which is used in competitions such as the Olympics',
            'which can be used for swinging, releasing, and catching',
            'which can be used by both men and women'
        ],
        [
            'which is a structure created by spiders',
            'which is used for catching prey and sheltering spider eggs',
            'which is made of silk produced by the spider',
            'which can be various shapes and sizes',
            'which can be found in trees, bushes, and buildings',
            'which can be harmful or beneficial to humans',
            'which can be used for scientific research and artistic inspiration',
            'which can be destroyed or preserved by human activities'
        ],
        [
            'which is a stringed musical instrument',
            'which is used in many genres of music',
            'which has a solid body or hollow body',
            'which has a long neck and fretted fingerboard',
            'which has one or more pickups for amplification',
            'which can be played with fingers or a pick',
            'which can produce a wide range of sounds and effects',
            'which is often associated with rock and roll'
        ],
        [
            'which is a small carnivorous mammal',
            'which is found in southern Africa',
            'which has a slender body and pointed face',
            'which has dark bands around the eyes',
            'which is known for its social behavior and sentinel duty',
            'which lives in groups called mobs or gangs',
            'which feeds on insects, small mammals, and reptiles',
            'which is featured in popular culture and nature documentaries'
        ],
        [
            'which is a piece of office furniture',
            'which is used for storing files and documents',
            'which is typically made of metal or wood',
            'which has several drawers for organizing files',
            'which may have a locking mechanism for security',
            'which is often found in offices and workspaces',
            'which can be purchased in various sizes and styles'
        ],
        [
            'which is a clear soup',
            'which is typically made with meat or fish broth',
            'which is often served as a starter or appetizer',
            'which may include vegetables, meat, or noodles',
            'which is often garnished with herbs or croutons',
            'which is often served hot',
            'which is a popular dish in French cuisine'
        ],
        [
            'which is a type of sea creature',
            'which is part of the phylum Cnidaria',
            'which has a gelatinous, bell-shaped body',
            'which has long, trailing tentacles',
            'which may have a stinging mechanism for defense and hunting',
            'which can be found in oceans around the world',
            'which may have different colors and patterns',
            'which can be dangerous to humans if touched'
        ],
        [
            'which is a piece of armor',
            'which is worn to protect the torso',
            'which may be made of leather, metal, or other materials',
            'which may include pieces for the back and shoulders',
            'which may have decorative or functional details',
            'which was used by ancient civilizations such as the Greeks and Romans',
            'which is still used in some forms of modern combat',
            'which is often associated with knights and medieval warfare'
        ],
        [
            'which is a small mammal',
            'which is native to North America',
            'which has a dark brown fur with a distinctive black mask',
            'which is known for its playful and curious behavior',
            'which is often found in grasslands and prairies',
            'which is an endangered species',
            'which has been the subject of conservation efforts',
            'which is often kept in captivity for breeding programs'
        ],
        [
            'which is a type of vehicle',
            'which is used to transport schoolchildren',
            'which is typically yellow in color',
            'which may have flashing lights and a stop sign for safety',
            'which is designed with high seating capacity',
            'which is often used for field trips and extracurricular activities',
            'which is regulated by state and federal laws',
            'which is an important part of public education'
        ],
        [
            'which is a small breed of dog',
            'which is known for its curly, hypoallergenic coat',
            'which may be black, white, or other colors',
            'which has a distinctive pom-pom haircut',
            'which is often kept as a companion animal',
            'which is intelligent and trainable',
            'which has been used for various purposes such as hunting and performing',
            'which is a popular breed for dog shows and competitions'
        ],
        [
            'which is a type of multihull sailboat',
            'which has two parallel hulls connected by a deck',
            'which is designed for speed and stability',
            'which can be used for racing or leisure sailing',
            'which is often used for beachcat sailing',
            'which is popular in coastal regions and island areas',
            'which can be made from a variety of materials, including wood, fiberglass, and aluminum',
            'which can range in size from small beachcat catamarans to large ocean-going vessels'
        ],
        [
            'which is a piece of snorkeling equipment',
            'which allows the user to breathe underwater',
            'which consists of a tube and a mouthpiece',
            'which is usually made of plastic or rubber',
            'which can be adjusted for a comfortable fit',
            'which is often used in conjunction with fins and a mask',
            'which allows the user to explore underwater environments',
            'which is popular for recreational and sports activities'
        ],
        [
            'which is a musical instrument',
            'which is a member of the woodwind family',
            'which produces a rich, warm tone',
            'which has a conical bore and a double reed',
            'which is played by blowing air through the reed',
            'which is often used in orchestral and chamber music',
            'which has a wide range of notes',
            'which requires a lot of practice to master'
        ],
        [
            'which is a type of fence',
            'which is made of wooden stakes or poles',
            'which is woven together in a zigzag pattern',
            'which is often used for agricultural purposes',
            'which is designed to keep animals in or out',
            'which is easy to construct and repair',
            'which is commonly found in rural areas',
            'which has a rustic, traditional look'
        ],
        [
            'which is a wild dog species',
            'which is native to Africa',
            'which has a mottled coat of brown, black, and white fur',
            'which has large, rounded ears',
            'which is a highly efficient hunter',
            'which hunts in packs',
            'which can run at speeds up to 45 mph',
            'which is endangered due to habitat loss and hunting'
        ],
        [
            'which is a breed of dog',
            'which is popular as a family pet and service animal',
            'which has a golden, wavy coat of fur',
            'which is friendly and intelligent',
            'which is often used as a therapy dog',
            'which is a good retriever of birds and game',
            'which is easy to train and loyal',
            'which is susceptible to health issues such as hip dysplasia and cancer'
        ],
        [
            'which is an amusement ride',
            'which consists of a rotating platform with seats',
            'which is often decorated with lights and music',
            'which is powered by an electric motor',
            'which moves in a circular motion',
            'which can move up and down',
            'which is popular at fairs and carnivals',
            'which can be enjoyed by riders of all ages'
        ],
        [
            'which is a large warship',
            'which is designed to carry and launch military aircraft',
            'which is heavily armed with missiles, guns, and other weapons',
            'which can operate for extended periods of time at sea',
            'which is typically accompanied by a group of supporting vessels',
            'which can serve as a mobile airbase for military operations'
        ],
        [
            'which is a machine used to make copies of documents and images',
            'which uses electrostatic technology to transfer toner onto paper',
            'which can produce high-quality reproductions quickly and easily',
            'which can be used in offices, libraries, and other workplaces',
            'which has a variety of functions including scanning, printing, and faxing'
        ],
        [
            'which is a small fox species',
            'which is native to the Arctic regions of the Northern Hemisphere',
            'which has thick, white fur that helps it to blend in with its snowy environment',
            'which has a round face and small ears',
            'which is well adapted to living in cold climates',
            'which feeds on small mammals, birds, and fish'
        ],
        [
            'which is a small accessory used to hold hair in place',
            'which can be made of various materials such as plastic, metal, or fabric',
            'which comes in a variety of sizes, shapes, and designs',
            'which can be decorated with jewels, pearls, or other embellishments',
            'which can be worn for both practical and decorative purposes',
            'which can be used in various hairstyles such as ponytails, braids, or updos'
        ],
        [
            'which is a type of roof made of overlapping tiles',
            'which is commonly used in Mediterranean and Spanish-style architecture',
            'which can be made of various materials such as clay, concrete, or slate',
            'which provides good insulation and ventilation',
            'which can be durable and long-lasting',
            'which can be aesthetically pleasing and add value to a home'
        ],
        [
            'which is a breed of dog',
            'which is native to the Balearic Islands of Spain',
            'which is known for its slender and graceful appearance',
            'which has short, smooth, and dense fur',
            'which comes in a variety of colors including white, red, and tan',
            'which is a skilled hunter of rabbits and other small game'
        ],
        [
            'which is a colorful bird',
            'which is native to Central and South America',
            'which has a large and colorful bill',
            'which has bright feathers in shades of green, blue, orange, and yellow',
            'which is known for its loud calls and songs',
            'which can fly long distances and reach high speeds'
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
            'which is a garment',
            'which is worn as an outer layer',
            'which is made of a single piece of fabric',
            'which covers the upper body and arms',
            'which has a hole in the center for the head',
            'which is often worn in South America and Mexico',
            'which is available in various colors and patterns',
            'which is made from materials such as wool or cotton'
        ],
        [
            'which is a dessert',
            'which is made of layers of fruit, cake, and custard',
            'which is often served in a glass dish',
            'which can be topped with whipped cream or nuts',
            'which is commonly found in English cuisine',
            'which can be made in a variety of flavors',
            'which is a popular party dessert',
            'which is easy to make and customize'
        ],
        [
            'which is a timekeeping device',
            'which consists of two glass bulbs connected by a narrow neck',
            'which contains sand or a granular substance',
            'which takes an hour to pass from the upper bulb to the lower bulb',
            'which can be used for timing various activities',
            'which is often used for decorative purposes',
            'which is available in various colors and sizes',
            'which is a popular gift item'
        ],
        [
            'which is a decorative screen',
            'which is used to protect against heat and sparks',
            'which is made of metal or glass',
            'which is often placed in front of a fireplace',
            'which can have intricate designs or patterns',
            'which can be folded for easy storage',
            'which is a popular home accessory',
            'which can be customized to match any decor'
        ],
        [
            'which is a type of wolf',
            'which is native to North America',
            'which has a white fur coat',
            'which has a broad snout and small ears',
            'which is larger than a coyote but smaller than a gray wolf',
            'which is known for its intelligence and adaptability',
            'which is an important predator in many ecosystems',
            'which is often hunted for its fur'
        ],
        [
            'which is a type of sign',
            'which is used to indicate the name of a street',
            'which is often mounted on a pole or attached to a building',
            'which can be made of metal, plastic, or other materials',
            'which is essential for navigation and wayfinding',
            'which is typically found in urban areas',
            'which can include additional information such as block numbers or directions',
            'which is regulated by local governments'
        ],
        [
            'which is a type of parabolic mirror',
            'which is used to concentrate solar energy',
            'which is often used in solar power plants',
            'which can produce high temperatures',
            'which can generate electricity',
            'which has a large reflective surface',
            'which can track the sun to maximize energy collection',
            'which is typically made of glass or metal'
        ],
        [
            'which is a type of marine fish',
            'which is found in the western Atlantic Ocean',
            'which has a colorful yellow and purple body',
            'which has a distinctive black spot on its head',
            'which is popular in the aquarium trade',
            'which feeds on algae and small invertebrates',
            'which can live up to 25 years in captivity',
            'which is also known as the yellowface angelfish'
        ],
        [
            'which is a breed of large Hungarian dog',
            'which has a distinctive corded coat',
            'which was originally bred to guard livestock',
            'which is intelligent and independent-minded',
            'which requires a lot of grooming',
            'which can weigh up to 100 pounds',
            'which is affectionate with its family',
            'which can be protective of its territory'
        ],
        [
            'which is a type of store',
            'which sells books and other reading materials',
            'which often has a wide selection of genres',
            'which can offer new and used books',
            'which can also sell magazines and newspapers',
            'which can provide a quiet and cozy reading environment',
            'which can offer book clubs and author events',
            'which can be a popular destination for book lovers'
        ],
        [
            'which is a type of container',
            'which is typically made of wood or plastic',
            'which is used to transport and store goods',
            'which can come in various sizes and shapes',
            'which can have lids for secure storage',
            'which can be stacked for efficient storage',
            'which is often used in shipping and storage industries',
            'which can also be used for DIY projects and furniture'
        ],
        [
            'which is a type of curtain',
            'which is used in theaters and auditoriums',
            'which can be made of various materials such as velvet and silk',
            'which can be motorized or manually operated',
            'which can have various opening and closing patterns',
            'which can have different colors and designs',
            'which can enhance the stage ambiance and performances',
            'which can also be used in home theaters and media rooms'
        ],
        [
            'which is a type of armored vehicle',
            'which is designed for combat and military operations',
            'which can have a turret for a gun or cannon',
            'which can have tracks or wheels for mobility',
            'which can have various types of armor for protection',
            'which can carry a crew of several soldiers',
            'which can be used for reconnaissance, defense, or offense',
            'which has played a major role in modern warfare'
        ],
        [
            'which is a marine mammal',
            'which is related to manatees',
            'which is found in warm coastal waters of the Indian and Pacific Oceans',
            'which has a rounded snout and flippers',
            'which is known for its grazing habits',
            'which feeds on sea grass and algae',
            'which is a vulnerable species due to habitat loss and hunting'
        ],
        [
            'which is a breed of dog',
            'which is known for its black spots on white fur',
            'which has a short, smooth coat',
            'which is a medium-sized dog',
            'which is often used as a firehouse or carriage dog',
            'which is energetic and friendly',
            'which requires regular exercise and grooming'
        ],
        [
            'which is a sensory organ',
            'which is responsible for hearing and balance',
            'which is located on either side of the head',
            'which is made up of three parts: the outer, middle, and inner ear',
            'which is a delicate structure that can be damaged by loud noises or infections',
            'which can be affected by conditions such as tinnitus and vertigo',
            'which is an important part of the body for communication and spatial awareness'
        ],
        [
            'which is a projectile',
            'which is designed to be self-propelled and guided',
            'which can be used for military or civilian purposes',
            'which can travel at high speeds and altitudes',
            'which can carry various payloads, such as explosives or sensors',
            'which can be launched from various platforms, such as land, sea, or air',
            'which can be guided by various means, such as GPS or radar'
        ],
        [
            'which is a type of mushroom',
            'which is found in forests and woodlands',
            'which has a distinctive cap and stem',
            'which can be eaten or used for medicinal purposes',
            'which can come in various colors, such as brown, red, or yellow',
            'which can have various textures, such as smooth or rough',
            'which can have various flavors, such as nutty or earthy'
        ],
        [
            'which is a citrus fruit',
            'which is round or oval in shape',
            'which has a bright orange color',
            'which has a thick, spongy skin',
            'which has a sweet or sour taste',
            'which is high in vitamin C',
            'which can be eaten fresh or used for juice or cooking'
        ],
        [
            'which is a container',
            'which is used for holding liquids or flowers',
            'which can be made of various materials, such as glass, ceramic, or metal',
            'which can come in various shapes, such as cylindrical or conical',
            'which can have various decorations, such as paintings or carvings',
            'which can be used for various purposes, such as decoration or storage',
            'which can come in various sizes, from small to large'
        ],
        [
            'which is a breed of hound dog',
            'which is known for its excellent sense of smell',
            'which was developed in the United States',
            'which has a short and dense coat',
            'which is typically tricolored',
            'which is used for hunting small game and raccoons',
            'which has a friendly and loyal disposition',
            'which requires regular exercise and socialization'
        ],
        [
            'which is a large carnivorous mammal',
            'which is native to Africa and some parts of Asia',
            'which has a distinctive mane around its neck',
            'which is known as the "king of the jungle"',
            'which is a social animal that lives in groups',
            'which is an apex predator',
            'which can run up to 50 miles per hour',
            'which is depicted in many cultures and religions'
        ],
        [
            'which is a slow-moving mammal',
            'which is native to Central and South America',
            'which has a unique adaptation for hanging from trees',
            'which has three toes on each foot',
            'which has a greenish-brown fur',
            'which spends most of its life hanging upside down',
            'which moves slowly and deliberately',
            'which eats mostly leaves and fruits'
        ],
        [
            'which is a cosmetic product',
            'which is used to color and enhance the lips',
            'which comes in many different colors and shades',
            'which is applied with a brush or directly from the tube',
            'which can be matte, glossy, or shimmery',
            'which is made from a variety of ingredients, such as wax, oils, and pigments',
            'which has been used for centuries by humans',
            'which is often associated with femininity and beauty'
        ],
        [
            'which is a diverse ecosystem',
            'which is found in shallow, warm waters',
            'which is home to a vast array of marine life',
            'which is made up of thousands of different species of coral',
            'which is being threatened by climate change and human activity',
            'which is an important source of food and income for many people',
            'which is considered one of the most beautiful natural wonders',
            'which is protected by many conservation efforts'
        ],
        [
            'which is a mechanical device',
            'which is used to wind and store fishing line',
            'which is used in various types of fishing, such as spinning and baitcasting',
            'which can be made from different materials, such as metal or plastic',
            'which has a handle that can be turned to wind the line',
            'which has a drag system that controls the tension on the line',
            'which can come in different sizes and designs',
            'which is an essential tool for many anglers'
        ],
        [
            'which is a type of container',
            'which is used to hold beer',
            'which is made from glass or plastic',
            'which comes in different shapes and sizes',
            'which has a narrow neck and a wider base',
            'which often has a label or logo printed on it',
            'which can be recycled and reused',
            'which is often associated with socializing and relaxation'
        ],
        [
            'which is a venomous snake',
            'which is native to sub-Saharan Africa',
            'which has a bright green coloration',
            'which can grow up to 2 meters in length',
            'which has a slender body and a long tail',
            'which is known for its aggressive behavior',
            'which feeds primarily on small rodents and birds',
            'which has neurotoxic venom that can cause respiratory failure'
        ],
        [
            'which is a flat-bottomed cooking pan',
            'which is typically made of metal',
            'which has a long handle',
            'which is used for frying, searing, and browning food',
            'which can come in various sizes and materials',
            'which can be made of cast iron, stainless steel, or non-stick coating',
            'which is a common kitchen utensil',
            'which is often used to make breakfast foods like eggs and pancakes'
        ],
        [
            'which is a versatile cooking vessel',
            'which is typically made of metal',
            'which has a round bottom and high walls',
            'which is used for stir-frying, sautéing, and deep-frying',
            'which can come in various sizes and materials',
            'which can be made of carbon steel, cast iron, or non-stick coating',
            'which is a common kitchen utensil in East and Southeast Asia',
            'which is often used to make dishes like stir-fry, noodles, and rice'
        ],
        [
            'which is a large waterbird',
            'which is native to North America, Europe, and Asia',
            'which has a long neck and broad wings',
            'which has a distinctive honking call',
            'which can weigh up to 20 pounds',
            'which is known for its migratory behavior',
            'which feeds primarily on grass and aquatic plants',
            'which can be hunted for sport or for meat'
        ],
        [
            'which is a large beetle',
            'which is native to tropical regions',
            'which has a distinctive horn on its head',
            'which can grow up to 6 centimeters in length',
            'which is known for its strength',
            'which feeds primarily on rotting wood and plant material',
            'which is a common pet in some cultures',
            'which can be used in insect fighting competitions'
        ],
        [
            'which is a small sailboat',
            'which has one or more masts with fore-and-aft sails',
            'which is often used for racing or cruising',
            'which can range in size from 10 to 40 feet',
            'which can be made of various materials like wood, fiberglass, or metal',
            'which requires a crew of one or more people',
            'which can be powered by wind or motor',
            'which is a common recreational activity in coastal areas'
        ],
        [
            'which is a type of footwear',
            'which is typically made of wood',
            'which has a thick sole and an open back',
            'which is worn in some cultures for work or leisure',
            'which can come in various styles and designs',
            'which can be decorated with carvings or paint',
            'which is known for its durability and comfort',
            'which is a symbol of some national cultures'
        ],
        [
            'which is a breed of domestic dog',
            'which is also known as the Persian Greyhound',
            'which originated in the Middle East',
            'which is tall and slender with long silky fur',
            'which is known for its speed and endurance',
            'which was historically used for hunting',
            'which is a loyal and gentle companion',
            'which requires daily exercise and regular grooming'
        ],
        [
            'which is a musical instrument',
            'which produces sound by striking metal bars',
            'which is often used in orchestral music',
            'which can be made of various metals including brass, bronze, and steel',
            'which is typically played with mallets or hammers',
            'which can produce a range of tones and harmonies',
            'which can be found in various sizes and shapes',
            'which is often used in religious and ceremonial music'
        ],
        [
            'which is a raised platform',
            'which is used for performances or presentations',
            'which can be found in theaters, concert halls, and other venues',
            'which is often equipped with lighting and sound equipment',
            'which can be designed to accommodate various types of events',
            'which can be divided into different levels or sections',
            'which is often the focal point of a performance or show',
            'which can be decorated to create a certain atmosphere or mood'
        ],
        [
            'which is a breed of domestic dog',
            'which originated in Germany',
            'which is medium-sized and muscular',
            'which has a short, smooth coat',
            'which is known for its loyalty and intelligence',
            'which was historically used for hunting and guarding',
            'which requires regular exercise and training',
            'which can be a good family pet with proper socialization'
        ],
        [
            'which is a steep, rugged rock face',
            'which is often found near a body of water',
            'which can be formed by erosion or tectonic activity',
            'which can provide a habitat for various animals and plants',
            'which can be dangerous to climb without proper equipment and training',
            'which can offer scenic views of the surrounding landscape',
            'which can be the site of recreational activities such as rock climbing and hiking',
            'which can be a symbol of strength and endurance'
        ],
        [
            'which is a small, social insect',
            'which can be found all over the world',
            'which lives in organized colonies or nests',
            'which has a hierarchical social structure',
            'which can have a queen, workers, and soldiers',
            'which is known for its ability to carry objects many times its body weight',
            'which is important for pollination and soil health',
            'which can be a pest in agriculture and household settings'
        ],
        [
            'which is a large gun',
            'which is often mounted on a wheeled carriage or tripod',
            'which is designed to fire heavy projectiles over long distances',
            'which can be found in various sizes and calibers',
            'which can be used for military or civilian purposes',
            'which was historically used in warfare and siege operations',
            'which can be a symbol of power and authority',
            'which can be found in museums and historical sites'
        ],
        [
            'which is a type of arachnid',
            'which is also known as a daddy longlegs',
            'which has a small body and long, thin legs',
            'which is not a spider and does not produce venom',
            'which feeds on small insects and other arthropods',
            'which is found in many parts of the world',
            'which is often seen in gardens and forests'
        ],
        [
            'which is a type of kitchen utensil',
            'which is used for mixing ingredients',
            'which is usually made of ceramic, glass, or metal',
            'which comes in various sizes and shapes',
            'which can be used for baking, cooking, or serving',
            'which is often part of a set of kitchenware',
            'which is dishwasher safe and easy to clean'
        ],
        [
            'which is a type of roundworm',
            'which is found in soil, water, and animals',
            'which is usually microscopic in size',
            'which has a simple body structure and no respiratory system',
            'which can be parasitic or free-living',
            'which plays an important role in nutrient cycling',
            'which can be used as a model organism in genetics research'
        ],
        [
            'which is a piece of gymnastics equipment',
            'which consists of two parallel bars',
            'which are usually made of wood or metal',
            'which are supported by uprights or bases',
            'which is used for performing various exercises and routines',
            'which requires strength, balance, and coordination',
            'which is part of men\'s artistic gymnastics',
            'which is also used in physical therapy and rehabilitation'
        ],
        [
            'which is a type of heavy-duty vehicle',
            'which is used for collecting and transporting garbage',
            'which has a large, open-top container or compactor',
            'which can be operated manually or with hydraulics',
            'which is equipped with a mechanical arm or loader',
            'which is driven by a specially licensed operator',
            'which is an essential part of waste management',
            'which can be found in many urban and suburban areas'
        ],
        [
            'which is a holder for a firearm',
            'which is worn on a person\'s body',
            'which is typically made of leather or nylon',
            'which has a strap or clip for attaching to a belt or waistband',
            'which comes in various sizes and styles for different types of guns',
            'which is used by law enforcement officers and civilians alike',
            'which provides easy access to a firearm for self-defense'
        ],
        [
            'which is a cylindrical container',
            'which is typically made of wood or metal',
            'which is used for storing and transporting liquids or solids',
            'which can vary in size from small to very large',
            'which can be used for holding water, wine, oil, or gunpowder',
            'which can be used as a decorative item or for industrial purposes',
            'which can be found in various settings such as farms, wineries, and oil refineries'
        ],
        [
            'which is a type of food',
            'which is typically made of a sausage in a bun',
            'which is often served with toppings such as ketchup, mustard, and relish',
            'which can be grilled, boiled, or steamed',
            'which can be made with different types of sausages such as beef, pork, or chicken',
            'which is a popular fast food item',
            'which is commonly associated with baseball games and picnics'
        ],
        [
            'which is a cloth used for cleaning',
            'which is typically made of cotton or microfiber',
            'which can be used wet or dry',
            'which is used for wiping surfaces such as counters, dishes, and floors',
            'which can be washed and reused multiple times',
            'which is often used in conjunction with cleaning solutions',
            'which is a common household item'
        ]
    ]
    suffix = '.'
    num_classes = 100

    # def __init__(self, root_dir, transform, mode, noise_ratio=0.8, type='red', require_status = False):
    def __init__(self, root_dir, transform, mode, noise_ratio=0.8, type='red', require_status=True):
        self.root_dir = root_dir
        # load data first
        # the clean-noisy status of each sample
        # %%
        # Load the list of image links from a file
        with open(os.path.abspath(root_dir) + f'/mini-imagenet-annotations.json', "r") as f:
            image_links = json.load(f)['data']

        # %%
        # img_path = []
        img_name = []
        # img_label = []
        img_status = []
        for i in image_links:
            # img_path.append(i[0]['image/uri'])
            img_name.append(i[0]['image/id'])
            # img_label.append(i[0]['image/class/label'])
            img_status.append(i[0]['image/class/label/is_clean'])
        img_name = np.array(img_name)
        # img_status = np.array(img_status)

        if mode == 'train':
            if type == 'red':
                split_file = os.path.abspath(root_dir) + f'/split/red_noise_nl_{noise_ratio}'
                data = []
                targets = []
                status = []
                # print('load all data into memory ....')
                with codecs.open(split_file, 'r', 'utf-8') as rf:
                    lines = 0
                    for line in rf:
                        temp = line.strip().split(' ')
                        image_name = temp[0]
                        target = temp[1]
                        img_path = '{}/all_images/{}'.format(self.root_dir, image_name)

                        img = Image.open(img_path).convert('RGB')
                        data.append(img)
                        targets.append(int(target))
                        part_name = image_name.split('.')[0]
                        if require_status:
                            if np.sum(img_name == part_name) == 0:
                                status.append(1)
                            else:
                                # status.append(img_status[np.where(img_name == part_name)[0]])
                                status.append(0)
                        lines = lines + 1
                        # if lines % 1000 == 0:
                        #     print(lines)
                # print('done ....')
                self.data = data
                self.label = targets
                self.status = status
            else:
                split_file = os.path.abspath(root_dir) + f'/split/blue_noise_nl_{noise_ratio}'
                data = []
                targets = []
                # print('load all data into memory ....')
                with codecs.open(split_file, 'r', 'utf-8') as rf:
                    # lines = 0
                    for line in rf:
                        temp = line.strip().split(' ')
                        image_name = temp[0]
                        target = temp[1]
                        img_path = '{}/all_images/{}'.format(self.root_dir, image_name)

                        img = Image.open(img_path).convert('RGB')
                        data.append(img)
                        targets.append(int(target))
                # print('done ....')
                self.data = data
                self.label = targets
                split_file = os.path.abspath(root_dir) + f'/split/blue_noise_nl_0.0'
                clean_targets = []
                # print('load all data into memory ....')
                with codecs.open(split_file, 'r', 'utf-8') as rf:
                    for line in rf:
                        temp = line.strip().split(' ')
                        target = temp[1]
                        clean_targets.append(int(target))
                # print('done ....')
                status = np.zeros_like(targets)
                status[np.where(np.array(clean_targets) == np.array(targets))[0]] = 1
                # print(status.sum())
                self.status = status

        else:  # mode = 'test'
            split_file = os.path.abspath(root_dir) + f'/split/clean_validation'
            data = []
            targets = []
            # print('load all data into memory ....')
            with codecs.open(split_file, 'r', 'utf-8') as rf:
                for line in rf:
                    temp = line.strip().split(' ')
                    image_name = temp[0]
                    target = temp[1]
                    img_path = '{}/all_validation/{}'.format(self.root_dir, image_name)

                    img = Image.open(img_path).convert('RGB')
                    data.append(img)
                    targets.append(int(target))
            # print('done ....')
            self.data = data
            self.label = targets
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.data)
