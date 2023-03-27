from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
import numpy as np
from collections import defaultdict
from transformers import DeiTFeatureExtractor, ConvNextImageProcessor
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import kmeans
import random
# https://github.com/SathwikTejaswi/deep-ranking/blob/master/Code/data_utils.py




class DRDataset(Dataset):

    def __init__(self, root='image_folder', transform=None, pair = False):
        if transform == None:
            transform = transforms.Compose(
                [
                    transforms.RandomVerticalFlip(.5),
                    transforms.RandomHorizontalFlip(.5),
                    transforms.ColorJitter(brightness=0, contrast=0, saturation=.2, hue=.1),
                    transforms.RandomResizedCrop(224, scale=(.8,1)), # Create a patch (random method)
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ]
            )
            self.augmented = False
        else:
            transform = transforms.Compose(
                [transforms.RandomVerticalFlip(.5),
                transforms.RandomHorizontalFlip(.5),
                transforms.ColorJitter(brightness=.4, contrast=.4, saturation=.4, hue=.4),
                transforms.RandomResizedCrop(224),
                transforms.RandomApply([transforms.GaussianBlur(23)]),
                transforms.RandomRotation(random.randint(0,360)),
                transforms.RandomGrayscale(0.05),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
                ]
            )
            self.augmented = True
        self.pair = pair
        self.root = root
        self.transform = transform
        self.rev_dict = {}
        self.image_dict = {}
        self.big_dict = {}
        L = []

        self.num_classes = 0

        self.num_elements = 0

        # i = count (of class) ; j = class name
        for i, j in enumerate(os.listdir(os.path.join(root))):
            self.rev_dict[i] = j
            self.image_dict[j] = np.array(os.listdir(os.path.join(root, j)))

            # k = image name 
            for k in os.listdir(os.path.join(root, j)):
                self.big_dict[self.num_elements] = (k, i)
                self.num_elements += 1 # total number of images 

            self.num_classes += 1

    # retrieve 3 images: the one at idx potisition, a random one of same class and a random one of different class (random different class)
    def _sample(self, idx):
        im, im_class = self.big_dict[idx]
        im2 = np.random.choice(self.image_dict[self.rev_dict[im_class]])
        numbers = list(range(im_class)) + list(range(im_class+1, self.num_classes))
        class3 = np.random.choice(numbers)
        im3 = np.random.choice(self.image_dict[self.rev_dict[class3]])
        p1 = os.path.join(self.root, self.rev_dict[im_class], im)
        p2 = os.path.join(self.root, self.rev_dict[im_class], im2)
        p3 = os.path.join(self.root, self.rev_dict[class3], im3)
        if not self.pair:
            return [p1, p2, p3]
        else: 
            # Make a negative pair
            if idx % 2 != 0:
                return [p1, p3]
            else:
                return [p1, p2]

    def _augmented_sample(self, idx):
        im, im_class = self.big_dict[idx]
        numbers = list(range(self.num_classes))
        class3 = np.random.choice(numbers)
        im3 = np.random.choice(self.image_dict[self.rev_dict[class3]])
        p1 = os.path.join(self.root, self.rev_dict[im_class], im)
        p3 = os.path.join(self.root, self.rev_dict[class3], im3)
        if not self.pair:
            return [p1, p1, p3]
        else: 
            # Make a negative pair
            if idx % 2 != 0:
                return [p1, p3]
            else:
                return [p1, p1]
    def __len__(self):
        return self.num_elements

    def __getitem__(self, idx):
        if self.augmented is False:
            paths = self._sample(idx)
        else:
            paths = self._augmented_sample(idx)
        images = []
        for i in paths:
            tmp = Image.open(i).convert('RGB')
            tmp = self.transform(tmp)
            images.append(tmp)
        if not self.pair:
            return (images[0], images[1], images[2])
        else:
            return (images[0], images[1], idx%2)


class TrainingDataset(Dataset):
    def __init__(self, root, name, samples_per_class, generalise, load, transformer=False):
        self.classes = os.listdir(root)
        self.classes.sort()
        
        # Keep only half the classes
        if generalise == 1:
            self.classes = self.classes[:len(self.classes) // 2 + 1]
        # Keep haf the images by sleectioning (arbittrarly) the classes to keep
        elif generalise == 2: 
            new_classes = []
            for i in range(10):
                new_classes.append(self.classes[i])
            for i in range(26):
                new_classes.append(self.classes[21 + i])
            self.classes = new_classes
        # Create new classes from the data through kmeans
        elif generalise == 3:
            # Create list of image paths 
            list_img = kmeans.make_list_images(self.classes, root)
            # Execute kmeans
            self.kmeans, self.labels, self.classes = kmeans.execute_kmeans(load, list_img)
        

        self.conversion = {x: i for i, x in enumerate(self.classes)} # Number given the clss
        self.conv_inv = {i: x for i, x in enumerate(self.classes)} # class given the number
        self.image_dict = {}
        self.image_list = defaultdict(list)

        print("================================")
        print("Loading dataset")
        print("================================")
        
        i = 0
        if generalise == 3:
            for img in list_img:
                self.image_dict[i] = (img, self.labels[i])
                self.image_list[self.labels[i]].append(img)
                i += 1
        else:
            for c in self.classes:
                for dir, subdirs, files in os.walk(os.path.join(root, c)):
                    for file in files:
                        img = os.path.join(dir, file)
                        cls = dir[dir.rfind("/") + 1:]
                        self.image_dict[i] = (img, self.conversion[cls])
                        self.image_list[self.conversion[cls]].append(img)
                        i += 1

        if name == 'deit' or name == 'cvt' or name == 'conv':
            self.transform = transforms.Compose(
                    [
                        transforms.RandomVerticalFlip(.5),
                        transforms.RandomHorizontalFlip(.5),
                        transforms.ColorJitter(brightness=0, contrast=0, saturation=.2, hue=.1),
                        transforms.RandomResizedCrop(224, scale=(.7,1)),
                        transforms.ToTensor()
                    ]
                )
            if name == 'deit':
                self.feature_extractor = DeiTFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224',
                                                                          size=224, do_center_crop=False,
                                                                          image_mean=[0.485, 0.456, 0.406],
                                                                          image_std=[0.229, 0.224, 0.225])
                                                                          
            elif name == 'cvt':
                self.feature_extractor = ConvNextImageProcessor.from_pretrained("microsoft/cvt-21", size=224, do_center_crop=False,
                                                                          image_mean=[0.485, 0.456, 0.406],
                                                                          image_std=[0.229, 0.224, 0.225])
            elif name == 'conv':
                self.feature_extractor = ConvNextImageProcessor.from_pretrained("facebook/convnext-tiny-224", size=224, do_center_crop=False,
                                                                          image_mean=[0.485, 0.456, 0.406],
                                                                          image_std=[0.229, 0.224, 0.225])
                
        else:
            self.transform = transforms.Compose(
                    [
                        transforms.RandomVerticalFlip(.5),
                        transforms.RandomHorizontalFlip(.5),
                        transforms.ColorJitter(brightness=0, contrast=0, saturation=.2, hue=.1),
                        transforms.RandomResizedCrop(224, scale=(.7,1)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                        )
                    ]
                )

        self.transformer = transformer

        self.samples_per_class = samples_per_class
        self.current_class = np.random.choice(self.classes)
        self.classes_visited = [self.current_class, self.current_class]
        self.n_samples_drawn = 0
        self.model_name = name
        self.is_init = True

    def __len__(self):
        return len(self.image_dict)

    # https://github.com/Confusezius/Deep-Metric-Learning-Baselines/blob/60772745e28bc90077831bb4c9f07a233e602797/datasets.py#L428
    def __getitem__(self, idx):
        # no image drawn and thus no class visited yet
        if self.is_init:
            self.current_class = self.classes[idx % len(self.classes)]
            self.classes_visited = [self.current_class]
            self.is_init = False
            # Select the class to draw from till we have drawn samples_per_class images

        if self.samples_per_class == 1:
            img = Image.open(self.image_dict[idx][0]).convert('RGB')
            return self.image_dict[idx][0], self.transform(img)

        if self.n_samples_drawn == self.samples_per_class:
            counter = [cls for cls in self.classes if cls not in self.classes_visited]
            if len(counter) == 0:
                self.current_class = self.classes[idx % len(self.classes)]
                self.classes_visited = [self.current_class]
                self.n_samples_drawn = 0
            else:
                self.current_class = counter[idx % len(counter)]
                self.classes_visited.append(self.current_class)
                self.n_samples_drawn = 0

        # Find the index corresponding to the class we want to draw from and then
        # retrieve an image from all the images belonging to that class.
        class_nbr = self.conversion[self.current_class]
        class_sample_idx = idx % len(self.image_list[class_nbr])
        self.n_samples_drawn += 1


        img = Image.open(self.image_list[class_nbr][class_sample_idx]).convert('RGB')

        if self.model_name == 'deit' or self.model_name == 'cvt' or self.model_name == 'conv':
            img = self.transform(img)
            return class_nbr, self.feature_extractor(images=img, return_tensors='pt')['pixel_values']
        
        return class_nbr, self.transform(img)

class AddDataset(Dataset):
    def __init__(self, root, model_name , transformer=False):
        self.root = root
        self.list_img = []
        self.transform = transforms.Compose(
                [
                    transforms.RandomVerticalFlip(.5),
                    transforms.RandomHorizontalFlip(.5),
                    transforms.ColorJitter(brightness=0, contrast=0, saturation=.2, hue=.1),
                    transforms.RandomResizedCrop(224, scale=(.7,1)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ]
            )

        self.transformer = transformer
    
        if model_name == 'deit':
            self.feature_extractor = DeiTFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224',
                                                                      size=224, do_center_crop=False,
                                                                      image_mean=[0.485, 0.456, 0.406],
                                                                      image_std=[0.229, 0.224, 0.225])
        elif model_name == 'cvt':
                self.feature_extractor = ConvNextImageProcessor.from_pretrained("microsoft/cvt-21", size=224, do_center_crop=False,
                                                                          image_mean=[0.485, 0.456, 0.406],
                                                                          image_std=[0.229, 0.224, 0.225])
                                                                      
        elif model_name == 'conv':
                self.feature_extractor = ConvNextImageProcessor.from_pretrained("facebook/convnext-tiny-224", size=224, do_center_crop=False,
                                                                          image_mean=[0.485, 0.456, 0.406],
                                                                          image_std=[0.229, 0.224, 0.225])

        self.classes = os.listdir(root)
        # self.classes = self.classes[:len(self.classes) // 2 + 1]

        # for cls in self.classes (subdir = class directories), files = all files in root directory)
        for subdir, dirs, files in os.walk(root):
            for f in files:
                self.list_img.append(os.path.join(subdir, f))

    def __len__(self):
        return len(self.list_img)

    def __getitem__(self, idx):
        img = Image.open(self.list_img[idx]).convert('RGB')

        if not self.transformer:
            return self.transform(img), self.list_img[idx]

        return self.feature_extractor(images=img, return_tensors='pt')['pixel_values'], self.list_img[idx]

class AddDatasetList(Dataset):
    def __init__(self, id, name_list, model_name, server_name='', transformer=False):
        self.list_img = []
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

        self.transformer = transformer

        if model_name == 'deit':
            self.feature_extractor = DeiTFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224',
                                                                      size=224, do_center_crop=False,
                                                                      image_mean=[0.485, 0.456, 0.406],
                                                                      image_std=[0.229, 0.224, 0.225])
        elif model_name == 'cvt':
            self.feature_extractor = ConvNextImageProcessor.from_pretrained("microsoft/cvt-21", size=224, do_center_crop=False,
                                                                          image_mean=[0.485, 0.456, 0.406],
                                                                          image_std=[0.229, 0.224, 0.225])
        elif model_name == 'conv':
                self.feature_extractor = ConvNextImageProcessor.from_pretrained("facebook/convnext-tiny-224", size=224, do_center_crop=False,
                                                                          image_mean=[0.485, 0.456, 0.406],
                                                                          image_std=[0.229, 0.224, 0.225])

        self.server_name = server_name
        self.id = id

        for n in name_list:
            self.list_img.append(n[0])

    def __len__(self):
        return len(self.list_img)

    def __getitem__(self, idx):
        # Images names are enough alone to retrieve the image file
        if self.server_name == '':
            img = Image.open(os.path.join(self.list_img[idx], str(idx+self.id) + '.png')).convert('RGB')
            if not self.transformer:
                return self.transform(img), os.path.join(self.list_img[idx], str(idx+self.id)  + '.png')
            return self.feature_extractor(images=img, return_tensors='pt')['pixel_values'], os.path.join(
                self.list_img[idx], str(idx+self.id)  + '.png')
        # Images names need to be appended to the server name to retrieve the files 
        img = Image.open(os.path.join(self.list_img[idx], self.server_name + '_' + str(idx+self.id) + '.png')).convert('RGB')
        if not self.transformer:
            return self.transform(img), os.path.join(self.list_img[idx],
                                                     self.server_name + '_' + str(idx+self.id)  + '.png')
        return self.feature_extractor(images=img, return_tensors='pt')['pixel_values'], os.path.join(
            self.list_img[idx], self.server_name + '_' + str(idx+self.id)  + '.png')

class AddSlide(Dataset):
    def __init__(self, patches, slide):
        self.patches = patches
        self.slide = slide
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

    def __len__(self):
        if self.slide.level_count > 1:
            return self.patches.shape[0] * 2
        else:
            return self.patches.shape[0]

    def __getitem__(self, key):
        if key < self.patches.shape[0]:
            return self.transform(self.slide.read_region((self.patches[key, 1] * 224, self.patches[key, 0] * 224), 0,
                                                         (224, 224)).convert('RGB'))
        else:
            return self.transform(self.slide.read_region((self.patches[key-self.patches.shape[0], 1] * 224, self.patches[key-self.patches.shape[0], 0] * 224), 1,
                                                         (224, 224)).convert('RGB'))
