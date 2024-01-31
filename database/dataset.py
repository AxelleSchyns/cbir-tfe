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

#  This file contains functions to load the dataset for training and indexing. 


# ----------------------------------- Training ----------------------------------- #

# https://github.com/SathwikTejaswi/deep-ranking/blob/master/Code/data_utils.py
# Class specific to the methods trained using contrastive learning or deep ranking 
class DRDataset(Dataset): # TODO: test validation protocol

    def __init__(self, root='image_folder', transform=None, pair = False, contrastive = True, appl = None, need_val = 0):
        # Settings of the transforms 
        if transform == None: # Supervised models 
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
        else: #    Self-supervised models
            # Custom transforms
            """transform = transforms.Compose(
                [transforms.RandomVerticalFlip(.5),
                transforms.RandomHorizontalFlip(.5),
                transforms.ColorJitter(brightness=.4, contrast=.4, saturation=.4, hue=.4),
                transforms.RandomResizedCrop(224, scale = (.8,1)),
                transforms.RandomApply([transforms.GaussianBlur(23)]),
                transforms.RandomRotation(random.randint(0,360)),
                transforms.RandomGrayscale(0.05),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
                ]
            )"""
            # AMDIM transforms https://github.com/Lightning-Universe/lightning-bolts/blob/5669578aba733bd9a7f0403e43dd6cfdcfd91aac/src/pl_bolts/transforms/self_supervised/amdim_transforms.py
            transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(p=0.5),
                 transforms.RandomResizedCrop(size=224),
                 transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                 transforms.RandomGrayscale(p=0.25),
                 transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],    
                                         std=[0.229, 0.224, 0.225])])
                 
            #SimCLR transforms, pytorch lightning bolt 
            """transform = transforms.Compose(
                [transforms.RandomResizedCrop(size=224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter()], 0.8),
                transforms.RandomGrayscale(0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.5),
                transforms.transforms.ToTensor()])"""
            self.augmented = True

        self.appl = appl 
        if appl == 'Unique':
            self.transform_2 = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
                ]
            )
        
        self.contrastive = contrastive
        self.pair = pair
        self.root = root
        self.transform = transform
        self.rev_dict = {}
        self.image_dict = {}
        self.big_dict = {}

        self.num_classes = 0

        self.num_elements = 0

        for index_class, class_name in enumerate(os.listdir(os.path.join(root))):
            self.rev_dict[index_class] = class_name
            self.image_dict[class_name] = np.array(os.listdir(os.path.join(root, class_name)))
            images = os.listdir(os.path.join(root, class_name))                                                                     
            cpt_c = 0
            nb_im_c = len(images)
            print(nb_im_c)
            for image_name in images:
                if need_val != 1:
                    if need_val == 0 or cpt_c < nb_im_c * 0.85:
                        self.big_dict[self.num_elements] = (image_name, index_class)
                        self.num_elements += 1 # total number of images 
                else:
                    if cpt_c >= nb_im_c * 0.85:
                        self.big_dict[self.num_elements] = (image_name, index_class)
                        self.num_elements += 1
                cpt_c += 1

            self.num_classes += 1

    # retrieve 3 images: the one at idx position, a random one of same class and a random one of different class (random different class)
    def _sample(self, idx):
        im, im_class_ind = self.big_dict[idx]
        im_class = self.rev_dict[im_class_ind]
        im2 = np.random.choice(self.image_dict[im_class])
        while im2 == im:
            im2 = np.random.choice(self.image_dict[im_class])
        numbers = list(range(im_class_ind)) + list(range(im_class_ind+1, self.num_classes))
        class3 = np.random.choice(numbers)
        im3 = np.random.choice(self.image_dict[self.rev_dict[class3]])
        p1 = os.path.join(self.root, im_class, im)
        p2 = os.path.join(self.root, im_class, im2)
        p3 = os.path.join(self.root, self.rev_dict[class3], im3)
        return [p1, p2, p3]

    # Retrieve three images, two times the one at idx position and one random 
    def _augmented_sample(self, idx):
        im, im_class_ind = self.big_dict[idx]
        numbers = list(range(self.num_classes))
        neg_class_ind = np.random.choice(numbers)
        neg_class = self.rev_dict[neg_class_ind]
        im3 = np.random.choice(self.image_dict[neg_class])
        while im3 == im:
            im3 = np.random.choice(self.image_dict[neg_class])
        p1 = os.path.join(self.root, self.rev_dict[im_class_ind], im)
        p3 = os.path.join(self.root, neg_class, im3)
        return [p1, p1, p3]
    
    def __len__(self):
        return self.num_elements

    def __getitem__(self, idx):
        if self.augmented is False:
            paths = self._sample(idx)
        else:
            paths = self._augmented_sample(idx)

        images = []
        # AMDIM pipeline
        if self.appl is None:
            for i in paths:
                tmp = Image.open(i).convert('RGB')
                tmp = self.transform(tmp)
                images.append(tmp)
        elif self.appl == 'Unique': # Unique pipeline
            images.append(self.transform_2(Image.open(paths[0]).convert('RGB')))
            images.append(self.transform(Image.open(paths[1]).convert('RGB')))
            images.append(self.transform_2(Image.open(paths[2]).convert('RGB')))

        elif self.appl == 'simclr': # SimCLR pipeline
            images.append(self.transform(Image.open(paths[0]).convert('RGB')))
            images.append(self.transform(Image.open(paths[1]).convert('RGB')))
            images.append(self.transform(Image.open(paths[2]).convert('RGB')))

        if not self.pair:
            return (images[0], images[1], images[2])
        else:
            if idx%2 != 0 and self.contrastive:
                return (images[0], images[2], idx%2)
            else:
                return (images[0], images[1], idx%2)

# Class for the rest of the methods for training 
class TrainingDataset(Dataset):
    def __init__(self, root, model_name, samples_per_class, generalise, load_kmeans, need_val=0, informative_samp = True):

        # 1. Load the dataset + generalise it if needed
        self.classes = os.listdir(root)
        self.classes.sort()
        # Keep only half the classes (arbitrarly)
        if generalise == 1:
            list_classes = ['janowczyk2_0','janowczyk2_1', 'lbpstroma_113349434', 'lbpstroma_113349448', 'mitos2014_0', 'mitos2014_1', 'mitos2014_2', 
                            'patterns_no_aug_0', 'patterns_no_aug_1', 'tupac_mitosis_0', 'tupac_mitosis_1', 'ulg_lbtd_lba_406558', 'ulg_lbtd_lba_4762', 
                            'ulg_lbtd_lba_4763', 'ulg_lbtd_lba_4764', 'ulg_lbtd_lba_4765', 'ulg_lbtd_lba_4766', 'ulg_lbtd_lba_4767', 'ulg_lbtd_lba_4768', 
                            'umcm_colorectal_01_TUMOR', 'umcm_colorectal_02_STROMA', 'umcm_colorectal_03_COMPLEX', 'umcm_colorectal_04_LYMPHO', 
                            'umcm_colorectal_05_DEBRIS', 'umcm_colorectal_06_MUCOSA', 'umcm_colorectal_07_ADIPOSE', 'umcm_colorectal_08_EMPTY', 
                            'warwick_crc_0', 'camelyon16_0', 'camelyon16_1', 'iciar18_micro_113351562', 'iciar18_micro_113351588', 
                            'iciar18_micro_113351608', 'iciar18_micro_113351628']
            self.classes = list_classes
            self.classes.sort()
                    
        # Keep haf the images by selectioning (arbitrarly) the classes to keep
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
            self.kmeans, self.labels, self.classes = kmeans.execute_kmeans(load_kmeans, list_img)
        

        # 2. Create a dictionary to convert class name to number and vice versa
        self.conversion = {x: i for i, x in enumerate(self.classes)} # Number given the class
        self.conv_inv = {i: x for i, x in enumerate(self.classes)} # class given the number
        self.image_dict = {}
        self.image_list = defaultdict(list)


        print("================================")
        print("Loading dataset")
        print("================================")
        
        # 3. Create a dictionary of images and their class
        i = 0
        if generalise == 3:
            for img in list_img:
                self.image_dict[i] = (img, self.labels[i])
                self.image_list[self.labels[i]].append(img)
                i += 1
        else:
            for c in self.classes:
                for dir, subdirs, files in os.walk(os.path.join(root, c)):
                    nb_im_c = len(files)
                    cpt_c = 0
                    files.sort()
                    for file in files:
                        if need_val != 1:
                            if need_val == 0 or cpt_c <= 0.98 * nb_im_c:
                                img = os.path.join(dir, file)
                                cls = dir[dir.rfind("/") + 1:]
                                self.image_dict[i] = (img, self.conversion[cls])
                                self.image_list[self.conversion[cls]].append(img)
                                i += 1
                        else:
                            if cpt_c > 0.98 * nb_im_c:
                                img = os.path.join(dir, file)
                                cls = dir[dir.rfind("/") + 1:]
                                self.image_dict[i] = (img, self.conversion[cls])
                                self.image_list[self.conversion[cls]].append(img)
                                i += 1
                        cpt_c += 1
        
        # 4. Create the transformation to apply to the images (depends on model)
        if model_name == 'deit' or model_name == 'cvt' or model_name == 'conv':
            self.transformer = True
            self.transform = transforms.Compose(
                    [
                        transforms.RandomVerticalFlip(.5),
                        transforms.RandomHorizontalFlip(.5),
                        transforms.ColorJitter(brightness=0, contrast=0, saturation=.2, hue=.1),
                        transforms.RandomResizedCrop(224, scale=(.7,1)),
                        transforms.ToTensor()
                    ]
                )
            if model_name == 'deit':
                self.feature_extractor = DeiTFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224',
                                                                          size=224, do_center_crop=False,
                                                                          image_mean=[0.485, 0.456, 0.406],
                                                                          image_std=[0.229, 0.224, 0.225])
                                                                          
            elif model_name == 'cvt' or model_name == 'conv':
                self.feature_extractor = ConvNextImageProcessor.from_pretrained("microsoft/cvt-21", size=224, do_center_crop=False,
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

            self.transformer = False

        # 5. Set remaining variables
        self.samples_per_class = samples_per_class
        self.current_class = np.random.choice(self.classes)
        self.classes_visited = [self.current_class, self.current_class] 
        self.n_samples_drawn = 0
        self.model_name = model_name
        self.is_init = True
        self.needs_val = need_val
        self.informative_samp = informative_samp



    def __len__(self):
        return len(self.image_dict)

    # https://github.com/Confusezius/Deep-Metric-Learning-Baselines/blob/60772745e28bc90077831bb4c9f07a233e602797/datasets.py#L428
    def __getitem__(self, idx):
        if self.informative_samp == "False" or self.needs_val == 1:
            img = Image.open(self.image_dict[idx][0]).convert('RGB')
            return self.image_dict[idx][1], self.transform(img), 
        else:
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

# ----------------------------------- Indexing ----------------------------------- #
class AddDataset(Dataset):
    def __init__(self, root, model_name, generalise=0):
        self.root = root
        self.model_name = model_name
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
    
        if model_name == 'deit':
            self.feature_extractor = DeiTFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224',
                                                                      size=224, do_center_crop=False,
                                                                      image_mean=[0.485, 0.456, 0.406],
                                                                      image_std=[0.229, 0.224, 0.225])
            self.transformer = True
        elif model_name == 'cvt' or model_name == 'conv':
                self.feature_extractor = ConvNextImageProcessor.from_pretrained("microsoft/cvt-21", size=224, do_center_crop=False,
                                                                          image_mean=[0.485, 0.456, 0.406],
                                                                          image_std=[0.229, 0.224, 0.225])
                self.transformer = True
        else:
            self.transformer = False

        self.classes = os.listdir(root)
        if generalise == 1:
            list_classes = ['janowczyk2_0','janowczyk2_1', 'lbpstroma_113349434', 'lbpstroma_113349448', 'mitos2014_0', 'mitos2014_1', 'mitos2014_2', 
                            'patterns_no_aug_0', 'patterns_no_aug_1', 'tupac_mitosis_0', 'tupac_mitosis_1', 'ulg_lbtd_lba_406558', 'ulg_lbtd_lba_4762', 
                            'ulg_lbtd_lba_4763', 'ulg_lbtd_lba_4764', 'ulg_lbtd_lba_4765', 'ulg_lbtd_lba_4766', 'ulg_lbtd_lba_4767', 'ulg_lbtd_lba_4768', 
                            'umcm_colorectal_01_TUMOR', 'umcm_colorectal_02_STROMA', 'umcm_colorectal_03_COMPLEX', 'umcm_colorectal_04_LYMPHO', 
                            'umcm_colorectal_05_DEBRIS', 'umcm_colorectal_06_MUCOSA', 'umcm_colorectal_07_ADIPOSE', 'umcm_colorectal_08_EMPTY', 
                            'warwick_crc_0', 'camelyon16_0', 'camelyon16_1', 'iciar18_micro_113351562', 'iciar18_micro_113351588', 
                            'iciar18_micro_113351608', 'iciar18_micro_113351628']
            
            for c in self.classes[:]:
                if c in list_classes:
                    self.classes.remove(c)

        elif generalise == 2:
            list_classes = ['camelyon16_0', 'camelyon16_1', 'iciar18_micro_113351562', 'iciar18_micro_113351588', 'iciar18_micro_113351608',
                            'cells_no_aug_0', 'cells_no_aug_1', 'glomeruli_no_aug_0', 'glomeruli_no_aug_1', 'lbpstroma_113349434', 'lbpstroma_113349448',
                            'mitos2014_0', 'mitos2014_1', 'mitos2014_2', 'patterns_no_aug_0', 'patterns_no_aug_1', 'tupac_mitosis_0', 'tupac_mitosis_1',
                            'ulb_anapath_lba_4711', 'ulb_anapath_lba_4712', 'ulb_anapath_lba_4713', 'ulb_anapath_lba_4714', 'ulb_anapath_lba_4715',
                            'ulb_anapath_lba_4720', 'ulb_anapath_lba_68567', 'ulb_anapath_lba_485565', 'ulb_anapath_lba_672444', 'ulg_bonemarrow_0',
                            'ulg_bonemarrow_1', 'ulg_bonemarrow_2', 'ulg_bonemarrow_3', 'ulg_bonemarrow_4', 'ulg_bonemarrow_5', 'ulg_bonemarrow_6','ulg_bonemarrow_7']
            for c in self.classes[:]:
                if c in list_classes:
                    self.classes.remove(c)
                    
        for c in self.classes:
                for dir, subdir, files in os.walk(os.path.join(root, c)):
                    for f in files:
                        self.list_img.append(os.path.join(dir, f))

    def __len__(self):
        return len(self.list_img)

    def __getitem__(self, idx):
        img = Image.open(self.list_img[idx]).convert('RGB')

        if self.model_name == 'deit' or self.model_name == 'cvt' or self.model_name == 'conv':
            return self.feature_extractor(images=img, return_tensors='pt')['pixel_values'], self.list_img[idx]
        
        else:
            return self.transform(img), self.list_img[idx]

   