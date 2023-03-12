import os
import sys
from argparse import ArgumentParser
from collections import defaultdict

import builder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import sklearn
import sklearn.metrics
import torch
from db import encode
from models import Model
from openpyxl import load_workbook
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import DeiTFeatureExtractor, ConvNextImageProcessor

def test_each_class(model, dataset, extractor, measure, name, excel_path):
    classes = sorted(os.listdir(dataset))
    res = np.zeros((len(classes), 12))
    i = 0
    for c in classes:
        r = test(model, dataset, extractor, measure, False, False, c, False)
        res[i][:] = r
        i += 1
    df = pd.DataFrame(res, columns=["top_1_acc", "top_5_acc", "top_1_proj", "top_5_proj", "top_1_sim", "top_5_sim", "maj_acc_class", "maj_acc_proj", "maj_acc_sim", "t_tot", "t_model", "t_search"])
    df.index = classes

    book = load_workbook(excel_path)
    writer = pd.ExcelWriter(excel_path, engine = 'openpyxl')
    writer.book = book
    if name is None:
        name = "Sheet ?" 
    df.to_excel(writer, sheet_name = name)
    writer.close()


class TestDataset(Dataset):
    def __init__(self, root, measure, generalise,name=None, class_name =None):
        self.root = root

        self.dic_img = defaultdict(list)
        self.img_list = []

        self.classes = os.listdir(root)
        self.classes = sorted(self.classes)
        
        self.conversion = {x: i for i, x in enumerate(self.classes)}
        if class_name is not None:
            for c in self.classes:
                if c == class_name:
                    self.classes = [c]
                    break
        elif name is not None:
            start = False
            new_c = []
            for c in self.classes:
                end_class = c.rfind('_')
                if name == c[0:end_class]:
                    start = True
                    new_c.append(c)
                elif start == True:
                    break
            self.classes = new_c
                    
        elif measure == 'remove':
            self.classes.remove('camelyon16_0')
            self.classes.remove('janowczyk6_0')


        if generalise:
            self.classes = self.classes[len(self.classes) // 2:]

        
        if measure != 'random':
            for i in self.classes:
                for img in os.listdir(os.path.join(root, str(i))):
                    self.img_list.append(os.path.join(root, str(i), img))
        else:
            for i in self.classes:
                for img in os.listdir(os.path.join(root, str(i))):
                    self.dic_img[i].append(os.path.join(root, str(i), img))

            to_delete = []

            while True:
                for key in self.dic_img:
                    if (not self.dic_img[key]) is False:
                        img = np.random.choice(self.dic_img[key]) # Take an image of the class at random
                        self.dic_img[key].remove(img)
                        self.img_list.append(img)
                    else: # no images in the class?
                        to_delete.append(key)

                for key in to_delete:
                    self.dic_img.pop(key, None)

                to_delete.clear()
                if len(self.img_list) > 1000 or len(self.dic_img) == 0:
                    break

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        return self.img_list[idx]

def test(model, dataset, extractor, measure, generalise, project_name, class_name, see_cms):

    data = TestDataset(dataset, measure, generalise, project_name, class_name)

    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False,
                                         num_workers=4, pin_memory=True)

    acc = 0
    acc_proj = 0
    acc_sim = 0 

    ground_truth = []
    predictions = []
    for i, image in enumerate(loader):
        image_tensor = Image.open(image[0]).convert('RGB')
        if extractor == "deit" or extractor =="cvt" or extractor == "vision": # feat_extract is None in case of non transformer model thus True here 
            if model.model_name == 'deit':
                feat_extract = DeiTFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224',
		                                                         size=224, do_center_crop=False,
		                                                         image_mean=[0.485, 0.456, 0.406],
		                                                         image_std=[0.229, 0.224, 0.225]) 
            elif model.model_name == 'cvt':
                    feat_extract = ConvNextImageProcessor.from_pretrained("microsoft/cvt-21", size=224, do_center_crop=False,
                                                                            image_mean=[0.485, 0.456, 0.406],
                                                                            image_std=[0.229, 0.224, 0.225])
            elif model.model_name == 'conv':
                    feat_extract = ConvNextImageProcessor.from_pretrained("facebook/convnext-tiny-224", size=224, do_center_crop=False,
                                                                            image_mean=[0.485, 0.456, 0.406],
                                                                            image_std=[0.229, 0.224, 0.225])
            
            image_tensor = feat_extract(images=image_tensor, return_tensors='pt')['pixel_values'] # Applies the processing for the transformer model
	
        else:
            image_tensor = transforms.Resize((224, 224))(image_tensor)
            image_tensor = transforms.ToTensor()(image_tensor)
            image_tensor= transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(image_tensor)
            
        if extractor == 'vgg16' or extractor == 'resnet50':
            out = encode(model, image_tensor.to(device=next(model.parameters()).device).view(-1, 3, 224, 224))
            out = out.reshape([out.shape[0],model.num_features])
        else:
            # Retrieves the result from the model
            out = model(image_tensor.to(device=next(model.parameters()).device).view(-1, 3, 224, 224))
        label = list(data.conversion.keys())[torch.argmax(out)]
        
        end_test = image[0].rfind("/")
        begin_test = image[0].rfind("/", 0, end_test) + 1
        
        ground_truth.append(data.conversion[image[0][begin_test: end_test]])
        predictions.append(label)
        end_retr_proj = label.rfind("_")
        end_test_proj = image[0][begin_test: end_test].rfind("_")
        
        # Class retrieved is same as query
        if label == image[0][begin_test: end_test]: 
            acc += 1
            acc_proj += 1
            acc_sim += 1
                
        # Class retrieved is in the same project as query
        elif label[0:end_retr_proj] == image[0][begin_test: begin_test + end_test_proj]:
            acc_proj += 1
            acc_sim += 1
            
            
        # Class retrieved is in a project whose content is similar to the query ->> check
        else:       
            name_query = image[0][begin_test: begin_test + end_test_proj]
            name_retr = label[0:end_retr_proj]
            # 'janowczyk'
            if name_query[0:len(name_query)-2] == name_retr[0:len(name_retr)-2]:
                acc_sim += 1
            elif name_query == "cells_no_aug" and name_retr == "patterns_no_aug":
                acc_sim += 1
            elif name_retr == "cells_no_aug" and name_query == "patterns_no_aug":
                acc_sim += 1
            elif name_retr == "mitos2014" and name_query == "tupac_mitosis":
                acc_sim += 1
            elif name_query == "mitos2014" and name_retr == "tupac_mitosis":
                acc_sim += 1

    print("Accuracy : ", acc / data.__len__())
    print("Accuracy of project: ", acc_proj / data.__len__())
    print("Accuracy with similarity: ", acc_sim / data.__len__())
    
    if see_cms:
        rows_lab = []
        rows = []
        for el in ground_truth:
            if el not in rows:
                rows.append(el)
                rows_lab.append(list(data.conversion.keys())[el])
        rows = sorted(rows)
        rows_lab = sorted(rows_lab)
        
        # Confusion matrix based on top 1 accuracy
        columns = []
        columns_lab = []
        new_pred = []
        for el in predictions:
            new_pred.append(data.conversion[el])
            if el not in columns_lab:
                columns_lab.append(el)
                columns.append(data.conversion[el])
        columns = sorted(columns)
        columns_lab=sorted(columns_lab)
        cm = sklearn.metrics.confusion_matrix(ground_truth, new_pred, labels=range(len(os.listdir(data.root)))) # classes predites = colonnes)
        # ! only working cause the dic is sorted and sklearn is creating cm by sorting the labels
        
        df_cm = pd.DataFrame(cm[np.ix_(rows, columns)], index=rows_lab, columns=columns_lab)
        
        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True, xticklabels=True, yticklabels=True)
        plt.show()
        
        
    return [acc/ data.__len__(), acc_proj/ data.__len__(), acc_sim/ data.__len__()]
    

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        '--num_features',
        help='number of features to extract',
        default=67,
        type=int
    )

    parser.add_argument(
        '--path',
        default='patch/val'
    )

    parser.add_argument(
        '--extractor',
        default='densenet'
    )

    parser.add_argument(
        '--dr_model',
        action="store_true"
    )

    parser.add_argument(
        '--weights',
        help='file that contains the weights of the network',
        default='weights'
    )

    parser.add_argument(
        '--gpu_id',
        default=0,
        type=int
    )

    parser.add_argument(
        '--measure',
        help='random samples from validation set <random>, remove camelyon16_0 and janowczyk6_0 <remove>, all in separated class <separated> or all <all>',
        default = 'random'
    )

    parser.add_argument(
        '--generalise',
        help='use only half the classes to compute the accuracy'
    )
    
    parser.add_argument(
        '--project_name',
        help='name of the project of the dataset onto which to test the accuracy',
        default=None
    )
    
    
    parser.add_argument(
        '--class_name',
         help='name of the class of the dataset onto which to test the accuracy',
         default=None
    )
    
    parser.add_argument(
        '--excel_path',
        help='path to the excel file where the results must be saved',
        default = None
    )
    
    parser.add_argument(
        '--name',
        help='name to give to the sheet of the excel file',
        default = None
    )
    args = parser.parse_args()
    
    if args.gpu_id >= 0:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'

    if not os.path.isdir(args.path):
        print('Path mentionned is not a folder')
        exit(-1)
    if args.class_name is not None and args.class_name not in os.listdir(args.path):
        print("Class name does not exist")
        exit(-1)
    if args.extractor == 'vgg16' or args.extractor == 'vgg11' or args.extractor == 'resnet18' or args.extractor == 'resnet50':
        model = builder.BuildAutoEncoder(args)  
        builder.load_dict(args.weights, model)
        model.model_name = args.extractor
        model.num_features = args.num_features
    else:
        model = Model(num_features=args.num_features, name=args.weights, model=args.extractor,
                  use_dr=args.dr_model, device=device, classification=True) # eval est par defaut true
    if args.excel_path is not None:
        test_each_class(model, args.path, args.extractor, args.measure, args.name, args.excel_path)
    else:
        r = test(model, args.path, args.extractor, args.measure, args.generalise, args.project_name, args.class_name, False)
