from models import Model
import torch
import db
from PIL import Image
from argparse import ArgumentParser
from collections import Counter, defaultdict
from torch.utils.data import Dataset
import os
import numpy as np
import sklearn
import time
import builder
import seaborn as sn
import matplotlib.pyplot as plt
import sklearn.metrics
import pandas as pd
from openpyxl import load_workbook
import builder
import utils
import pickle

def compute_old_new(label, name, class_im, image, wrong_old, wrong_new, data, div):
    kmeans = pickle.load(open("kmeans.pkl","rb"))
    batch_data = np.array([utils.load_image(image[0])])
    lab = kmeans.predict(batch_data)[0]

    class_retr = utils.get_class(name)
    if str(lab) == label: 
        if class_retr != class_im:
            wrong_old[data.conversion[class_im]] += 1
            div[data.conversion[class_im]] += 1
    
    else:
        if class_retr == class_im:
            wrong_new[data.conversion[class_im]]+=1
            div[data.conversion[class_im]] += 1
    return wrong_new, wrong_old, div


def compute_results_kmeans(labels, image, top_1_k, top_5_k, maj_k, predictions_kmeans, ground_truth_kmeans):
    kmeans = pickle.load(open("kmeans.pkl","rb"))
    batch_data = np.array([utils.load_image(image[0])])
    label = kmeans.predict(batch_data)[0]
    ground_truth_kmeans.append(label)
    already_found_5 = 0
    conversion_kmeans =  []
    for i in range(kmeans.cluster_centers_.shape[0]):
        conversion_kmeans.append(str(i))

    
    for j in range(5):
        if j == 0:
            if labels[j] in conversion_kmeans:
                predictions_kmeans.append(int(labels[j]))
            else:
                predictions_kmeans.append(-1)
        if labels[j] == str(label):
            if already_found_5 == 0:
                top_5_k += 1
                if j == 0:
                    top_1_k += 1
            already_found_5 += 1
        
    if already_found_5 > 2:
        maj_k += 1
    
    return top_1_k, top_5_k, maj_k, predictions_kmeans, ground_truth_kmeans




def compute_results(names, data, predictions, class_im, proj_im, top_1_acc, top_5_acc, maj_acc, predictions_maj):
    similar = names[:5]
    temp = []
    already_found_5 = 0
    already_found_5_proj = 0
    already_found_5_sim = 0
    
    for j in range(len(similar)):
        class_retr = utils.get_class(similar[j])
        temp.append(class_retr)
        proj_retr = utils.get_proj(similar[j])
        
        # Retrieves label of top 1 result for confusion matrix
        if j == 0:
            # if class retrieved is in class of data given
            if class_retr in data.conversion:
                predictions.append(data.conversion[class_retr])
            else:
                predictions.append("other") # to keep a trace of the data for the cm
        
        # Class retrieved is same as query
        if class_retr == class_im: 
            if already_found_5 == 0:
                top_5_acc[0] += 1
                if already_found_5_proj == 0:
                    top_5_acc[1] += 1
                if already_found_5_sim == 0:
                    top_5_acc[2] += 1
        
                if j == 0:
                    top_1_acc[0] += 1
                    if already_found_5_proj == 0:
                        top_1_acc[1] += 1
                    if already_found_5_sim == 0:
                        top_1_acc[2] += 1
            already_found_5 += 1 # One of the 5best results matches the label of the query ->> no need to check further
            already_found_5_proj += 1
            already_found_5_sim +=1
            
        # Class retrieved is in the same project as query
        elif proj_retr == proj_im:
            if already_found_5_proj == 0:
                top_5_acc[1] += 1
                if already_found_5_sim == 0:
                    top_5_acc[2] += 1
                if j == 0:
                    if already_found_5_sim == 0:
                        top_1_acc[2] += 1
                    top_1_acc[1] += 1
            already_found_5_sim += 1
            already_found_5_proj += 1
        
        
        # Class retrieved is in a project whose content is similar to the query ->> check
        else:       
            # 'janowczyk'
            if proj_im[0:len(proj_im)-2] == proj_retr[0:len(proj_retr)-2]:
                if already_found_5_sim == 0: 
                    top_5_acc[2] += 1
                    if j == 0:
                        top_1_acc[2] += 1
                already_found_5_sim += 1
            elif proj_im == "cells_no_aug" and proj_retr == "patterns_no_aug":
                if already_found_5_sim == 0: 
                    top_5_acc[2] += 1
                    if j == 0:
                        top_1_acc[2] += 1
                already_found_5_sim += 1
            elif proj_retr == "cells_no_aug" and proj_im == "patterns_no_aug":
                if already_found_5_sim == 0: 
                    top_5_acc[2] += 1
                    if j == 0:
                        top_1_acc[2] += 1
                already_found_5_sim += 1
            elif proj_retr == "mitos2014" and proj_im == "tupac_mitosis":
                if already_found_5_sim == 0: 
                    top_5_acc[2] += 1
                    if j == 0:
                        top_1_acc[2] += 1
                already_found_5_sim += 1
            elif proj_im == "mitos2014" and proj_retr == "tupac_mitosis":
                if already_found_5_sim == 0: 
                    top_5_acc[2] += 1
                    if j == 0:
                        top_1_acc[2] += 1
                already_found_5_sim += 1
    if already_found_5 > 2:
        maj_acc[0] += 1
    if already_found_5_proj > 2:
        maj_acc[1] += 1
    if already_found_5_sim > 2:
        maj_acc[2] += 1
    predictions_maj.append(data.conversion[max(set(temp), key = temp.count)])

    return predictions, predictions_maj, top_1_acc, top_5_acc, maj_acc

def display_cm_kmeans(ground_truth_k, predictions_k):
    cm = sklearn.metrics.confusion_matrix(ground_truth_k, predictions_k, labels = range(10))
    plt.figure(figsize = (10,7))
    sn.heatmap(cm, annot=True, xticklabels=True, yticklabels=True)
    plt.show()

def display_cm(ground_truth, data, predictions, predictions_maj):
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
    for el in predictions:
        if el not in columns:
            columns.append(el)
            columns_lab.append(list(data.conversion.keys())[el])
    columns = sorted(columns)
    columns_lab=sorted(columns_lab)
        
    cm = sklearn.metrics.confusion_matrix(ground_truth, predictions, labels=range(len(os.listdir(data.root)))) # classes predites = colonnes)
    # ! only working cause the dic is sorted and sklearn is creating cm by sorting the labels
    df_cm = pd.DataFrame(cm[np.ix_(rows, columns)], index=rows_lab, columns=columns_lab)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, xticklabels=True, yticklabels=True)
    plt.show()
    
    # Confusion matrix based on maj_class accuracy:
    columns = []
    columns_lab = []
    for el in predictions_maj:
        if el not in columns:
            columns.append(el)
            columns_lab.append(list(data.conversion.keys())[el])
    columns = sorted(columns)
    columns_lab = sorted(columns_lab)
    cm = sklearn.metrics.confusion_matrix(ground_truth, predictions_maj, labels=range(len(os.listdir(data.root)))) # classes predites = colonnes)
    # ! only working cause the dic is sorted and sklearn is creating cm by sorting the labels
    df_cm = pd.DataFrame(cm[np.ix_(rows, columns)], index=rows_lab, columns=columns_lab)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, xticklabels=True, yticklabels=True)
    plt.show()

def test_each_class(model, dataset, db_name, extractor, measure, name, excel_path, label, generalise=0):
    classes = sorted(os.listdir(dataset))
    if generalise == 3:
        res = np.zeros((len(classes), 18))
    else:
        res = np.zeros((len(classes), 12))
    i = 0
    for c in classes:
        r = test(model, dataset, db_name, extractor, measure, False, False, c, False, label=label, generalise=generalise)
        res[i][:] = r
        i += 1
    if generalise == 3:
        df = pd.DataFrame(res, columns=["top_1_acc", "top_5_acc", "top_1_proj", "top_5_proj", "top_1_sim", "top_5_sim", "maj_acc_class", "maj_acc_proj", "maj_acc_sim", "t_tot", "t_model", "t_search", "top_1_k", "top_5_k","maj_k","wrong_new", "wrong_old", "div"])
    else:
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
        # User has specify the classes whose results he wants to compute
        if class_name is not None:
            for c in self.classes:
                if c == class_name:
                    self.classes = [c]
                    break
        # User has specify the project he wants to compute the results of
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

        if generalise == 1:
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

def test(model, dataset, db_name, extractor, measure, generalise, project_name, class_name, see_cms, label):
    database = db.Database(db_name, model, True, extractor=='transformer')

    data = TestDataset(dataset, measure, generalise, project_name, class_name)

    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False,
                                         num_workers=4, pin_memory=True)

    top_1_acc = np.zeros((3,1)) # in order: class - project - similarity 
    top_5_acc = np.zeros((3,1))
    maj_acc = np.zeros((3,1))

    nbr_per_class = Counter()

    ground_truth = []
    predictions = []
    predictions_maj = []
    
    if generalise == 3:
        ground_truth_k = []
        predictions_k = []
        top_1_k = 0
        top_5_k = 0
        maj_k = 0
        wrong_old = np.zeros((67,1))
        wrong_new = np.zeros((67,1))
        div = np.zeros((67,1))

    t_search = 0
    t_model = 0
    t_tot = 0

    for i, image in enumerate(loader):
        t = time.time()
        names, _, t_model_tmp, t_search_tmp = database.search(Image.open(image[0]).convert('RGB'), extractor, retrieve_class=label, generalise=generalise)
        t_tot += time.time() - t
        t_model += t_model_tmp
        t_search += t_search_tmp

        # Retrieve class of images
        class_im = utils.get_class(image[0])
        proj_im = utils.get_proj(image[0])
        nbr_per_class[class_im] += 1
        ground_truth.append(data.conversion[class_im])

        if generalise == 3:
            labs = names[1]
            names = names[0]
            top_1_k, top_5_k, maj_k, ground_truth_k, predictions_k = compute_results_kmeans( labs, image, top_1_k, top_5_k, maj_k, predictions_k, ground_truth_k)
            
            wrong_new, wrong_old, div = compute_old_new(labs[0], names[0], class_im, image, wrong_old, wrong_new, data, div)

        # Compute accuracy 
        predictions, predictions_maj, top_1_acc, top_5_acc, maj_acc = compute_results(names, data, predictions, class_im, proj_im, top_1_acc,top_5_acc,maj_acc,predictions_maj)


    print("top-1 accuracy : ", top_1_acc[0] / data.__len__())
    print("top-5 accuracy : ", top_5_acc[0] / data.__len__())
    print("top-1 accuracy proj : ", top_1_acc[1] / data.__len__())
    print("top-5 accuracy proj : ", top_5_acc[1] / data.__len__())
    print("top-1 accuracy sim : ", top_1_acc[2]/ data.__len__())
    print("top-5 accuracy sim : ", top_5_acc[2] / data.__len__())
    print("maj accuracy class : ", maj_acc[0] / data.__len__())
    print("maj accuracy proj : ", maj_acc[1] / data.__len__())
    print("maj accuracy sim : ", maj_acc[2] / data.__len__())
    print(generalise)
    if generalise == 3:
        print("Top 1 accuracy on new labels: ", top_1_k / data.__len__())
        print("Top 5 accuracy on new labels: ", top_5_k / data.__len__())
        print("Maj accuracy on new labels: ", maj_k / data.__len__())
        if class_name is None:
            for j in range(67):
                if nbr_per_class[list(data.conversion.keys())[j]] != 0:
                    wrong_new[j] = wrong_new[j] / nbr_per_class[list(data.conversion.keys())[j]]
                    wrong_old[j] = wrong_old[j] / nbr_per_class[list(data.conversion.keys())[j]]
                    div[j] = div[j] / nbr_per_class[list(data.conversion.keys())[j]]
        print("Percentage of wrong old labels, correct new labels per class: ", wrong_old)
        print("Percentage of correct old labels, wrong new labels per class: ", wrong_new)
        print("Percentage of divergence per class: ", div)
    print('t_tot:', t_tot)
    print('t_model:', t_model)
    print('t_search:', t_search)
    
    if see_cms:
        display_cm(ground_truth, data, predictions, predictions_maj)
        if generalise == 3:
            display_cm_kmeans(ground_truth_k, predictions_k)
        
    if generalise == 3:
        if class_name is None:
            return [top_1_acc[0]/ data.__len__(), top_5_acc[0]/ data.__len__(), top_1_acc[1]/ data.__len__(), top_5_acc[1]/ data.__len__(), top_1_acc[2]/ data.__len__(), top_5_acc[2]/ data.__len__(), maj_acc[0]/ data.__len__(), maj_acc[1]/ data.__len__(), maj_acc[2]/ data.__len__(), t_tot, t_model, t_search, top_1_k, top_5_k, maj_k]
        else:
            return [top_1_acc[0]/ data.__len__(), top_5_acc[0]/ data.__len__(), top_1_acc[1]/ data.__len__(), top_5_acc[1]/ data.__len__(), top_1_acc[2]/ data.__len__(), top_5_acc[2]/ data.__len__(), maj_acc[0]/ data.__len__(), maj_acc[1]/ data.__len__(), maj_acc[2]/ data.__len__(), t_tot, t_model, t_search, top_1_k, top_5_k, maj_k, wrong_new[class_name], wrong_old[class_name], div[class_name]]
        
    else:
        return [top_1_acc[0]/ data.__len__(), top_5_acc[0]/ data.__len__(), top_1_acc[1]/ data.__len__(), top_5_acc[1]/ data.__len__(), top_1_acc[2]/ data.__len__(), top_5_acc[2]/ data.__len__(), maj_acc[0]/ data.__len__(), maj_acc[1]/ data.__len__(), maj_acc[2]/ data.__len__(), t_tot, t_model, t_search]
    

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        '--num_features',
        help='number of features to extract',
        default=128,
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
        '--db_name',
        default='db'
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
        help='use only half the classes to compute the accuracy',
        default = 0,
        type= int
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

    parser.add_argument(
        '--retrieve_class',
        default = "true"
    )

    parser.add_argument(
        '--parallel',
        action = 'store_true'
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
    if args.extractor == 'vgg16' or args.extractor == 'vgg11' or args.extractor == 'resnet18' or args.extractor == "vgg16":
            model = builder.BuildAutoEncoder(args) 
            builder.load_dict(args.weights, model)
            model.model_name = args.extractor
            model.num_features = args.num_features
    else:
        model = Model(num_features=args.num_features, name=args.weights, model=args.extractor,
                  use_dr=args.dr_model, device=device, parallel=args.parallel) # eval est par defaut true
    if args.excel_path is not None:
        test_each_class(model, args.path, args.db_name, args.extractor, args.measure, args.name, args.excel_path, args.retrieve_class)
    else:
        r = test(model, args.path, args.db_name, args.extractor, args.measure, args.generalise, args.project_name, args.class_name, True, label = args.retrieve_class)
