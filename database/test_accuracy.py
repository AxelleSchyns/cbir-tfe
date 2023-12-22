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
import seaborn as sn
import matplotlib.pyplot as plt
import sklearn.metrics
import pandas as pd
import utils
import pickle
from transformers import DeiTFeatureExtractor, ConvNextImageProcessor
from torchvision import transforms

# This function looks into the errors made with the old labels and the errors made with the new labels obtained through kmeans
# - label = K-mean label of the retrieved image
# - ret_image = name of the retrieved image
# - class_im = class of the query
# - image = query image
# - wrong_old = list containing the number of wrong old labels per class at current
# - wrong_new = list containing the number of wrong new labels per class at current
# - data = dataset object containing the data
# - div = list containing the number of divergences bteween old and new per class at current
def compute_old_new(label, ret_image, class_im, image, wrong_old, wrong_new, data, div, lab):

    # Get the original label of the retrieved image
    class_retr = utils.get_class(ret_image)

    # K-means label of the retrieved image match the Kmean label of the query
    if str(lab) == label: 
        # Old label of the retrieved image is different from the old label of the query (-> divergence) 
        if class_retr != class_im:
            wrong_old[data.conversion[class_im]] += 1
            div[data.conversion[class_im]] += 1
    
    else: # K-means label of the retrieved image is different from the Kmean label of the query
        # Old label of the retrieved image is the same as the old label of the query (-> divergence)
        if class_retr == class_im:
            wrong_new[data.conversion[class_im]]+=1
            div[data.conversion[class_im]] += 1
    return wrong_new, wrong_old, div

# This function computes the results given the search results and the query for the kmeans protocol
# - labels = K-Means labels of the most similar images in the database
# - image = query image 
# - top_1_k = top-1 accuracy at current
# - top_5_k = top-5 accuracy at current
# - maj_k = maj accuracy at current
# - predictions_kmeans = complete list of the label of the top-1 result for each query
# - ground_truth_kmeans = complete list of Kmeans labels for  each query
def compute_results_kmeans(labels, image, top_1_k, top_5_k, maj_k, predictions_kmeans, ground_truth_kmeans, label, kmeans):
    ground_truth_kmeans.append(label)

    already_found_5 = 0

    # dictionary to convert the labels of kmeans from integers to strings
    conversion_kmeans =  []
    for i in range(kmeans.cluster_centers_.shape[0]):
        conversion_kmeans.append(str(i))

    for j in range(5):
        # Top-1 label 
        if j == 0:
            if labels[j] in conversion_kmeans:
                predictions_kmeans.append(int(labels[j]))
            else:
                predictions_kmeans.append(-1)
        # label of retrieved image is the same as query
        if labels[j] == str(label):
            if already_found_5 == 0:
                top_5_k += 1
                if j == 0:
                    top_1_k += 1
            already_found_5 += 1
        
    if already_found_5 > 2:
        maj_k += 1
    
    return top_1_k, top_5_k, maj_k, predictions_kmeans, ground_truth_kmeans



# This function computes the results given the search results and the query
# - names = list of the most similar images in the database
# - data = dataset object containing the data
# - predictions = complete list of the label of the top-1 result for each query
# - class_im = class of the query
# - proj_im = project of the query
# - top_1_acc = list containing the top-1 accuracy at current
# - top_5_acc = list containing the top-5 accuracy at current
# - maj_acc = list containing the maj accuracy at current
# - predictions_maj = complete list of the label of the maj result for each query
# - weights = list containing the weights of each class for the weighted protocol
def compute_results(names, data, predictions, class_im, proj_im, top_1_acc, top_5_acc, maj_acc, predictions_maj, weights):
    similar = names[:5]
    temp = []
    already_found_5 = 0
    already_found_5_proj = 0
    already_found_5_sim = 0
    if len(data.classes) == 1:
        idx_class = 0
    else:
        idx_class = data.conversion[class_im]
    for j in range(len(similar)):
        # Gets the class and project of the retrieved image
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
                top_5_acc[0] += weights[idx_class]
                if already_found_5_proj == 0:
                    top_5_acc[1] += weights[idx_class]
                if already_found_5_sim == 0:
                    top_5_acc[2] += weights[idx_class]
        
                if j == 0:
                    top_1_acc[0] += weights[idx_class]
                    if already_found_5_proj == 0:
                        top_1_acc[1] += weights[idx_class]  
                    if already_found_5_sim == 0:
                        top_1_acc[2] += weights[idx_class]
            already_found_5 += 1 # One of the 5best results matches the label of the query ->> no need to check further
            already_found_5_proj += 1
            already_found_5_sim +=1
            
        # Class retrieved is in the same project as query
        elif proj_retr == proj_im:
            if already_found_5_proj == 0:
                top_5_acc[1] += weights[idx_class]
                if already_found_5_sim == 0:
                    top_5_acc[2] += weights[idx_class]
                if j == 0:
                    if already_found_5_sim == 0:
                        top_1_acc[2] += weights[idx_class]
                    top_1_acc[1] += weights[idx_class]
            already_found_5_sim += 1
            already_found_5_proj += 1
        
        
        # Class retrieved is in a project whose content is similar to the query ->> check
        else:       
            # 'janowczyk'
            if proj_im[0:len(proj_im)-2] == proj_retr[0:len(proj_retr)-2]:
                if already_found_5_sim == 0: 
                    top_5_acc[2] += weights[idx_class]
                    if j == 0:
                        top_1_acc[2] += weights[idx_class]
                already_found_5_sim += 1
            elif proj_im == "cells_no_aug" and proj_retr == "patterns_no_aug":
                if already_found_5_sim == 0: 
                    top_5_acc[2] += weights[idx_class]
                    if j == 0:
                        top_1_acc[2] += weights[idx_class]
                already_found_5_sim += 1
            elif proj_retr == "cells_no_aug" and proj_im == "patterns_no_aug":
                if already_found_5_sim == 0: 
                    top_5_acc[2] += weights[idx_class]
                    if j == 0:
                        top_1_acc[2] += weights[idx_class]
                already_found_5_sim += 1
            elif proj_retr == "mitos2014" and proj_im == "tupac_mitosis":
                if already_found_5_sim == 0: 
                    top_5_acc[2] += weights[idx_class]
                    if j == 0:
                        top_1_acc[2] += weights[idx_class]
                already_found_5_sim += 1
            elif proj_im == "mitos2014" and proj_retr == "tupac_mitosis":
                if already_found_5_sim == 0: 
                    top_5_acc[2] += weights[idx_class]
                    if j == 0:
                        top_1_acc[2] += weights[idx_class]
                already_found_5_sim += 1
    if already_found_5 > 2:
        maj_acc[0] += weights[idx_class]
    if already_found_5_proj > 2:
        maj_acc[1] += weights[idx_class]
    if already_found_5_sim > 2:
        maj_acc[2] += weights[idx_class]
    # Get label of the majority class for confusion matrix
    predictions_maj.append(data.conversion[max(set(temp), key = temp.count)])

    return predictions, predictions_maj, top_1_acc, top_5_acc, maj_acc

# Creates and displays the confusion matrix relative to kmeans accuracy (simpler as no labels names)
def display_cm_kmeans(ground_truth_k, predictions_k):
    cm = sklearn.metrics.confusion_matrix(ground_truth_k, predictions_k, labels = range(67))
    plt.figure(figsize = (10,7))
    sn.heatmap(cm, annot=True, xticklabels=True, yticklabels=True)
    plt.show()

# Creates and displays the confusion matrix relative to top-1 and maj accuracy 
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
        
    cm = sklearn.metrics.confusion_matrix(ground_truth, predictions, labels=range(len(os.listdir(data.root)))) # classes predites = colonnes
    # ! only working cause the dic is sorted and sklearn is creating cm by sorting the labels
    df_cm = pd.DataFrame(cm[np.ix_(rows, columns)], index=rows_lab, columns=columns_lab)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, xticklabels=True, yticklabels=True)
    #plt.show()
    # save the confusion matrix
    plt.savefig('/home/labarvr4090/Documents/Axelle/cytomine/cbir-tfe/cms/confusion_matrix_top1.png')
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
    #plt.show()
    # save the confusion matrix
    plt.savefig('/home/labarvr4090/Documents/Axelle/cytomine/cbir-tfe/cms/confusion_matrix_maj.png')
# Compute the metrics per class of the dataset 
# - model = python object containing the feature extractor
# - dataset = path to the dataset from which to get the queries
# - db_name = name of the database in which the data is indexed
# - extractor = name of the feature extractor used
# - measure = protocol for the results (random, remove, all, weighted)
# - name = name to give to the excel sheet in which to write the results
# - excel_path = path to the excel file in which to write the results
# - label = if the database to search into is the labelled, unlabelled or mixed one ("True", "False", "mixed")
# - generalise = 0 if no generalisation, 1 -2 for half the classes, 3 if generalisation on kmeans

def test_each_class(model, dataset, db_name, extractor, measure, name, excel_path, label, generalise=0):
    classes = sorted(os.listdir(dataset))
    if generalise == 3:
        res = np.zeros((len(classes), 19))
    else:
        res = np.zeros((len(classes), 13))

    # Compute the results for each class
    i = 0
    for c in classes:
        print("Class: ", c)
        r = test(model, dataset, db_name, extractor, measure, project_name = False, class_name= c, see_cms= False, label=label, generalise=generalise, stat = True)
        res[i][:] = r
        i += 1
    if generalise == 3:
        df = pd.DataFrame(res, columns=["top_1_acc", "top_5_acc", "top_1_proj", "top_5_proj", "top_1_sim", "top_5_sim", "maj_acc_class", "maj_acc_proj", "maj_acc_sim", "t_tot", "t_model", "t_search", "t_transfer", "top_1_k", "top_5_k","maj_k","wrong_new", "wrong_old", "div"])
    else:
        df = pd.DataFrame(res, columns=["top_1_acc", "top_5_acc", "top_1_proj", "top_5_proj", "top_1_sim", "top_5_sim", "maj_acc_class", "maj_acc_proj", "maj_acc_sim", "t_tot", "t_model", "t_search", "t_transfer"])
    df.index = classes


    writer = pd.ExcelWriter(excel_path, engine="openpyxl", mode="a")
    """book = load_workbook(excel_path)
    writer = pd.ExcelWriter(excel_path, engine = 'openpyxl')
    writer.book = book"""
    if name is None:
        name = "Sheet ?" 
    df.to_excel(writer, sheet_name = name)
    writer.close()

# Dataset class for the test
class TestDataset(Dataset):
    # - model = python object containing the feature extractor
    # - root = path to the dataset from which to get the queries
    # - measure = protocol for the results (random, remove, separated, all, weighted)
    # - generalise = 0 if no generalisation, 1 -2 for half the classes, 3 if generalisation on kmeans
    # - name = name of the project to compute the results of if only one project is wanted
    # - class_name = name of the class to compute the results of if only one class is wanted
    def __init__(self, model, root, measure, generalise,name=None, class_name =None):
        
        self.root = root

        # Define the transforms 
        if model.model_name == 'deit':
                self.feat_extract = DeiTFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224',
		                                                         size=224, do_center_crop=False,
		                                                         image_mean=[0.485, 0.456, 0.406],
		                                                         image_std=[0.229, 0.224, 0.225]) 
                self.transformer = True
        elif model.model_name == 'cvt':
                self.feat_extract = ConvNextImageProcessor.from_pretrained("microsoft/cvt-21", size=224, do_center_crop=False,
		                                                                  image_mean=[0.485, 0.456, 0.406],
		                                                                  image_std=[0.229, 0.224, 0.225])
                self.transformer = True
        elif model.model_name == 'conv':
                self.feat_extract = ConvNextImageProcessor.from_pretrained("facebook/convnext-tiny-224", size=224, do_center_crop=False,
		                                                                  image_mean=[0.485, 0.456, 0.406],
		                                                                  image_std=[0.229, 0.224, 0.225])
                self.transformer = True
        else:
                self.feat_extract = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )])
                self.transformer = False

        self.dic_img = defaultdict(list)
        self.img_list = []

        self.classes = os.listdir(root)
        self.classes = sorted(self.classes)
        
        self.conversion = {x: i for i, x in enumerate(self.classes)}
        # User has specify the classe whose results he wants to compute
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

        # Keep only half the classes 
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
            #self.classes = self.classes[len(self.classes) // 2:]
            self.conversion = {x: i for i, x in enumerate(self.classes)}

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
          
            self.conversion = {x: i for i, x in enumerate(self.classes)}
        # Register images in list and compute weights for weighted protocol
        if measure != 'random':
            if measure == "weighted":
                weights = np.zeros(len(self.classes))
                for i in self.classes:
                    weights[self.conversion[i]] = 1 / len(os.listdir(os.path.join(root, str(i))))
                    for img in os.listdir(os.path.join(root, str(i))):
                        self.img_list.append(os.path.join(root, str(i), img))
                self.weights = weights
            else:
                for i in self.classes:
                    for img in os.listdir(os.path.join(root, str(i))):
                        self.img_list.append(os.path.join(root, str(i), img))
        # Selection of the 1020 images for the random protocol
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
        
        img = Image.open(self.img_list[idx]).convert('RGB')

        if not self.transformer:
            return self.feat_extract(img), self.img_list[idx]

        return self.feat_extract(images=img, return_tensors='pt')['pixel_values'], self.img_list[idx]


# Main function of the file, calls the different computation functions and display functions given the value of the parameters
# - model = python object containing the feature extractor 
# - dataset = path to the dataset from which to get the queries
# - db_name = name of the database in which the data is indexed
# - extractor = name of the feature extractor used
# - measure = protocol for the results (random, remove, separated, all, weighted)
# - generalise = 0 if no generalisation, 1 -2 for half the classes, 3 if generalisation on kmeans
# - project_name = name of the project to compute the results of if only one project is wanted
# - class_name = name of the class to compute the results of if only one class is wanted
# - see_cms = True if the graphics of the results must be displayed
# - label = if the database to search into is the labelled, unlabelled or mixed one ("True", "False", "mixed")
# - stat = True if the protocol is stat, False otherwise. Controls the display of the results in terminal 
def test(model, dataset, db_name, extractor, measure, generalise, project_name, class_name, see_cms, label, stat = False):

    # Load database
    database = db.Database(db_name, model, True)

    # Load data 
    data = TestDataset(model, dataset, measure, generalise, project_name, class_name)
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False,
                                         num_workers=4, pin_memory=True)
    
    # Load weights 
    if measure == 'weighted':
        weights = data.weights
    elif measure == 'remove':
        weights = np.ones(len(data.conversion))
    else:
        weights = np.ones(len(data.classes))
    

    top_1_acc = np.zeros((3,1)) # in order: class - project - similarity 
    top_5_acc = np.zeros((3,1))
    maj_acc = np.zeros((3,1))

    nbr_per_class = Counter()

    ground_truth = []
    predictions = []
    predictions_maj = []
    
    # Define new metrics for K-means 
    if generalise == 3:
        ground_truth_k = []
        predictions_k = []
        top_1_k = 0
        top_5_k = 0
        maj_k = 0
        wrong_old = np.zeros((len(data.classes),1))
        wrong_new = np.zeros((len(data.classes),1))
        div = np.zeros((len(data.classes),1))

    t_search = 0
    t_model = 0
    t_transfer = 0
    t_tot = 0

    # For each image in the dataset, search for the 5 most similar images in the database and compute the accuracy
    for i, (image, filename) in enumerate(loader):
            
        if (i+1) % 1000000 == 0:
            print(i)
        # Search for the 5 most similar images in the database
        t = time.time()
        names, _, t_model_tmp, t_search_tmp, t_transfer_tmp = database.search(image, extractor, generalise=generalise)
        image = filename
        t_tot += time.time() - t
        t_model += t_model_tmp
        t_transfer += t_transfer_tmp
        t_search += t_search_tmp

        # Retrieve class and project of query 
        class_im = utils.get_class(image[0])
        proj_im = utils.get_proj(image[0])
        nbr_per_class[class_im] += 1
        ground_truth.append(data.conversion[class_im])

        # Get results for K-means 
        if generalise == 3:
            # Retrieve the K-mean label of the query image
            kmeans = pickle.load(open("weights_folder/kmeans_104.pkl","rb"))
            batch_data = np.array([utils.load_image(image[0])])
            lab = kmeans.predict(batch_data)[0]
            labs = names[1]
            names = names[0]
            top_1_k, top_5_k, maj_k, ground_truth_k, predictions_k = compute_results_kmeans( labs, image, top_1_k, top_5_k, maj_k, predictions_k, ground_truth_k, lab, kmeans)
            
            wrong_new, wrong_old, div = compute_old_new(labs[0], names[0], class_im, image, wrong_old, wrong_new, data, div, lab)

        # Compute accuracy 
        predictions, predictions_maj, top_1_acc, top_5_acc, maj_acc = compute_results(names, data, predictions, class_im, proj_im, top_1_acc,top_5_acc,maj_acc,predictions_maj, weights)


    if measure == 'weighted':
        s = len(data.classes) # In weighted, each result was already divided by the length of the class
    else:
        s = data.__len__()
    
    # Display results in terminal
    if not stat:
        print("top-1 accuracy : ", top_1_acc[0] / s)
        print("top-5 accuracy : ", top_5_acc[0] / s)
        print("top-1 accuracy proj : ", top_1_acc[1] / s)
        print("top-5 accuracy proj : ", top_5_acc[1] / s)
        print("top-1 accuracy sim : ", top_1_acc[2]/ s)
        print("top-5 accuracy sim : ", top_5_acc[2] / s)
        print("maj accuracy class : ", maj_acc[0] / s)
        print("maj accuracy proj : ", maj_acc[1] / s)
        print("maj accuracy sim : ", maj_acc[2] / s)

        if generalise == 3:
            s = data.__len__()
            print("Top 1 accuracy on new labels: ", top_1_k / s)
            print("Top 5 accuracy on new labels: ", top_5_k / s)
            print("Maj accuracy on new labels: ", maj_k / s)
            if class_name is None:
                for j in range(len(data.classes)):
                    if nbr_per_class[list(data.conversion.keys())[j]] != 0:
                        wrong_new[j] = wrong_new[j] / nbr_per_class[list(data.conversion.keys())[j]]
                        wrong_old[j] = wrong_old[j] / nbr_per_class[list(data.conversion.keys())[j]]
                        div[j] = div[j] / nbr_per_class[list(data.conversion.keys())[j]]
            print("Percentage of wrong old labels, correct new labels per class: ", wrong_old)
            print("Percentage of correct old labels, wrong new labels per class: ", wrong_new)
            print("Percentage of divergence per class: ", div)
        print('t_tot:', t_tot)
        print('t_model:', t_model)
        print('t_transfer:', t_transfer)
        print('t_search:', t_search)

    # Display results in graphics
    if see_cms:
        display_cm(ground_truth, data, predictions, predictions_maj)
        if generalise == 3:
            display_cm_kmeans(ground_truth_k, predictions_k)
    
    # Return results
    if generalise == 3:
        if class_name is None:
            return [top_1_acc[0]/ s, top_5_acc[0]/ s, top_1_acc[1]/ s, top_5_acc[1]/ s, top_1_acc[2]/ s, top_5_acc[2]/ s, maj_acc[0]/ s, maj_acc[1]/ s, maj_acc[2]/ s, t_tot, t_model, t_search, t_transfer, top_1_k, top_5_k, maj_k]
        else:
            return [top_1_acc[0]/ s, top_5_acc[0]/ s, top_1_acc[1]/ s, top_5_acc[1]/ s, top_1_acc[2]/ s, top_5_acc[2]/ s, maj_acc[0]/ s, maj_acc[1]/ s, maj_acc[2]/ s, t_tot, t_model, t_search, t_transfer, top_1_k, top_5_k, maj_k, wrong_new[class_name], wrong_old[class_name], div[class_name]]
        
    else:
        return [top_1_acc[0]/ s, top_5_acc[0]/ s, top_1_acc[1]/ s, top_5_acc[1]/ s, top_1_acc[2]/ s, top_5_acc[2]/ s, maj_acc[0]/ s, maj_acc[1]/ s, maj_acc[2]/ s, t_tot, t_model, t_search, t_transfer]
    
# This function executes the random protocol 50 times for stability of the results
def stat(model, dataset, db_name, extractor, generalise, project_name, class_name, label):
    # Do 50 times the experiment
    top_1_acc = np.zeros((3,10))
    top_5_acc = np.zeros((3,10))
    maj_acc = np.zeros((3,10))

    ts = np.zeros((4,10))
    for i in range(10):
        top_1_acc[0][i], top_5_acc[0][i], top_1_acc[1][i], top_5_acc[1][i], top_1_acc[2][i], top_5_acc[2][i], maj_acc[0][i], maj_acc[1][i], maj_acc[2][i], ts[0][i], ts[1][i], ts[2][i], ts[3][i] =  test(model, dataset, db_name, extractor, "random", generalise, project_name, class_name, False, label = label, stat = True)


    print("Top 1 accuracy: ", np.mean(top_1_acc[0]), " +- ", np.std(top_1_acc[0]))
    print("Top 5 accuracy: ", np.mean(top_5_acc[0]), " +- ", np.std(top_5_acc[0]))
    print("Maj accuracy: ", np.mean(maj_acc[0]), " +- ", np.std(maj_acc[0]))
    print("Top 1 accuracy on project: ", np.mean(top_1_acc[1]), " +- ", np.std(top_1_acc[1]))
    print("Top 5 accuracy on project: ", np.mean(top_5_acc[1]), " +- ", np.std(top_5_acc[1]))
    print("Maj accuracy on project: ", np.mean(maj_acc[1]), " +- ", np.std(maj_acc[1]))
    print("Top 1 accuracy on sim: ", np.mean(top_1_acc[2]), " +- ", np.std(top_1_acc[2]))
    print("Top 5 accuracy on sim: ", np.mean(top_5_acc[2]), " +- ", np.std(top_5_acc[2]))
    print("Maj accuracy on sim: ", np.mean(maj_acc[2]), " +- ", np.std(maj_acc[2]))
    print('t_tot:', np.mean(ts[0]/1020), "+-", np.std(ts[0]/1020))
    print('t_model:', np.mean(ts[1]/1020), "+-", np.std(ts[1]/1020))
    print('t_transfer:', np.mean(ts[3]/1020), "+-", np.std(ts[3]/1020))
    print('t model complet:', np.mean((ts[1]+ts[3])/1020), "+-", np.std((ts[1]+ts[3])/1020))
    print('t_search:', np.mean(ts[2]/1020), "+-", np.std(ts[2]/1020))

    return 0

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
        help='random samples from validation set <random>, remove camelyon16_0 and janowczyk6_0 <remove>, all in separated class <separated>, all <all>, weighted <weighted> or stat <stat>',
        default = 'random'
    )

    parser.add_argument(
        '--generalise',
        help='0 to use entire training set, 1 to use first separation in half, 2 to use second separation in half, 3 to use kmeans',
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
    
    model = Model(num_features=args.num_features, weight=args.weights, model=args.extractor,
                  use_dr=args.dr_model, device=device, parallel=args.parallel) # eval est par defaut true
        
    # Compute results per class and save them in given excel file 
    if args.measure == "separated":
        if args.excel_path is None:
            print("Please give the path to the excel file where to save the results")
            exit(-1)
        test_each_class(model, args.path, args.db_name, args.extractor, args.measure, args.name, args.excel_path, args.retrieve_class)
    else:
        # Random protocol realised 50 times to make stat
        if args.measure == "stat":
            stat(model, args.path, args.db_name, args.extractor, args.generalise, args.project_name, args.class_name, args.retrieve_class)
        else: # Other protocols: weighted - default - remove - random 
            r = test(model, args.path, args.db_name, args.extractor, args.measure, args.generalise, args.project_name, args.class_name, True, label = args.retrieve_class)
