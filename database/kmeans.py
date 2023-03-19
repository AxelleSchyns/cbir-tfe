import random
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import confusion_matrix, silhouette_score
import utils
import time

def make_list_images(classes, root):
    list_img = []
    for c in classes:
        for dir, subdirs, files in os.walk(os.path.join(root, c)):
            for file in files:
                img = os.path.join(dir, file)
                list_img.append(img)
    random.shuffle(list_img)
    return list_img

def load_kmeans(list_img):
    labels = []
    kmeans = pickle.load(open("kmeans.pkl","rb"))
    dic_labs = pickle.load(open("labels_kmeans.pkl","rb"))
    for im in list_img:
        if im in dic_labs:
            labels.append(dic_labs[im])
        else:
            labels.append(kmeans.predict(np.array([utils.load_image(im)]))[0])
    return kmeans, labels

def train_kmeans(n_clusters, list_img):
    print("Start of kmeans training")
    t = time.time()
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=128, init="k-means++",n_init = 3)

    for batch_paths in utils.batch_image_paths(list_img, 128):
        batch_data = np.array([utils.load_image(path) for path in batch_paths])
        kmeans.partial_fit(batch_data)
    print("KMeans training time is: "+ str(time.time() - t))
    pickle.dump(kmeans, open("kmeans.pkl","wb"))

    return kmeans

def elbow_plot(list_img):
    print("Start of Elbow plot")
    ks = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 67]
    distortions = []
    for k in ks:
        t = time.time()
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=128, init="k-means++",n_init = 3)
        # Load the images in batches and update the clusters
        for batch_paths in utils.batch_image_paths(list_img, 128):
                batch_data = np.array([utils.load_image(path) for path in batch_paths])
                kmeans.partial_fit(batch_data)
        distortions.append(kmeans.inertia_)
        print("Time taken is: "+ str(time.time() - t))
    plt.figure(figsize=(16,8))
    plt.plot(ks, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

    exit(0)

def get_labels(list_img, kmeans):
    labels = []

    dic_labs = {}
    silhouette_scores = []
    t = time.time()

    i = 0
    temp_labs = None  # If the batch contains only one new label, has to combine it with the next one 
    temp_batch = None # Because the silhouette score needs at least 2 different labels

    for batch_paths in utils.batch_image_paths(list_img, 128):
        if i%1000 == 0:
            print("at iteration: "+str(i)+"/"+str(int(len(list_img)/128)))
        i+= 1

        batch_data = np.array([utils.load_image(path) for path in batch_paths])
        labs = kmeans.predict(batch_data)

        j = 0
        for l in labs:
            dic_labs[batch_paths[j]] = l 
            j+= 1
            labels.append(l)

        # Previous batch contained only one label ->> merge it with the new
        if temp_labs is not None:
            labs = np.concatenate((temp_labs, labs))
            batch_data = np.concatenate((temp_batch, batch_data))
            temp_labs = None

        # Check the number of different labels in batch 
        temp_l, _ = np.unique(labs, return_counts = True)
        if len(temp_l) > 1:
            batch_silhouette_score = silhouette_score(batch_data, labs)
            silhouette_scores.append(batch_silhouette_score)
        else:
            temp_labs = labs
            temp_batch = batch_data

    pickle.dump(dic_labs, open("labels_kmeans.pkl","wb"))
    print("The silhouette score is: "+str(np.mean(silhouette_scores)))
    print("Labels predictions took: "+str(time.time() - t))

    return labels

def histo(labels):
    labels_u, counts = np.unique(labels, return_counts = True)
    plt.bar(labels_u, counts, align = "center")
    for label_u, count in zip(labels_u, counts):
        plt.text(label_u, count, str(count), ha = 'center', va = 'bottom')
    plt.show()

def cm(list_img, labels):

    # Retrieval of old labels - conversion to int
    original_labels = []
    for n in list_img:
        class_im = utils.get_class(n)
        original_labels.append(class_im)  
    past_class = np.unique(original_labels)
    past_class.sort()
    dic = {x: i for i, x in enumerate(past_class)} 
    og_labels_int = []
    for el in original_labels:
        og_labels_int.append(dic[el])

    # Retrieval of  new labels
    new_labels = labels
    
    # Keep only rows corresponding to original labels
    rows = []
    rows_lab = []
    for el in og_labels_int:
        if el not in rows:
            rows.append(el)
            rows_lab.append(list(dic.keys())[el])
    rows = sorted(rows)
    rows_lab = sorted(rows_lab)

    # Keep only rows corresponding to clusters
    columns = []
    columns_lab = []
    for el in labels:
        if el not in columns:
            columns.append(el)
            columns_lab.append(str(el))
    columns = sorted(columns)
    columns_lab = sorted(columns_lab)

    # cms
    cm = confusion_matrix(og_labels_int, new_labels) # classes predites = colonnes)
    # ! only working cause the dic is sorted and sklearn is creating cm by sorting the labels
    df_cm = pd.DataFrame(cm[np.ix_(rows, columns)], index=rows_lab, columns=columns_lab)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True,xticklabels=True, yticklabels=True)
    plt.show()

def display_graph(labels, list_img):
    # Histogram of new classes
    histo(labels)

    # Confusion matrix 
    cm(list_img, labels)

def execute_kmeans(load, list_img):
    n_clusters = 10
    classes = [i for i in range(0, n_clusters)]
    if load == "testing":
        elbow_plot(list_img)
    elif load == "complete":
        kmeans, labels = load_kmeans(list_img)
    else:
        if load != "partial":
            print(len(list_img))
            kmeans = train_kmeans(n_clusters, list_img)
        else:
            kmeans = pickle.load(open("kmeans.pkl","rb"))   
        labels = get_labels(list_img, kmeans)

        display_graph(labels, list_img)
    
    return kmeans, labels, classes

    