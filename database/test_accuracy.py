from models import Model
import torch
import db
from PIL import Image
from argparse import ArgumentParser, ArgumentTypeError
from collections import Counter, defaultdict
from torch.utils.data import Dataset
import os
import numpy as np
import sklearn
import time
import builder

def load_dict(resume_path, model):
    if os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path)
        model_dict = model.state_dict()
        model_dict.update(checkpoint['state_dict'])
        model.load_state_dict(model_dict)
        # delete to release more space
        del checkpoint
    else:
        sys.exit("=> No checkpoint found at '{}'".format(resume_path))
    return model

class TestDataset(Dataset):
    def __init__(self, root, measure, generalise):
        self.root = root

        self.dic_img = defaultdict(list)
        self.img_list = []

        classes = os.listdir(root)
        classes = sorted(classes)

        if measure == 'remove':
            classes.remove('camelyon16_0')
            classes.remove('janowczyk6_0')

        classes_tmp = []

        if generalise:
            classes = classes[len(classes) // 2:]

        self.conversion = {x: i for i, x in enumerate(classes)}
        if measure != 'random':
            for i in classes:
                for img in os.listdir(os.path.join(root, str(i))):
                    self.img_list.append(os.path.join(root, str(i), img))
        else:
            for i in classes:
                for img in os.listdir(os.path.join(root, str(i))):
                    self.dic_img[i].append(os.path.join(root, str(i), img))

            nbr_empty = 0
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

def test(model, dataset, db_name, extractor, measure, generalise):
    database = db.Database(db_name, model, True, extractor=='transformer')

    data = TestDataset(dataset, measure, generalise)

    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False,
                                         num_workers=4, pin_memory=True)

    top_1_acc = 0
    top_5_acc = 0
    top_1_acc_proj = 0
    top_5_acc_proj = 0
    top_1_acc_sim = 0
    top_5_acc_sim = 0

    dic_top5 = Counter()
    dic_top1 = Counter()

    nbr_per_class = Counter()

    ground_truth = []
    predictions = []

    t_search = 0
    t_model = 0
    t_tot = 0

    for i, image in enumerate(loader):
        t = time.time()
        names, _, t_model_tmp, t_search_tmp = database.search(Image.open(image[0]).convert('RGB'), extractor)
        t_tot += time.time() - t
        t_model += t_model_tmp
        t_search += t_search_tmp
        
        similar = names[:5]

        already_found_5 = False
        already_found_5_proj = False
        already_found_5_sim = False
	# Retrieve class of images
        end_test = image[0].rfind("/")
        begin_test = image[0].rfind("/", 0, end_test) + 1
        #print(image[0][begin_test:end_test])
        #print(data.conversion)
        nbr_per_class[image[0][begin_test: end_test]] += 1
        ground_truth.append(data.conversion[image[0][begin_test: end_test]])

        for j in range(len(similar)):
            end_retr = similar[j].rfind("/")
            begin_retr = similar[j].rfind("/", 0, end_retr) + 1
            # ???
            if j == 0:
            	# if class retrieved is in class of data given 
                if similar[j][begin_retr:end_retr] in data.conversion:
                    predictions.append(data.conversion[similar[j][begin_retr:end_retr]])
                else:
                    predictions.append(1000) # why?

            if similar[j][begin_retr:end_retr] == image[0][begin_test: end_test] \
                and already_found_5 is False:
                top_5_acc += 1
                if already_found_5_proj is False:
                    top_5_acc_proj += 1
                if already_found_5_sim is False:
                    top_5_acc_sim += 1
                dic_top5[similar[j][begin_retr:end_retr]] += 1
           
                if j == 0:
                    dic_top1[similar[j][begin_retr:end_retr]] += 1
                    top_1_acc += 1
                    if already_found_5_proj is False:
                        top_1_acc_proj += 1
                    if already_found_5_sim is False:
                        top_1_acc_sim += 1
                already_found_5 = True # One of the 5best results matches the label of the query ->> no need to check further
                already_found_5_proj = True
                already_found_5_sim = True
        
            elif already_found_5_proj is False:
                end_retr_proj = similar[j][begin_retr:end_retr].rfind("_")
                end_test_proj = image[0][begin_test: end_test].rfind("_")
                if similar[j][begin_retr:begin_retr+end_retr_proj] == image[0][begin_test: begin_test + end_test_proj]:
                    top_5_acc_proj += 1
                    already_found_5_proj = True 
                    if already_found_5_sim is False:
                        top_5_acc_sim += 1
                    if j == 0:
                        if already_found_5_sim is False:
                            top_1_acc_sim += 1
                        top_1_acc_proj += 1
                    already_found_5_sim = True
            elif already_found_5_sim is False: 
                end_retr_proj = similar[j][begin_retr:end_retr].rfind("_")
                end_test_proj = image[0][begin_test: end_test].rfind("_")
                name_query = image[0][begin_test: begin_test + end_test_proj]
                name_retr = similar[j][begin_retr:begin_retr+end_retr_proj]
                # 'janowczyk'
                if name_query[0:len(name_query)-2] == name_retr[0:len(name_retr)-2]:
                    top_5_acc_sim += 1
                    if j == 0:
                        top_1_acc_sim += 1
                    already_found_5_sim = True
                elif name_query == "cells_no_aug" and name_retr == "patterns_no_aug":
                    top_5_acc_sim += 1
                    if j == 0:
                        top_1_acc_sim += 1
                    already_found_5_sim = True
                elif name_retr == "cells_no_aug" and name_query == "patterns_no_aug":
                    top_5_acc_sim += 1
                    if j == 0:
                        top_1_acc_sim += 1
                    already_found_5_sim = True
                elif name_retr == "mitos2014" and name_query == "tupac_mitosis":
                    top_5_acc_sim += 1
                    if j == 0:
                        top_1_acc_sim += 1
                    already_found_5_sim = True
                elif name_query == "mitos2014" and name_retr == "tupac_mitosis":
                    top_5_acc_sim += 1
                    if j == 0:
                        top_1_acc_sim += 1
                    already_found_5_sim = True
                
                
                 
        # print("top 1 accuracy {}, round {}".format((top_1_acc / (i + 1)), i + 1))
        # print("top 5 accuracy {}, round {} ".format((top_5_acc / (i + 1)), i + 1))


    # print("top1:")
    # for key in sorted(dic_top1.keys()):
    #     print(key.replace("_", "\_") + " & " + str(round(dic_top1[key] / nbr_per_class[key], 2)) + "\\\\")
    # print("top5:")
    # for key in sorted(dic_top5.keys()):
    #     print(key.replace("_", "\_") + " & " + str(round(dic_top5[key] / nbr_per_class[key], 2)) + "\\\\")
    print("top-1 accuracy : ", top_1_acc / data.__len__())
    print("top-5 accuracy : ", top_5_acc / data.__len__())
    print("top-1 accuracy proj : ", top_1_acc_proj / data.__len__())
    print("top-5 accuracy proj : ", top_5_acc_proj / data.__len__())
    print("top-1 accuracy sim : ", top_1_acc_sim / data.__len__())
    print("top-5 accuracy sim : ", top_5_acc_sim / data.__len__())
    print(data.__len__())
    print('t_tot:', t_tot)
    print('t_model:', t_model)
    print('t_search:', t_search)

    import seaborn as sn
    import matplotlib.pyplot as plt
    import sklearn.metrics
    import pandas as pd
    cm = sklearn.metrics.confusion_matrix(ground_truth, predictions) # classes predites = colonnes)
    df_cm = pd.DataFrame(cm, index=data.conversion.keys(), columns=data.conversion.keys())
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, xticklabels=True, yticklabels=True)
    plt.show()


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
        help='random samples from validation set <random>, remove camelyon16_0 and janowczyk6_0 <remove> or all <all>',
        default = 'random'
    )

    parser.add_argument(
        '--generalise',
        help='use only half the classes to compute the accuracy'
    )

    args = parser.parse_args()

    if args.gpu_id >= 0:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'

    if not os.path.isdir(args.path):
        print('Path mentionned is not a folder')
        exit(-1)
    if args.extractor == 'vgg16' or args.extractor == 'resnet18':
            model = builder.BuildAutoEncoder(args)     
	    #total_params = sum(p.numel() for p in model.parameters())
	    #print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))
	    
            load_dict(args.weights, model)
            model.model_name = args.extractor
            model.num_features = args.num_features
    else:
        model = Model(num_features=args.num_features, name=args.weights, model=args.extractor,
                  use_dr=args.dr_model, device=device) # eval est par defaut true

    test(model, args.path, args.db_name, args.extractor, args.measure, args.generalise)
