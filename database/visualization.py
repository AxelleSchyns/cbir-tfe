

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import utils


def summary_image(path):
    classes = os.listdir(path)
    classes.sort()
    plt.figure(figsize=(7, 10))
    for class_ in classes:
        images = os.listdir(os.path.join(path, class_))
        idx = np.random.randint(0, len(images))
        img = Image.open(os.path.join(path, class_, images[idx])).convert('RGB')

        # save image in subplot
        plt.subplot(10, 7, classes.index(class_)+1)

        # For resized images: 
        img = transforms.RandomResizedCrop(224, scale=(.7,1))(img)
        # Remove scale and ticks
        plt.xticks([])
        plt.yticks([])
        # stick images to one another
        plt.subplots_adjust(wspace=0, hspace=0)


        """# For normal images:
        # Select ticks font size
        plt.tick_params(axis='both', which='major', labelsize=6)

        # stick images to one another
        plt.subplots_adjust(wspace=1, hspace=1.5)

        # Alternate labels position height
        if classes.index(class_) % 2 == 0:
            plt.xlabel(class_, fontsize=6, rotation=0, labelpad=1)
        else:
            plt.xlabel(class_, fontsize=6, rotation=0, labelpad=7)"""
        
        plt.imshow(img)
    
    plt.show()

def count_im_class(path, class_):
    images = os.listdir(os.path.join(path, class_))
    return len(images)

def count_maj(train, test, val):
    sets = [train, test, val]
    tot_cam = 0
    tot_jan = 0
    tot_im = 0
    for s in sets:
        set_name = s.split("/")[-1]
        nb_im_cam = count_im_class(s, "camelyon16_0")
        nb_im_jan = count_im_class(s, "janowczyk6_0")
        print("For set " + str(set_name) + ": Camelyon16_0 has "+ str(nb_im_cam) + " images")
        print("For set " + str(set_name) + ": janowczyk6_0 has "+ str(nb_im_jan) + " images")
        su = 0
        for c in os.listdir(s):
            su += count_im_class(s, c)
        print("For set " + str(set_name) + "the total number of images is "+str(su) + " images")
        print("The percentages are: "+str(nb_im_cam/su) + " for camelyon16_0 and "+str(nb_im_jan/su) + " for janowczyk6_0")
        print("So a percentage together of "+str(nb_im_cam/su + nb_im_jan/su) + " for a total of images together of " + str(nb_im_cam + nb_im_jan) + " images")

        tot_cam += nb_im_cam
        tot_jan += nb_im_jan
        tot_im += su
    print("For the whole database: Camelyon16_0 has "+ str(tot_cam) + " images, for a percentage of "+str(tot_cam/tot_im))
    print("For the whole database: janowczyk6_0 has "+ str(tot_jan) + " images, for a percentage of "+str(tot_jan/tot_im))
    print("For the whole database: the total number of images is "+str(tot_im) + " images")
    print("Together, the percentage is: "+str(tot_cam/tot_im +tot_jan/tot_im) + ", for a total of images together of " + str(tot_cam + tot_jan) + " images")
        
def bar_plot(train, test, val):
    sets = [train, test, val]
    i = 0
    plt.figure(figsize=(15, 9)) # width - height
    for s in sets:
        i += 1
        set_name = s.split("/")[-1]
        nb_per_class = []
        classes = os.listdir(s)
        classes.sort()
        tot = 0
        for c in classes:
            tot += count_im_class(s, c)
            if c != "camelyon16_0" and c != "janowczyk6_0":
                nb_per_class.append(count_im_class(s, c))
        classes = utils.rename_classes(s)
        classes.remove("camelyon16_0")
        classes.remove("janowczyk6_0")

        # bar plot in subplots
        plt.subplot(3, 1, i)
        #plt.bar(classes, nb_per_class)
        # Increasing bar widths
        plt.bar(classes, nb_per_class)
        # No ticks if not the last subplot
        # Add percentage of images for each class on top of the bar 
        for j in range(len(classes)):
            if classes[j] != "camelyon16_0" or classes[j] != "janowczyk6_0":
                # Computation of the value up to 4 decimals
                val = nb_per_class[j]/tot*100
                if val > 1:
                    val = round(val, 2)
                    val = "{:.2f}".format(val)
                else:
                    val = round(val, 3)
                    val = "{:.3f}".format(val)

                # Y position of text
                if i == 1:
                    if nb_per_class[j] > np.max(nb_per_class) -10000:
                        y = nb_per_class[j] - 7000
                        x = j+0.6
                    else:
                        y = nb_per_class[j] + 2000
                        x = j - 0.15
                else:
                    if nb_per_class[j] > np.max(nb_per_class)-1000:
                        y = nb_per_class[j] - 1000
                        x = j+0.6
                    else:
                        y = nb_per_class[j] + 250
                        x = j - 0.15
                plt.text(x = x, y = y, s = str(val)+"%", size = 7, rotation = 90)

        if i != 3:
            plt.xticks([])
        else:
            # Rotate labels
            plt.xticks(rotation=90)
            
            # change font size
            plt.tick_params(axis="x", which='major', labelsize=8)
        plt.yticks(rotation=180)
        plt.ylabel("Number of images in "+str(set_name))
        # add padding between subplots
        plt.subplots_adjust(hspace=0.1)
        # Remove vertical space on top of first subplot
        plt.subplots_adjust(top=0.95)
        # Add bottom space to see labels entierely
        plt.subplots_adjust(bottom=0.2)
    plt.show()

def width_height(train, test, val):
    sets = [train, test, val]
    widths = np.zeros((67, 1))
    heights = np.zeros((67, 1))
    nb = np.zeros((67, 1))
    for s in range(len(sets)):
        classes = os.listdir(sets[s])
        classes.sort()
        for c in range(len(classes)):
            images = os.listdir(os.path.join(sets[s], classes[c]))
            for im in images:
                im = Image.open(os.path.join(sets[s], classes[c], im))
                widths[c] += im.size[0]
                heights[c] += im.size[1]
                nb[c] += 1
            print(c)
    widths = widths/nb
    heights = heights/nb
    print(classes)
    print(widths)
    print(heights)
    print(np.mean(widths))
    print(np.mean(heights))


def vis_transf(path):
    # Visualize the transformation
    # Load the image
    img = Image.open(path)
    # Transform the image
    transform1 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),])

    transform2 = transforms.Compose([
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0.2, hue=0.1),
        transforms.ToTensor(),])
    
    transform3 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    plt.figure(figsize=(15, 9))
    plt.subplot(1, 5, 1)
    plt.imshow(img)
    transform_list = [transform1, transform2, transform3, transform]
    for t in transform_list:
        plt.subplot(1, 5, transform_list.index(t)+2)
        im = t(img)
        im = np.array(im.permute(1, 2, 0))
        plt.imshow(im)

    plt.show()

def resized_vis(paths):
    transformRes = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8,1)),
        transforms.ToTensor(),])
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 8), dpi=100, sharex=True, sharey=True)
    for p in paths:
        img = Image.open(p)

        #plt.subplot(2, 3, paths.index(p)+4)
        im = transformRes(img)
        im = np.array(im.permute(1, 2, 0))
        #plt.imshow(im)
        ax[1, paths.index(p)].imshow(im)
        
        ax[0, paths.index(p)].imshow(img)
    # Remove vertical space on the bottom and top for the whole plot
    plt.subplots_adjust(top=0.95)
    plt.subplots_adjust(bottom=0.05)

    plt.show()

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(4,7), dpi=100, sharex=True, sharey=True)
    
    img = Image.open(paths[1])
    ax[0].imshow(img)
    im = transformRes(img)
    im = np.array(im.permute(1, 2, 0))
    ax[1].imshow(im)
    plt.show()







if __name__ == "__main__":
    val = "/home/labarvr4090/Documents/Axelle/cytomine/Data/validation"
    train = "/home/labarvr4090/Documents/Axelle/cytomine/Data/train"
    test = "/home/labarvr4090/Documents/Axelle/cytomine/Data/test"

    #summary_image(train)

    #count_maj(train, test, val)

    #bar_plot(train, test, val)
    
    #width_height(train, test, val)

    p = '/home/labarvr4090/Documents/Axelle/cytomine/Data/train/janowczyk5_1/01_117238845_1502_393_250_250_2.png'
    p2 = '/home/labarvr4090/Documents/Axelle/cytomine/Data/train/janowczyk6_0/8863_idx5_x651_y1551_class0.png'
    p3 = '/home/labarvr4090/Documents/Axelle/cytomine/Data/train/iciar18_micro_113351562/17_118316818_512_0_512_512.png'
    paths = [p, p2, p3]
    #vis_transf(p)

    resized_vis(paths)
