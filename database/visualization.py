

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


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
        for c in os.listdir(s):
            if c != "camelyon16_0" and c != "janowczyk6_0":
                nb_per_class.append(count_im_class(s, c))
        classes = os.listdir(s)
        classes.sort()
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
                val = round(nb_per_class[j]/sum(nb_per_class), 4)*100
                val = "{:.2f}".format(val)

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
                plt.text(x = x, y = y, s = str(val)+"%", size = 6, rotation = 90)

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



if __name__ == "__main__":
    val = "/home/labarvr4090/Documents/Axelle/cytomine/Data/validation"
    train = "/home/labarvr4090/Documents/Axelle/cytomine/Data/train"
    test = "/home/labarvr4090/Documents/Axelle/cytomine/Data/test"

    #summary_image(train)

    count_maj(train, test, val)

    bar_plot(train, test, val)
    
          