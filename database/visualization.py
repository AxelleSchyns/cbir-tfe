

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

if __name__ == "__main__":
    path = "/home/labarvr4090/Documents/Axelle/cytomine/Data/test"
    summary_image(path)