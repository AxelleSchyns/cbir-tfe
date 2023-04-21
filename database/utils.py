import numpy as np
from PIL import Image
import torch 
import os

def load_image(image_path):
    with Image.open(image_path) as image:
        image = image.convert('RGB')
        image = image.resize((224,224))
        image = np.array(image, dtype=np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return image.reshape(-1)

# Define a function to generate batches of image paths
def batch_image_paths(image_paths, batch_size):
    for i in range(0, len(image_paths), batch_size):
        yield image_paths[i:i+batch_size]

def encode(model, img):
    with torch.no_grad():
        code = model.module.encoder(img).cpu()
    return code

def get_class(path):
    end_c = path.rfind("/")
    begin_c = path.rfind("/", 0, end_c) + 1
    return path[begin_c:end_c]

def get_proj(path):
    end_c = path.rfind("/")
    begin_c = path.rfind("/", 0, end_c) + 1

    end_proj = path[begin_c:end_c].rfind("_")
    return path[begin_c:begin_c+end_proj]

def rename_classes(path):
    classes = os.listdir(path)
    classes.sort()
    cpt_c = 0
    old_name = ""
    new_classes = []
    for c in classes:
        c_project = get_proj(c)

        if c_project == old_name:
            cpt_c += 1
        else: 
            cpt_c = 0

        new_classes.append(c_project + "_" + str(cpt_c))
        old_name = c_project
    return new_classes