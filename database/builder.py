import os
import sys
import torch
## From  Horizon2333  github, https://github.com/Horizon2333/imagenet-autoencoderhttps://github.com/Horizon2333/imagenet-autoencoder
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

