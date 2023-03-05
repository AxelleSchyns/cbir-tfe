import torch.nn as nn
import torch.nn.parallel as parallel
import os
import sys
import torch
import vgg, resnet

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

def BuildAutoEncoder(args):
    if args.extractor in ["vgg11", "vgg13", "vgg16", "vgg19"]:
        configs = vgg.get_configs(args.extractor)
        model = vgg.VGGAutoEncoder(configs)

    elif args.extractor in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        configs, bottleneck = resnet.get_configs(args.extractor)
        model = resnet.ResNetAutoEncoder(configs, bottleneck)
    
    model = nn.DataParallel(model).cuda()

    return model
