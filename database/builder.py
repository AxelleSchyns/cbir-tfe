import torch.nn as nn
import torch.nn.parallel as parallel

import vgg, resnet

def BuildAutoEncoder(args):

    if args.extractor in ["vgg11", "vgg13", "vgg16", "vgg19"]:
        configs = vgg.get_configs(args.extractor)
        model = vgg.VGGAutoEncoder(configs)

    elif args.extractor in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        configs, bottleneck = resnet.get_configs(args.extractor)
        model = resnet.ResNetAutoEncoder(configs, bottleneck)
    
    model = nn.DataParallel(model).cuda()

    return model
