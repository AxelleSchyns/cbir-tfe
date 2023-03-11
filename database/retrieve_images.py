from db import Database
from argparse import ArgumentParser, ArgumentTypeError
import models
from PIL import Image
import time
import torch
import os
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

class ImageRetriever:
    def __init__(self, db_name, model):
        self.db = Database(db_name, model, True)

    def retrieve(self, image, extractor, nrt_neigh=10):
        return self.db.search(image,extractor, nrt_neigh)[0]

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        '--path',
        help='path to the image',
    )

    parser.add_argument(
        '--extractor',
        help='feature extractor that is used',
        default='densenet'
    )

    parser.add_argument(
        '--db_name',
        help='name of the database',
        default='db'
    )

    parser.add_argument(
        '--num_features',
        help='number of features to extract',
        default=128,
        type=int
    )

    parser.add_argument(
        '--weights',
        help='file that contains the weights of the network',
        default='weights'
    )

    parser.add_argument(
        '--dr_model',
        action="store_true"
    )

    parser.add_argument(
        '--gpu_id',
        default=0,
        type=int
    )

    parser.add_argument(
        '--nrt_neigh',
        default=10,
        type=int
    )
    parser.add_argument(
        '--results_dir'
    )
    args = parser.parse_args()

    if args.gpu_id >= 0:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'

    if args.path is None:
        print(usage)
        exit(-1)

    if not os.path.isfile(args.path):
        print('Path mentionned is not a file')
        exit(-1)
    
    if args.extractor == 'vgg16' or args.extractor == 'resnet18':
        model = builder.BuildAutoEncoder(args)     
    #total_params = sum(p.numel() for p in model.parameters())
    #print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))
        
        load_dict(args.weights, model)
        model.model_name = args.extractor
        model.num_features = args.num_features
    
    else:
        model = models.Model(model=args.extractor, num_features=args.num_features, name=args.weights,
                           use_dr=args.dr_model, device=device)

    retriever = ImageRetriever(args.db_name, model)

    names = retriever.retrieve(Image.open(args.path).convert('RGB'), args.extractor, args.nrt_neigh)
    dir = args.results_dir
    names_only = []
    for n in names:
        names_only.append(n[n.rfind('/')+1:])
        img = Image.open(n)
        img.save(os.path.join(dir,n[n.rfind('/')+1:]))
        
    print("The names of the nearest images are: "+str(names_only))
    
