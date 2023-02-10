from db import Database
from argparse import ArgumentParser, ArgumentTypeError
import models
import torch
import time
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

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        '--path',
        help='path to the folder that contains the images to add',
    )

    parser.add_argument(
        '--extractor',
        help='feature extractor that is used',
        default='densenet'
    )

    parser.add_argument(
        '--weights',
        help='file that contains the weights of the network',
        default='weights'
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
        '--rewrite',
        help='if the database already exists, rewrite it',
        action='store_true'
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

    parser.add_argument('--labeled', action='store_true')

    args = parser.parse_args()

    if args.gpu_id >= 0:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'

    if args.path is None:
        print(usage)
        exit(-1)

    if not os.path.isdir(args.path):
        print("The path mentionned is not a folder")
        exit(-1)
    
    if args.extractor == 'vgg16':
	    
	    model = builder.BuildAutoEncoder(args)     
	    #total_params = sum(p.numel() for p in model.parameters())
	    #print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))
	    
	    load_dict(args.weights, model)
	    model.model_name = args.extractor
	    model.num_features = args.num_features
    if args.extractor == 'resnet50':
	    
	    model = builder.BuildAutoEncoder(args)     
	    #total_params = sum(p.numel() for p in model.parameters())
	    #print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))
	    
	    load_dict(args.weights, model)
	    model.model_name = args.extractor
	    model.num_features = args.num_features
    	
    else:
        model = models.Model(model=args.extractor, use_dr=args.dr_model, num_features=args.num_features, name=args.weights,
                           device=device)

    if model is None:
        print("Unkown feature extractor")
        exit(-1)

    database = Database(args.db_name, model, load= not args.rewrite, transformer=args.extractor=='transformer')
    t = time.time()
    database.add_dataset(args.path, args.extractor)
    print("T_indexing = "+str(time.time() - t))
