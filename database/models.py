import torchvision.models as models
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import DeiTForImageClassification
import dataset
import numpy as np
import time
from loss import MarginLoss, ProxyNCA_prob, NormSoftmax, SimpleBCELoss, ContrastiveLoss, SoftTriple, InfoNCE
from efficientnet_pytorch import EfficientNet as EffNet # TODO: test pytorch efficient net to have only one library
from transformers import CvtForImageClassification, ConvNextForImageClassification, CvtConfig, CvtModel
from torchvision import transforms
from argparse import ArgumentParser
import os
import matplotlib.pyplot as plt
import autoencoders as ae
from byol_pytorch import BYOL as BYOL_pytorch
from arch import BYOL, fully_connected
from utils import create_weights_folder

# TODO: test unweighted archs
#archs = { "resnet": models.resnet50(), "densenet": models.densenet121(), "effnet":EffNet().from_name('efficientnet-b0'), "knet": models.densenet121(), 
#         "vision": models.vit_b_16(), "cvt": CvtModel(CvtConfig()), "deit": DeiTForImageClassification.base_model, 
#         "vae": ae.VAE(), "auto":ae.AutoEncoder(), "resnet50": ae.BuildAutoEncoder("resnet50"), 
#         "byol": models.resnet50(), "byol2": BYOL(64,67)}
archs_weighted = {"resnet": models.resnet50(weights='ResNet50_Weights.DEFAULT'), "densenet": models.densenet121(weights='DenseNet121_Weights.DEFAULT'),
                  "effnet":EffNet.from_pretrained('efficientnet-b0'), "knet": models.densenet121(weights='DenseNet121_Weights.DEFAULT'),
                  "vision": models.vit_b_16(weights = 'ViT_B_16_Weights.DEFAULT'), "cvt":ConvNextForImageClassification.from_pretrained('facebook/convnext-tiny-224'),
                  "deit": DeiTForImageClassification.from_pretrained('facebook/deit-base-distilled-patch16-224'), 
                  "vae": ae.VAE(), "auto":ae.AutoEncoder(), "resnet50": ae.BuildAutoEncoder("resnet50"),
                  "byol": models.resnet50(weights='ResNet50_Weights.DEFAULT'), "byol2": BYOL(64,67)}


class Model(nn.Module):
    def __init__(self, model='densenet', eval=True, batch_size=32, num_features=128,
                 weight='weights', use_dr=True, device='cuda:0', freeze=False, classification = False, parallel = True, scratch = False):
        super(Model, self).__init__()
        self.parallel = parallel
        self.num_features = num_features
        self.norm = nn.functional.normalize
        self.weight = weight
        self.model_name = model
        self.device = device
        self.model_name = model
        self.classification = classification

        if model == 'deit' or model == 'vision' or model == 'swin' or model == 'cvt' or model == 'conv':
            self.transformer = True
        else:
            self.transformer = False

        if model == "knet"  and device=='cuda:1':
             os.environ["CUDA_VISIBLE_DEVICES"] = "1" # Data parallel module takes by default first gpu available -> so set only available to 1 and reindex it
             device = 'cuda:0'
        
        #--------------------------------------------------------------------------------------------------------------
        #                              Settings of deep ranking
        #--------------------------------------------------------------------------------------------------------------
        if use_dr and not self.transformer:
            out_features = 4096 # From deep ranking, 
            # Second and third network for deep ranking - parameters from the article (see page 18 of the thesis)
            # 3 input channels because 3 networks are put in // and fed a different image. 
            self.first_conv1 = nn.Conv2d(3, 96, kernel_size=8, padding=1, stride=16).to(device=device)
            self.first_conv2 = nn.MaxPool2d(3, 4, 1).to(device=device)

            self.second_conv1 = nn.Conv2d(3, 96, kernel_size=7, padding=4, stride=32).to(device=device)
            self.second_conv2 = nn.MaxPool2d(7, 2, 3).to(device=device)
            
            # Final linear embedding of deep ranking
            self.linear = nn.Linear(7168, num_features).to(device=device)
            self.use_dr = True
        else:
            out_features = num_features
            self.use_dr = False
        
        #--------------------------------------------------------------------------------------------------------------
        #                              Settings of the model
        #--------------------------------------------------------------------------------------------------------------
        if model in ['resnet', 'densenet', 'effnet', 'knet', 'vision', 'cvt', 'deit']:
            self.forward_function = self.forward_model

        if scratch is False:
            self.model = archs_weighted[model]
        else:
            self.model = archs_weighted[model].to(device=device) # TODO: change when archs are tested

        
        if model == "knet":
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.features = nn.Sequential(self.model.features , nn.AdaptiveAvgPool2d(output_size= (1,1)))
            self.model = self.model.to(device=device)
            num_ftrs = self.model.classifier.in_features
            self.model = fully_connected(self.model.features, num_ftrs, 30)
            self.model = self.model.to(device=device)
            self.model = nn.DataParallel(self.model)
            self.model.load_state_dict(torch.load('database/KimiaNet_Weights/weights/KimiaNetPyTorchWeights.pth'))
        elif model == 'vae':
            self.encode = self.model.encode
            self.reparameterize = self.model.reparameterize
            self.decode = self.model.decode
        elif model == 'resnet50':
            exp = self.model[1]
            if exp !=4 and exp != "3b":
                self.model = self.model[0].module
            else:
                self.model = self.model[0]
        elif model == 'byol':
            learner = BYOL_pytorch(
                self.model, 
                image_size = 224,
                hidden_layer = 'avgpool',
            )
            self.model = learner
        self.model = self.model.to(device=device)
        #----------------------------------------------------------------------------------------------------------------
        #                                  Freeze of model parameters
        #----------------------------------------------------------------------------------------------------------------
        if freeze and not self.transformer:
            for param in self.model.parameters():
                param.requires_grad = False
        elif freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        #----------------------------------------------------------------------------------------------------------------
        #                                   Modification of the model's last layer
        #----------------------------------------------------------------------------------------------------------------
        # in features parameters = the nb of features from the models last layer
        # Given the model, this layer has a different name: classifier densenet, .fc resnet, ._fc effnet,... 
        # Localisation can be found by displaying the model: print(self.model) and taking the last layer name
        if classification is True:
            out_features = 67
        if model == 'densenet':
            self.model.classifier = nn.Linear(self.model.classifier.in_features, out_features).to(device=device)
        elif model == 'resnet':
            self.model.fc = nn.Linear(self.model.fc.in_features, out_features).to(device=device)
        elif model == "effnet":
            self.model._fc = nn.Linear(self.model._fc.in_features, out_features).to(device=device)
        elif model == "knet":
            self.model.module.fc_4 = nn.Linear(self.model.module.fc_4.in_features, out_features).to(device=device)
        elif model == 'vision':
            self.model.heads.head = nn.Linear(self.model.heads.head.in_features, out_features).to(device=device)
        elif model == 'deit' or model == 'cvt':
            self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, num_features).to(device=device)
            if freeze: 
                for module in filter(lambda m: type(m) == nn.LayerNorm, self.model.modules()):
                    module.eval()
                    module.train = lambda _: None
        
        if eval == True:
            if model ==  'resnet50':
                if exp != "3b":   
                    ae.load_dict(weight, self.model)
                    self.model = self.model.module
                else:
                    self.load_state_dict(torch.load(weight))
            elif model == 'byol2':
                self.model.load_state_dict(torch.load(weight)["state_dict"])
                self.forward_function = self.model.forward
            else:
                self.load_state_dict(torch.load(weight))
            self.eval()
            self.eval = True
        else:
            if parallel:
                self.model = nn.DataParallel(self.model)
            self.train()
            self.eval = False
            self.batch_size = batch_size

    def forward_model(self, input):
        tensor1 = self.model(input)
        if self.model_name == 'deit' or self.model_name == 'cvt':
            tensor1 = tensor1.logits
        tensor1 = self.norm(tensor1)

        # deep ranking model requires additional norm
        if self.use_dr:
            tensor2 = self.first_conv1(input)
            tensor2 = self.first_conv2(tensor2)
            tensor2 = torch.flatten(tensor2, start_dim=1)

            tensor3 = self.second_conv1(input)
            tensor3 = self.second_conv2(tensor3)
            tensor3 = torch.flatten(tensor3, start_dim=1)

            tensor4 = self.norm(torch.cat((tensor2, tensor3), 1))

            return self.norm(self.linear(torch.cat((tensor1, tensor4), 1)))

        return tensor1

    def forward(self, input):
        return self.forward_function(input)
    
    def get_optim(self, data, loss, lr, decay, beta_lr, lr_proxies):

        if self.classification:
            loss_function = nn.CrossEntropyLoss() 
            to_optim = [{'params':self.parameters(),'lr':lr,'weight_decay':decay}]

            optimizer = torch.optim.Adam(to_optim)
        elif loss == 'margin':
            loss_function = MarginLoss(self.device, n_classes=len(data.classes))

            to_optim = [{'params':self.parameters(),'lr':lr,'weight_decay':decay},
                        {'params':loss_function.parameters(), 'lr':beta_lr, 'weight_decay':0}]

            optimizer = torch.optim.Adam(to_optim)
        elif loss == 'proxy_nca_pp':
            loss_function = ProxyNCA_prob(len(data.classes), self.num_features, 3, device)

            to_optim = [
                {'params':self.parameters(), 'weight_decay':0},
                {'params':loss_function.parameters(), 'lr': lr_proxies}, # Allows to update automaticcaly the proxies vectors when doing a step of the optimizer
            ]

            optimizer = torch.optim.Adam(to_optim, lr=lr, eps=1)
        elif loss == 'softmax':
            loss_function = NormSoftmax(0.05, len(data.classes), self.num_features, lr_proxies, self.device)

            to_optim = [
                {'params':self.parameters(),'lr':lr,'weight_decay':decay},
                {'params':loss_function.parameters(),'lr':lr_proxies,'weight_decay':decay}
            ]

            optimizer = torch.optim.Adam(to_optim)
        elif loss == 'softtriple':
            # Official - paper implementation
            loss_function = SoftTriple(self.device)
            to_optim = [{"params": self.parameters(), "lr": 0.0001},
                                  {"params": loss_function.parameters(), "lr": 0.01}] 
            optimizer = torch.optim.Adam(to_optim, eps=0.01, weight_decay=0.0001)
        else:
            to_optim = [{'params':self.parameters(),'lr':lr,'weight_decay':decay}] # For byol: lr = 3e-4
            optimizer = torch.optim.Adam(to_optim)
            loss_function = None
        return optimizer, loss_function

    def train_model(self, loss, epochs, training_dir, lr, decay, beta_lr, lr_proxies, sched, gamma, informative_samp = True, generalise = 0, load_kmeans = None):
        data = dataset.TrainingDataset(root = training_dir, model_name = self.model_name, samples_per_class= 2, generalise = generalise, load_kmeans = load_kmeans, informative_samp = informative_samp)
        print('Size of dataset', data.__len__())

        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size,
                                             shuffle=True, num_workers=16,
                                             pin_memory=True)
        
        optimizer, loss_function = self.get_optim(data, loss, lr, decay, beta_lr, lr_proxies)

        if sched == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif sched == 'step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs//2, epochs],
                                                            gamma=gamma)
        
        loss_stds = []
        loss_mean = []

        # Creation of the folder to save the weight
        weight_path = create_weights_folder(self.model_name)

        try:
            for epoch in range(epochs):
                start_time = time.time()
                loss_list = []
                for i, (labels, images) in enumerate(loader):
                    if i%1000 == 0:
                        print(i)
                    
                    images_gpu = images.to(self.device)
                    labels_gpu = labels.to(self.device)

                    # Autoencoders training
                    if self.model_name == "auto":
                        loss, _, _ = ae.grad_auto(self.model, images_gpu.view(-1, 3, 224, 224)) 
                    elif self.model_name == "vae":
                        recon_batch, mu, logvar = self.model(images_gpu)
                        loss = ae.loss_function(recon_batch, images_gpu.view(-1, 3, 224, 224), mu, logvar)
                    elif self.model_name == "resnet50":
                        out = self.model(images_gpu)
                        loss = nn.functional.mse_loss(images_gpu, out, reduction='mean') 
                    
                    # Byol training
                    elif self.model_name == "byol":
                        loss = self.model(images_gpu)
                    
                    # Supervised training
                    else:
                        if self.transformer:
                            out = self.forward(images_gpu.view(-1, 3, 224, 224))
                        else:
                            out = self.forward(images_gpu)
                        if loss_function is None:
                            print("This model requires a specific loss. Please specifiy one. ")
                            exit(-1)
                        loss = loss_function(out, labels_gpu)
                    
                    # Update
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                    if self.model_name == "byol":
                        self.model.update_moving_average()

                    loss_list.append(loss.item())
                
                print("epoch {}, loss = {}, time {}".format(epoch, np.mean(loss_list),
                                                            time.time() - start_time))
                
                print("\n----------------------------------------------------------------\n")
                loss_mean.append(np.mean(loss_list))
                loss_stds.append(np.std(loss_list))
                if sched != None:
                    scheduler.step()

                # Remove data parallel to save model
                try:
                    self.model = self.model.module
                except AttributeError:
                    self.model = self.model

                torch.save(self.state_dict(), weight_path + "/epoch_"+str(epoch))
                if self.parallel or self.model_name == "knet": # Knet is by default parallel 
                    self.model = nn.DataParallel(self.model)

            plt.errorbar(range(epochs), loss_mean, yerr=loss_stds, fmt='o--k',
                         ecolor='lightblue', elinewidth=3)
            plt.savefig(weight_path+"/training_loss.png")
                    
        
        except KeyboardInterrupt:
            print("Interrupted")
        

    def train_dr(self, data, num_epochs, lr, loss_name, augmented, contrastive, sched, gamma):
        
        # Loading of the data
        if loss_name == 'triplet' or (loss_name == 'infonce' and contrastive):
            pair = False
        else:
            pair = True
        if not augmented:
            augmented = None

        data = dataset.DRDataset(data, pair = pair, transform = augmented, contrastive=contrastive, appl=None)
        print('Size of dataset', data.__len__())

        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size,
                                             shuffle=True, num_workers=12,
                                             pin_memory=True)
        
        # Setting of the loss
        if loss_name == 'triplet':
            loss_function = torch.nn.TripletMarginLoss()
        elif loss_name == 'BCE':
            loss_function = SimpleBCELoss()
        elif loss_name == 'contrastive':
            loss_function = ContrastiveLoss()
        elif loss_name == 'cosine':
            loss_function = torch.nn.CosineEmbeddingLoss()
        elif loss_name == 'infonce':
            loss_function = InfoNCE(contrastive)
        else:
            print('Unknown loss function')
            return
        
        # setting of the optimizer & scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if sched == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif sched == 'step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[num_epochs//2, num_epochs],
                                                            gamma=gamma)
            
        # training loo
        loss_stds = []
        loss_means = []
        try:
            for epoch in range(num_epochs):
                start_time = time.time()
                loss_list = []
                for i, (image0, image1, image2) in enumerate(loader):
                    if i%1000 == 0:
                        print("at batch:"+str(i)+" on "+str(int(data.__len__()/self.batch_size)))

                    image0 = image0.to(device=self.device)
                    image1 = image1.to(device=self.device)

                    out0 = self.forward(image0).cpu()
                    out1 = self.forward(image1).cpu()

                    if not pair:
                        image2 = image2.to(device=self.device)
                        out2 = self.forward(image2).cpu()
                    else:
                        out2 = image2 
                        if loss_name == 'cosine':
                            out2[out2 == 1] = -1
                            out2[out2 == 0] = 1

                    loss = loss_function(out0, out1, out2)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                    loss_list.append(loss.item())
                    
                print("epoch {}, batch {}, loss = {}, time = {}".format(epoch, i,
                                                             np.mean(loss_list), time.time() - start_time))
                loss_means.append(np.mean(loss_list))
                loss_stds.append(np.std(loss_list)) 

                if sched != None:
                    scheduler.step()
                elif (epoch + 1) % 4:
                    lr /= 2
                    for param in optimizer.param_groups:
                        param['lr'] = lr

                try:
                    self.model = self.model.module
                except AttributeError:
                    self.model = self.model
                torch.save(self.state_dict(), self.name+str(epoch))
                if self.parallel:
                    self.model = nn.DataParallel(self.model).to(self.device)

            plt.errorbar(range(num_epochs), loss_means, yerr=loss_stds, fmt='o--k',
                         ecolor='lightblue', elinewidth=3)
            plt.savefig(self.model_name+"_loss.png")
        except KeyboardInterrupt:
            print("Interrupted")

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        '--num_features',
        type=int,
        help='number of features to use',
        default=128
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32
    )

    parser.add_argument(
        '--model',
        help='feature extractor to use',
        default='densenet'
    )

    parser.add_argument(
        '--weights',
        default='weights'
    )

    parser.add_argument(
        '--training_data',
    )

    parser.add_argument(
        '--dr_model',
        action="store_true"
    )

    parser.add_argument(
        '--num_epochs',
        type=int,
        default=10
    )

    parser.add_argument(
        '--scheduler',
        default='exponential',
        help='<exponential, step>'
    )

    parser.add_argument(
        '--gpu_id',
        default=0,
        type=int
    )

    parser.add_argument(
        '--loss',
        default=None,
        help='<margin, proxy_nca_pp, softmax, softtriple, triplet, contrastive, BCE, cosine>'
    )

    parser.add_argument(
        '--freeze',
        action='store_true'
    )

    parser.add_argument(
        '--generalise',
        default='0',
        help='train on only part of the classes of images',
        type=int
    )

    parser.add_argument(
        '--lr',
        default=0.0001,
        type=float
    )

    parser.add_argument(
        '--decay',
        default=0.0004,
        type=float
    )

    parser.add_argument(
        '--beta_lr',
        default=0.0005,
        type=float
    )

    parser.add_argument(
        '--gamma',
        default=0.3,
        type=float
    )

    parser.add_argument(
        '--lr_proxies',
        default=0.00001,
        type=float
    )

    parser.add_argument(
        '--classification',
        action='store_true'
    )

    parser.add_argument(
        '--parallel',
        action = 'store_true'
    )

    parser.add_argument(
        '--scratch',
        action = 'store_true'
    )

    parser.add_argument(
        '--load',
        default = None,
        type = str
    )

    parser.add_argument(
        '--augmented',
        action = 'store_true'
    )

    parser.add_argument(
        '--non_contrastive',
        action = 'store_true'
    )

    parser.add_argument(
        '--i_sampling',
        default = None,
        help='whether or not ot use informative (label-based) sampling',
        action='store_true'
    )



    args = parser.parse_args()

    if args.gpu_id >= 0:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'
    if args.generalise != 3 and args.load is True:
        print("Load cannot be used if the 3rd mode is not activated for generalize")
        exit(-1)
    if args.i_sampling is None:
        if args.model in ['auto', 'vae', 'resnet50', 'byol', 'byol2']:
            args.i_sampling = False
        else:
            args.i_sampling = True

    m = Model(model=args.model, eval=False, batch_size=args.batch_size,
              num_features=args.num_features, weight=args.weights,
              use_dr=args.dr_model, device=device, freeze=args.freeze, classification = args.classification, parallel=args.parallel, scratch=args.scratch)

    siamese_losses = ['triplet', 'contrastive', 'BCE', 'cosine', 'infonce']
    if args.loss in siamese_losses:
        m.train_dr(data = args.training_data, num_epochs = args.num_epochs, lr = args.lr, loss_name = args.loss, augmented=args.augmented, contrastive = not args.non_contrastive, sched= args.scheduler, gamma= args.gamma)
    else: 
        m.train_model(loss = args.loss, epochs = args.num_epochs, training_dir=args.training_data, sched = args.scheduler, lr = args.lr, decay = args.decay, gamma = args.gamma, beta_lr = args.beta_lr, lr_proxies = args.lr_proxies, informative_samp = args.i_sampling, generalise = args.generalise, load_kmeans = args.load)
    
    