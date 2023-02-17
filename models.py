import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from transformers import DeiTForImageClassification
import dataset
import numpy as np
import time
from loss import MarginLoss, ProxyNCA_prob, NormSoftmax
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
from transformers import CvtForImageClassification, ConvNextForImageClassification, AutoImageProcessor, ConvNextFeatureExtractor
from torchvision import transforms
from argparse import ArgumentParser, ArgumentTypeError

class fully_connected(nn.Module):
	"""docstring for BottleNeck"""
	def __init__(self, model, num_ftrs, num_classes):
		super(fully_connected, self).__init__()
		self.model = model
		self.fc_4 = nn.Linear(num_ftrs,num_classes)

	def forward(self, x):
		x = self.model(x)
		x = torch.flatten(x, 1)
		out_3 = self.fc_4(x)
		return  out_3
class Model(nn.Module):
    def __init__(self, model='densenet', eval=True, batch_size=32, num_features=128,
                 name='weights', use_dr=True, device='cuda:0', freeze=False):
        super(Model, self).__init__()
        print(device)
        self.num_features = num_features
        self.norm = nn.functional.normalize
	
        if model == 'deit' or model == 'vision' or model == 'swin' or model == 'cvt' or model == 'conv':
        	self.transformer = True
        else:
        	self.transformer = False

        self.name = name
        self.device = device
        self.model_name = model
        #--------------------------------------------------------------------------------------------------------------
        #                              Settings of deep ranking
        #--------------------------------------------------------------------------------------------------------------
        if use_dr and not self.transformer:
            out_features = 4096 # From deep ranking, 
            # Second and third network for deep ranking - parameters from the article (see page 18 of the thesis)
            # 3 input channels because 3 netwroks are put in // and fed a different image. 
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
        
        #----------------------------------------------------------------------------------------------------------------
        #                              Download of pretrained models
        #----------------------------------------------------------------------------------------------------------------
        if model == 'densenet':
            self.forward_function = self.forward_model
            self.model = models.densenet121(weights='DenseNet121_Weights.DEFAULT').to(device=device)
            #self.model = models.densenet121(weights=None).to(device=device)
        elif model == 'resnet':
            self.forward_function = self.forward_model
            self.model = models.resnet50(weights='ResNet50_Weights.DEFAULT').to(device=device)
        elif model == "vgg":
            self.forward_function = self.forward_model
            self.model = models.vgg19(weights="VGG19_Weights.DEFAULT").to(device=device)
        elif model == "inception":
            self.forward_function = self.forward_model
            self.model = models.inception_v3(weights="Inception_V3_Weights.DEFAULT").to(device=device)
        elif model == "effnet":
            self.forward_function = self.forward_model
            self.model = EfficientNet.from_pretrained('efficientnet-b0').to(device=device)
        elif model == "knet":
            model = models.densenet121(weights='DenseNet121_Weights.DEFAULT')
            for param in model.parameters():
                param.requires_grad = False
            model.features = nn.Sequential(model.features , nn.AdaptiveAvgPool2d(output_size= (1,1)))
            num_ftrs = model.classifier.in_features
            model_final = fully_connected(model.features, num_ftrs, 30)
            model = model.to(device)
            model_final = model_final.to(device)
            model_final = nn.DataParallel(model_final)
            model_final.load_state_dict(torch.load('database/KimiaNet_Weights/weights/KimiaNetPyTorchWeights.pth'))
            self.model = model_final
            self.forward_function = self.forward_model
            model = 'knet'
        elif model == "swin":
            self.forward_function = self.forward_model
            self.model = models.swin_v2_b(weights='Swin_V2_B_Weights.DEFAULT').to(device=device)
        elif model == 'vision':
            self.forward_function = self.forward_model
            self.model = models.vit_b_16(weights = 'ViT_B_16_Weights.DEFAULT').to(device=device)
        elif model == "conv":
            self.model = ConvNextForImageClassification.from_pretrained('facebook/convnext-tiny-224').to(device=device)
            self.forward_function = self.forward_model
        elif model == "cvt":
            print("hey")
            self.model =  CvtForImageClassification.from_pretrained('microsoft/cvt-21').to(device=device)
            print("hey2")
            self.forward_function = self.forward_model
        elif model == 'deit':
            self.forward_function = self.forward_model
            self.model = DeiTForImageClassification.from_pretrained('facebook/deit-base-distilled-patch16-224').to(device=device)
        
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
 	# in features parameters the nb of features from the models last layer
 	# Given the model, this layer has a different name: classifier densenet, .fc resnet, ._fc effnet,... 
 	# Localisation can be found by displaying the model: print(self.model) and taking the last layer name
 	
        if model == 'densenet':
            self.model.classifier = nn.Linear(self.model.classifier.in_features, out_features).to(device=device)
        elif model == 'resnet':
            self.model.fc = nn.Linear(self.model.fc.in_features, out_features).to(device=device)
        elif model == 'inception':
            self.model.fc = nn.Linear(self.model.fc.in_features, out_features).to(device=device)
            self.model.aux_logits = False
        elif model == "vgg":
            self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, out_features).to(device=device)
        elif model == "effnet":
            self.model._fc = nn.Linear(self.model._fc.in_features, out_features).to(device=device)
        elif model == "knet":
            self.model.module.fc_4 = nn.Linear(self.model.module.fc_4.in_features, out_features).to(device=device)
        elif model == "swin":
            self.model.head = nn.Linear(self.model.head.in_features, out_features).to(device=device)
        elif model == 'vision':
            self.model.heads.head = nn.Linear(self.model.heads.head.in_features, out_features).to(device=device)
        elif model == 'deit' or model == 'conv' or model == 'cvt':
            self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, num_features).to(device=device)
            for module in filter(lambda m: type(m) == nn.LayerNorm, self.model.modules()):
                module.eval()
                module.train = lambda _: None

        if eval == True:
            self.load_state_dict(torch.load(name))
            self.eval()
            self.eval = True
        else:
            self.train()
            self.eval = False
            self.batch_size = batch_size

    def forward_model(self, input):
        tensor1 = self.model(input)
        if self.model_name == 'deit' or self.model_name == 'conv' or self.model_name == 'cvt':
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

    def train_epochs(self, model, dir, epochs, sched, loss, generalise, lr, decay, beta_lr, gamma, lr_proxies):
        data = dataset.TrainingDataset(dir, model, 2, generalise, self.transformer)
        print('Size of dataset', data.__len__())

        if loss == 'margin':
            loss_function = MarginLoss(self.device, n_classes=len(data.classes))

            to_optim = [{'params':self.parameters(),'lr':lr,'weight_decay':decay},
                        {'params':loss_function.parameters(), 'lr':beta_lr, 'weight_decay':0}]

            optimizer = torch.optim.Adam(to_optim)
        elif loss == 'proxy_nca_pp':
            loss_function = ProxyNCA_prob(len(data.classes), self.num_features, 3, device)

            to_optim = [
                {'params':self.parameters(), 'weight_decay':0},
                {'params':loss_function.parameters(), 'lr': lr_proxies},
            ]

            optimizer = torch.optim.Adam(to_optim, lr=lr, eps=1)
        elif loss == 'softmax':
            loss_function = NormSoftmax(0.05, len(data.classes), self.num_features, lr_proxies, self.device)

            to_optim = [
                {'params':self.parameters(),'lr':lr,'weight_decay':decay},
                {'params':loss_function.parameters(),'lr':lr_proxies,'weight_decay':decay}
            ]

            optimizer = torch.optim.Adam(to_optim)

        if sched == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif sched == 'step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs//2, epochs],
                                                            gamma=gamma)

        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size,
                                             shuffle=True, num_workers=12,
                                             pin_memory=True)

        loss_list = []
        try:
            for epoch in range(epochs):
                start_time = time.time()

                for i, (labels, images) in enumerate(loader):
                    if i%1000 == 0:
                        print(i)
                    if model == 'inception': # Inception needs images of size at least 299 by 299
                    	images = transforms.Resize((299,299))(images)
                    images_gpu = images.to(device=self.device)
                    labels = labels.to(device=self.device)

                    if not self.transformer:
                        out = self.forward(images_gpu)
                    else:
                        out = self.forward(images_gpu.view(-1, 3, 224, 224))

                    loss = loss_function(out, labels)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                    loss_list.append(loss.item())

                print("epoch {}, loss = {}, time {}".format(epoch, np.mean(loss_list),
                                                            time.time() - start_time))

                print("\n----------------------------------------------------------------\n")
                if sched != None:
                    scheduler.step()

                torch.save(self.state_dict(), self.name)

        except KeyboardInterrupt:
            print("Interrupted")

    def train_dr(self, data, num_epochs, lr):
        data = dataset.DRDataset(data)
        print('Size of dataset', data.__len__())

        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size,
                                             shuffle=True, num_workers=12,
                                             pin_memory=True)
        loss_function = torch.nn.TripletMarginLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_list = []
        try:
            for epoch in range(num_epochs):
                start_time = time.time()

                for i, (image0, image1, image2) in enumerate(loader):
                    image0 = image0.to(device='cuda:0')
                    image1 = image1.to(device='cuda:0')
                    image2 = image2.to(device='cuda:0')

                    out0 = self.forward(image0).cpu()
                    out1 = self.forward(image1).cpu()
                    out2 = self.forward(image2).cpu()

                    loss = loss_function(out0, out1, out2)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                    loss_list.append(loss.item())

                print("epoch {}, batch {}, loss = {}".format(epoch, i,
                                                             np.mean(loss_list)))
                loss_list.clear()
                print("time for epoch {}".format(time.time()- start_time))

                torch.save(self.state_dict(), self.name)

                if (epoch + 1) % 4:
                    lr /= 2
                    for param in optimizer.param_groups:
                        param['lr'] = lr
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
        default=5
    )

    parser.add_argument(
        '--scheduler',
        default=None,
        help='<exponential, step>'
    )

    parser.add_argument(
        '--gpu_id',
        default=0,
        type=int
    )

    parser.add_argument(
        '--loss',
        default='margin',
        help='<margin, proxy_nca_pp, softmax, deep_ranking>'
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

    args = parser.parse_args()

    if args.gpu_id >= 0:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'

    m = Model(model=args.model, eval=False, batch_size=args.batch_size,
              num_features=args.num_features, name=args.weights,
              use_dr=args.dr_model, device=device, freeze=args.freeze)

    if args.loss == 'deep_ranking':
        m.train_dr(args.training_data, args.num_epochs, args.lr)
    else:
        m.train_epochs(args.model, args.training_data, args.num_epochs, args.scheduler, args.loss, args.generalise,
                       args.lr, args.decay, args.beta_lr, args.gamma, args.lr_proxies)
