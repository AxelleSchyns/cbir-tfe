import numpy as np
import vgg, resnet
import torch.nn as nn
import torch
import torch.nn.functional as F
import os
# This file contains 3 implementations of the autoencoder model

#---------------------------------------------------------------------------------------------------------
#                                          Implementation 1 
#---------------------------------------------------------------------------------------------------------
# Horizon2333, Github - https://github.com/Horizon2333/imagenet-autoencoder

# function to load the weights of the model pretrained on ImageNet (exp 4)
def load_pretrained(model):
    checkpoint = torch.load("/home/labarvr4090/Documents/Axelle/cytomine/cbir-tfe/weights_folder/imagenet-vgg16.pth")
    model_dict = model.state_dict()
    model_dict.update(checkpoint['state_dict'])
    model.load_state_dict(model_dict)
    del checkpoint
    return model

# function to load the weights of the model pretrained previsouly 
def load_dict(resume_path, model):
    if os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path)
        model_dict = model.state_dict()
        model_dict.update(checkpoint['state_dict'])
        model.load_state_dict(model_dict)
        del checkpoint
    else:
        sys.exit("=> No checkpoint found at '{}'".format(resume_path))
    return model

# function to build the autoencoder model
def BuildAutoEncoder(model_name):
    exp = 4 # set the experiment number 
    if model_name in ["vgg11", "vgg16"]:
        configs = vgg.get_configs(model_name)
        model = vgg.VGGAutoEncoder(configs, exp)

    elif model_name in ["resnet18", "resnet50"]:
        configs, bottleneck = resnet.get_configs(model_name)
        model = resnet.ResNetAutoEncoder(configs, bottleneck)
    if exp != "3b":
        model = nn.DataParallel(model)
    if exp == 4: 
        model = load_pretrained(model)
    return model, exp


#---------------------------------------------------------------------------------------------------------
#                                          Implementation 2
#---------------------------------------------------------------------------------------------------------
# Pytorch example - VAE - https://github.com/pytorch/examples/blob/main/vae/main.py

# function to build the autoencoder model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.exp = "2a"

        if self.exp == "1":
            self.fc1 = nn.Linear(784, 400)
            self.fc21 = nn.Linear(400, 20)
            self.fc22 = nn.Linear(400, 20)
            self.fc3 = nn.Linear(20, 400)
            self.fc4 = nn.Linear(400, 784)
        elif self.exp == "2a":
            self.fc1 = nn.Linear(224*224*3, 400)
            self.fc21 = nn.Linear(400, 20)
            self.fc22 = nn.Linear(400, 20)
            self.fc3 = nn.Linear(20, 400)
            self.fc4 = nn.Linear(400, 224*224*3)
        elif self.exp == "2b":
            self.fc0 = nn.Linear(224*224*3, 784)
            self.fc1 = nn.Linear(784, 400)
            self.fc21 = nn.Linear(400, 20)
            self.fc22 = nn.Linear(400, 20)
            self.fc3 = nn.Linear(20, 400)
            self.fc4 = nn.Linear(400, 784)
            self.fc5 = nn.Linear(784, 224*224*3)
        elif self.exp == "2c":
            self.fc0 = nn.Linear(224*224*3,1500)
            self.fc1 = nn.Linear(1500, 750)
            self.fc21 = nn.Linear(750, 128)
            self.fc22 = nn.Linear(750, 128)
            self.fc3 = nn.Linear(128, 750)
            self.fc4 = nn.Linear(750, 1500)
            self.fc5 = nn.Linear(1500, 224*224*3)
        elif self.exp == "3":
            # Encoder layers
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
            self.fc1 = nn.Linear(128 * 56 * 56, 1000)
            self.fc21 = nn.Linear(1000, 128)
            self.fc22 = nn.Linear(1000, 128)

            # Decoder layers
            self.fc3 = nn.Linear(128, 1000)
            self.fc4 = nn.Linear(1000, 128*56*56)
            self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.conv6 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)

    def encode(self, x):
        if self.exp == "1":
            x= x.view(-1, 784)
        elif self.exp == "2a" or self.exp == "2b" or self.exp == "2c" :
            x = x.view(-1, 224*224*3)

        if self.exp == "1" or self.exp == "2a":
            h1 = F.relu(self.fc1(x))
            return self.fc21(h1), self.fc22(h1)
        elif self.exp == "2b" or self.exp == "2c":
            h0 = F.relu(self.fc0(x))
            h1 = F.relu(self.fc1(h0))
            return self.fc21(h1), self.fc22(h1)
        elif self.exp == "3":
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            if len(x.shape) > 3:
                x = x.view(x.size(0), -1)
            else:
                x = x.view(1, -1)
            x = F.relu(self.fc1(x))
            return self.fc21(x), self.fc22(x)

    def decode(self, z):
        if self.exp == "1" or self.exp == "2a":
            h3 = F.relu(self.fc3(z))
            return torch.sigmoid(self.fc4(h3))
        elif self.exp == "2b" or self.exp == "2c":
            h3 = F.relu(self.fc3(z))
            h4 = F.relu(self.fc4(h3))
            return torch.sigmoid(self.fc5(h4))
        elif self.exp == "3":
            z = F.relu(self.fc3(z))
            z = F.relu(self.fc4(z))
            z = z.view(z.size(0), 128, 56, 56)
            z = F.relu(self.conv4(z))
            z = F.relu(self.conv5(z))
            return torch.sigmoid(self.conv6(z))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
        # 784 -> 400 -> 200 -> 200 -> 400 -> 784
    

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    exp = "2a"

    if exp == "1":
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    elif exp == "2a":
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 224*224*3), reduction='sum') 
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) 

        if torch.isnan(BCE):
            print("HEY")
            BCE = torch.tensor(np.finfo(np.float64).max)
        if torch.isnan(KLD):
            print("HEY")
            KLD = torch.tensor(np.finfo(np.float64).max)
        return BCE + KLD
    elif exp == "2b" or exp == "2c":
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 224*224*3), reduction='sum')
    elif exp == "3":
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    #
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


#----------------------------------------------------------------------------------------------------
#                                    Implementation 3
#----------------------------------------------------------------------------------------------------
# From Geek For geek - https://www.geeksforgeeks.org/contractive-autoencoder-cae/?ref=rp

# Building the model
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        exp = 5 # set the experiment number
        self.flatten_layer = nn.Flatten()

        if exp == 0 or exp == 1:
            self.dense1 = nn.Linear(784, 64)
            self.dense2 = nn.Linear(64, 32)
            self.bottleneck = nn.Linear(32, 16)
            self.dense4 = nn.Linear(16, 32)
            self.dense5 = nn.Linear(32, 64)
            self.dense_final = nn.Linear(64, 784)
        elif exp == 2 or exp == 3:
            self.dense1 = nn.Linear(784, 400)
            self.dense2 = nn.Linear(400, 200)
            if exp == 2:
                self.bottleneck = nn.Linear(200, 16)
                self.dense4 = nn.Linear(16, 200)
            elif exp == 3:
                self.bottleneck = nn.Linear(200, 50)
                self.dense4 = nn.Linear(50, 200)
            self.dense5 = nn.Linear(200, 400)
            self.dense_final = nn.Linear(400, 784)
        elif exp == 4:
            # 224*224*3 -> 64 -> 32 -> 16 
            self.dense1 = nn.Linear(224*224*3, 64)
            self.dense2 = nn.Linear(64, 32)
            self.bottleneck = nn.Linear(32, 16)
            self.dense4 = nn.Linear(16, 32)
            self.dense5 = nn.Linear(32, 64)
            self.dense_final = nn.Linear(64, 224*224*3)

        elif exp == 5:
            self.dense1 = nn.Linear(224*224*3, 512)
            self.dense2 = nn.Linear(512, 256)
            self.bottleneck = nn.Linear(256, 128)
            self.dense4 = nn.Linear(128, 256)
            self.dense5 = nn.Linear(256, 512)
            self.dense_final = nn.Linear(512, 224*224*3)
        elif exp == 6:
            # 224*224*3 -> 3000 -> 1500 -> 750
            self.dense1 = nn.Linear(224*224*3, 3000)
            self.dense2 = nn.Linear(3000, 1000)
            self.bottleneck = nn.Linear(1000, 750)
            self.dense4 = nn.Linear(750, 1000)
            self.dense5 = nn.Linear(1000, 3000)
            self.dense_final= nn.Linear(3000, 224*224*3)
        self.exp = exp

    def forward(self, inp):
        if self.exp == 0 or self.exp == 1 or self.exp == 2 or self.exp == 3:
            inp = inp.view(-1, 784)
        elif self.exp == 4 or self.exp == 5 or self.exp == 6:
            inp = inp.view(-1, 224*224*3)

        if self.exp == 0:
            x_reshaped = self.flatten_layer(inp)
            h1 = torch.sigmoid(self.dense1(x_reshaped))
            h2 = torch.sigmoid(self.dense2(h1))
            h3 = torch.sigmoid(self.bottleneck(h2))
            h4 = torch.sigmoid(self.dense4(h3))
            h5 = torch.sigmoid(self.dense5(h4))
            x = self.dense_final(h5)

            # Compute the weight matrix for the loss function 
            ws = None
            for i in range(int(x.shape[0]/192)):
                W = torch.matmul(torch.diag_embed(h1[i:i+192, :] * (1 - h1[i:i+192, :])), self.dense1.weight)
                W = torch.matmul(self.dense2.weight, W)
                W = torch.matmul(torch.diag_embed(h2[i:i+192, :] * (1 - h2[i:i+192, :])), W)
                W = torch.matmul(self.bottleneck.weight, W)
                W = torch.matmul(torch.diag_embed(h3[i:i+192, :] * (1 - h3[i:i+192, :]))  , W)

                if ws is None:
                    ws = W
                else:
                    ws = torch.cat((ws, W), axis=0)
            return x, x_reshaped, h3, ws
        else:
            #x_reshaped = inp #
            x_reshaped = self.flatten_layer(inp)
            x = nn.functional.relu(self.dense1(x_reshaped))
            x = nn.functional.relu(self.dense2(x))
            x = nn.functional.relu(self.bottleneck(x))
            x_hid = x
            x = nn.functional.relu(self.dense4(x))
            x = nn.functional.relu(self.dense5(x))
            #x = nn.functional.relu(self.dense6(x))
            x = self.dense_final(x)

        return x, x_reshaped, x_hid, self.bottleneck.weight

# Loss function (part 1)
def loss_auto(x, x_bar, h, W, model):
    exp = 5
    if exp == 0:
        reconstruction_loss = nn.functional.mse_loss(x, x_bar, reduction='mean')
        contractive = torch.sum(W**2, axis=(1,2))
        total_loss = reconstruction_loss + 1000 * contractive.mean()
    elif exp == 1 or exp == 2 or exp == 3:
        reconstruction_loss = nn.functional.mse_loss(x, x_bar, reduction='mean') * 784
        dh = h * (1 - h) # N_batch x N_hidden
        contractive = 100 * torch.sum(torch.matmul(dh**2, torch.square(W)), axis=1)
        total_loss = reconstruction_loss + contractive.mean()
    elif exp == 4 or exp == 5 or exp == 6:
        reconstruction_loss = nn.functional.mse_loss(x, x_bar, reduction='mean') * 224*224*3
        dh = h * (1 - h) # N_batch x N_hidden
        contractive = 100 * torch.sum(torch.matmul(dh**2, torch.square(W)), axis=1)
        total_loss = reconstruction_loss + contractive.mean()
    return total_loss

# Loss function (part 2) 
def grad_auto(model, inputs):
    exp = 5
    if exp == 0 or exp == 1 or exp == 2 or exp == 3:
        reconstruction, inputs_reshaped, hidden, W = model(inputs.view(-1, 784))
    else:
        reconstruction, inputs_reshaped, hidden, W = model(inputs.view(-1, 224*224*3))
    loss_value = loss_auto(inputs_reshaped, reconstruction, hidden, W, model)
    return loss_value, inputs_reshaped, reconstruction

