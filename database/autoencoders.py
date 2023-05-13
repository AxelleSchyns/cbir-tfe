import vgg, resnet
import torch.nn as nn
import torch
import torch.nn.functional as F



#---------------------------------------------------------------------------------------------------------
#                                          Implementation 1 
#---------------------------------------------------------------------------------------------------------
def load_pretrained(model):
    checkpoint = torch.load("/home/labarvr4090/Documents/Axelle/cytomine/cbir-tfe/weights_folder/imagenet-vgg16.pth")
    model_dict = model.state_dict()
    model_dict.update(checkpoint['state_dict'])
    model.load_state_dict(model_dict)
    del checkpoint
    return model

def BuildAutoEncoder(model_name):
    exp = 0
    if model_name in ["vgg11", "vgg16"]:
        configs = vgg.get_configs(model_name, exp)
        model = vgg.VGGAutoEncoder(configs)

    elif model_name in ["resnet18", "resnet50"]:
        configs, bottleneck = resnet.get_configs(model_name, exp)
        model = resnet.ResNetAutoEncoder(configs, bottleneck)
    if exp == 4: 
        model = load_pretrained(model)
    return model


#---------------------------------------------------------------------------------------------------------
#                                          Implementation 2
#---------------------------------------------------------------------------------------------------------
# Pytorch example - VAE - https://github.com/pytorch/examples/blob/main/vae/main.py
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.exp = "1"

        if self.exp == "1":
            print("1")
        elif self.exp == "2a":
            print("2a")
        elif self.exp == "2b":
            print("2b")
        elif self.exp == "2c":
            self.fc0 = nn.Linear(224*224*3,1500)
            self.fc1 = nn.Linear(1500, 750)
            self.fc21 = nn.Linear(750, 128)
            self.fc22 = nn.Linear(750, 128)
            self.fc3 = nn.Linear(128, 750)
            self.fc4 = nn.Linear(750, 1500)
            self.fc5 = nn.Linear(1500, 224*224*3)
        elif self.self.exp == "3":
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
            print("1")
        elif self.exp == "2a":
            print("2a")
        elif self.exp == "2b":
            print("2b")
        elif self.exp == "2c":
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
        if self.exp == "1":
            print("1")
        elif self.exp == "2a":
            print("2a")
        elif self.exp == "2b":
            print("2b")
        elif self.exp == "2c":
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
        if self.exp == "1":
            print("1")
        elif self.exp == "2a":
            print("2a")
        elif self.exp == "2b":
            print("2b")
        elif self.exp == "2c":
            mu, logvar = self.encode(x.view(-1, 224*224*3))
        elif self.exp == "3":
            mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
        # 784 -> 400 -> 200 -> 200 -> 400 -> 784
        """


    def forward(self, x):
        #mu, logvar = self.encode(x.view(-1, 784))
        mu, logvar = self.encode(x.view(-1, 224*224*3))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar"""

# Pytorch exampe VAE
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    exp = "1"

    if exp == "1":
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    elif exp == "2a" or exp == "2b" or exp == "2c":
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

