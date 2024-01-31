
import torch
import torch.nn as nn
import cdpath_models as cdpath_models 
from torchvision import models as torchvision_models
import vision_transformer as vits
import utils_dino 
from vision_transformer import DINOHead
"""from pytorch_lightning import LightningModule
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.loss import NegativeCosineSimilarity"""
import torch.nn.functional as F
# From Kimia Lab implementation 
class fully_connected(nn.Module):
	def __init__(self, model, num_ftrs, num_classes):
		super(fully_connected, self).__init__()
		self.model = model
		self.fc_4 = nn.Linear(num_ftrs,num_classes)

	def forward(self, x):
		x = self.model(x)
		x = torch.flatten(x, 1)
		out_3 = self.fc_4(x)
		return  out_3

"""class BYOL(LightningModule):
    def __init__(self, batch_size_per_device: int, num_classes: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size_per_device = batch_size_per_device

        resnet = models.resnet50()
        resnet.fc = Identity()  # Ignore classification head
        self.backbone = resnet
        self.projection_head = BYOLProjectionHead()
        self.student_backbone = copy.deepcopy(self.backbone)
        self.student_projection_head = BYOLProjectionHead()
        self.student_prediction_head = BYOLPredictionHead()
        self.criterion = NegativeCosineSimilarity()

        self.online_classifier = OnlineLinearClassifier(num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        return self.projection_head(x)"""

class KNet():
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        for param in model.parameters():
                param.requires_grad = False
        self.model.features = nn.Sequential(model.features , nn.AdaptiveAvgPool2d(output_size= (1,1)))
        num_ftrs = self.model.classifier.in_features
        self.model = fully_connected(self.model.features, num_ftrs, 30)
        self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load('database/KimiaNet_Weights/weights/KimiaNetPyTorchWeights.pth'))
        
class DINO():
    def __init__(self, model_name):
        super().__init__()
        if model_name in vits.__dict__.keys():
            self.model = vits.__dict__[model_name](patch_size=16,)
        elif model_name in torchvision_models.__dict__.keys():
            self.model = torchvision_models.__dict__[model_name]()

        self.model = utils_dino.MultiCropWrapper(self.model, DINOHead(
        2048,
        65536,
        use_bn=False,
        norm_last_layer=True
        ))
        self.model = self.model.backbone
    def load_weights(self, weight_path):
        state_dict = torch.load(weight_path)
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict['student'].items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k: v for k, v  in state_dict.items() if k.startswith("backbone.")}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = self.model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(weight_path, msg))
        self.model.eval()

    def to(self, device):
        self.model = self.model.to(device)
        return self

class cdpath():
    def __init__(self) -> None:
        super().__init__()
        self.generator = cdpath_models.Generator(64, 112, 512, backbone=True)

        self.discriminator = cdpath_models.Discriminator(64)
        self.discriminator_gauss = cdpath_models.Discriminator_Gauss(512)
    
    def load_weights(self, weight_path):
        checkpoint = torch.load(weight_path)
        self.generator = checkpoint['generator']
        self.discriminator = checkpoint['discriminator']
        self.discriminator_gauss = checkpoint['discriminator_gauss']
    
    def to(self, device):
        self.model = self.generator.to(device)
        return self
    

def scale_generator(x_batch, size, alpha, x_dim, rescale_size=224):

    x_batch = F.interpolate(x_batch, size=rescale_size, mode='bilinear',
                            align_corners=True)

    margin = (rescale_size - x_dim) // 2
    x_crop = x_batch[:, :, margin:rescale_size-margin, margin:rescale_size-margin]


    return x_crop