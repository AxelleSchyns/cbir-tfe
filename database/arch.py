
import torch
import copy
import torch.nn as nn
import torchvision.models as models
from torch import Tensor
from torch.nn import Identity


from pytorch_lightning import LightningModule
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.loss import NegativeCosineSimilarity

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

class BYOL(LightningModule):
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
        return self.projection_head(x)