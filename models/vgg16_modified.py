import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class VGG16Model(nn.Module):

    def __init__(self, num_classes=10, pretrained=True):

        super(VGG16Model, self).__init__()

        # Load VGG16
        if pretrained:
            weights = VGG16_Weights.DEFAULT
        else:
            weights = None

        self.vgg = vgg16(weights=weights)

        # Freeze convolutional layers ONLY if pretrained
        if pretrained:
            for param in self.vgg.features.parameters():
                param.requires_grad = False

        # Replace final classification layer
        self.vgg.classifier[6] = nn.Linear(
            self.vgg.classifier[6].in_features,
            num_classes
        )

    def forward(self, x):
        return self.vgg(x)