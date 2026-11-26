# Nested VGG16 class with dfn's
import torch
import torch.nn as nn
from torchvision import models
import sys
import os

# Add HWD+ (project root) to Python path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

# Custom functions
from functions.sparsemax import sparsemax
from functions.create_dfn_v1 import create_dfn

class Vgg16_dfn(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()

        self.num_classes = num_classes
        # First section
        weights = (
            models.VGG16_Weights.DEFAULT if pretrained else None
        )
        self.nn1 = models.vgg16(weights=weights)

        # Replace the final classification layer
        self.nn1.classifier[6] = nn.Linear(
            self.nn1.classifier[6].in_features,
            num_classes
        )
        
        # Number of parameters in the final classification layer
        final_layer = self.nn1.classifier[6]

        self.num_first_weights = (
            final_layer.weight.numel()
            + final_layer.bias.numel()
        )
        
        # Input of NN2:
        nn2_input_size = num_classes + self.num_first_weights

        # Second section [pi.weights] into [vectors]
        self.nn2 = nn.Sequential(
            nn.Linear(nn2_input_size, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(
                128,
                (num_classes) * (num_classes + 1)
            )
        )
    def forward(self, images):

        batch_size = images.shape[0]

        # ================================================
        # STAGE 1: IMAGE -> pi
        # ================================================

        class_logits = self.nn1(images)

        # Each pi lies between 0 and 1
        pi = torch.softmax(class_logits, dim=1)

        # ================================================
        # STAGE 2: EXTRACT WEIGHTS FROM NN1
        # ================================================

        final_layer = self.nn1.classifier[6]

        first_weights = torch.cat([
            final_layer.weight.reshape(-1),
            final_layer.bias.reshape(-1)
        ])

        # Repeat the same learned weights for each image.
        # This operation preserves gradient flow.
        first_weights = first_weights.unsqueeze(0).expand(
            batch_size, -1
        )

        # ================================================
        # STAGE 3: COMBINE pi AND FIRST-NETWORK WEIGHTS
        # ================================================

        nn2_input = torch.cat(
            [pi, first_weights],
            dim=1
        )

        # ================================================
        # STAGE 4: PRODUCE VECTORS
        # ================================================

        vector_logits = self.nn2(nn2_input)

        vector_logits = vector_logits.reshape(
            batch_size,
            self.num_classes,
            self.num_classes + 1
        )

        # Split each vector according to its index
        split_vectors = []
        dfn_vectors = []
        for i in range(self.num_classes):
            # Take the vector at index i
            current_logits = vector_logits[:, i, :]
        
            # First part: i + 1 components
            first_logits = current_logits[:, :i + 1]
        
            # Second part: remaining components
            second_logits = current_logits[:, i + 1:]
        
            # Normalize each part separately
            first_vector = sparsemax(first_logits, dim=1)
            second_vector = sparsemax(second_logits, dim=1)
        
            split_vectors.append((first_vector, second_vector))

            #Create the dfn
            dfn = create_dfn(first_vector,second_vector)
            dfn_vectors.append(dfn)

        return pi, split_vectors, dfn_vectors, class_logits
