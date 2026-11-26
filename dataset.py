import torch
from torch.utils.data import Dataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize from 500x500 to 224x224
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.transform = transform

        assert len(images) == len(labels), \
            "Number of images and labels must match"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.as_tensor(
            self.images[idx], dtype=torch.float32
        )

        label = torch.as_tensor(
            self.labels[idx], dtype=torch.long
        )

        # Add channel dimension for grayscale images
        if image.ndim == 2:
            image = image.unsqueeze(0)

        # Convert grayscale to 3 channels (RGB)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        # Resize and normalize the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label