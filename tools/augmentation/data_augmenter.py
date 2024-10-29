import PIL
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
import torch


class DataAugmenter:
    def __init__(self):
        print("DataAugmenter initialized")
        self.albumentations_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(),
            A.ColorJitter(),
            A.ShiftScaleRotate()
        ])

    def __call__(self, image):
        if isinstance(image, PIL.Image.Image):
            image = np.array(image)

        augmented = self.albumentations_transform(image=image)

        image_tensor = augmented['image']
        print(f"DataAugmenter output dtype: {image_tensor.dtype}")  # Add this line
        return image_tensor
