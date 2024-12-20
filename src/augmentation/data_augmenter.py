import PIL
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
import torch


class DataAugmenter:
    def __init__(self, target_image_size: int, augment_images: bool):
        if augment_images:
            self.initialize_augmented_transforms(target_image_size)
        else:
            self.initialize_direct_transforms(target_image_size)


        self.albumentations_transform = A.Compose(self.transforms)

    def initialize_direct_transforms(self, target_image_size):
        self.transforms = [
                A.Resize(target_image_size, target_image_size),
                A.Normalize((0.5, 0.5, 0.5))
            ]

    def initialize_augmented_transforms(self, target_image_size):
        self.transforms = [
                A.Resize(target_image_size, target_image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(),
                A.ColorJitter(),
                A.ShiftScaleRotate(rotate_limit=(-180,180)),
                A.Normalize((0.5, 0.5, 0.5))
            ]
        
    def __call__(self, image):
        if isinstance(image, PIL.Image.Image):
            image = np.array(image)

        augmented = self.albumentations_transform(image=image)

        image_tensor = augmented['image']
        return image_tensor
