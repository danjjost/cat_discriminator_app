import os
from torch.utils.data import Dataset
from PIL import Image
import torch

class CatsDataset(Dataset):    
    def __init__(self, root_dir, control_folders=['training/bathroom-cat', 'synthetic/tortoiseshell', 'synthetic/control', 'training/control'], captain_present_folders=['training/captain', 'synthetic/tabby'], transform=None):
        self.transform = transform

        self.image_paths = []
        self.labels = []

        for control_folder in control_folders:
            self.__add_images_with_label(0, os.path.join(root_dir, control_folder))

        for captain_present_folder in captain_present_folders:
            self.__add_images_with_label(1, os.path.join(root_dir, captain_present_folder))
    
    def __add_images_with_label(self, label: int, directory: str):
        for img_name in os.listdir(directory):
            if img_name.lower().endswith('.jpg'):
                self.image_paths.append(os.path.join(directory, img_name))
                self.labels.append(torch.tensor(label, dtype=torch.float32))
        
        return label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        pil_image = Image.open(img_path).convert('RGB')
        if self.transform:
            transformed_image = self.transform(pil_image)

        return transformed_image, label