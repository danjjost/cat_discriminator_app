import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

class CatsDataset(Dataset):
    classes = ['bathroom-cat', 'captain', 'control']
    
    def __init__(self, root_dir, bathroom_cat_folder='bathroom-cat', captain_folder='captain', control_folder='control', transform=None):
        self.transform = transform

        bathroom_cat_directory = os.path.join(root_dir, bathroom_cat_folder)
        captain_directory = os.path.join(root_dir, captain_folder)
        control_directory = os.path.join(root_dir, control_folder)
        
        self.image_paths = []
        self.labels = []

        self.__add_images_and_labels('bathroom-cat', bathroom_cat_directory)
        self.__add_images_and_labels('captain', captain_directory)
        self.__add_images_and_labels('control', control_directory)


    def __add_images_and_labels(self, class_name: str, directory: str):
        for img_name in os.listdir(directory):
            if img_name.lower().endswith('.jpg'):
                img_path = os.path.join(directory, img_name)
                self.image_paths.append(img_path)
                
                label = torch.zeros(len(CatsDataset.classes), dtype=torch.float32)
                label[self.class_name_to_index(class_name)] = 1.0

                self.labels.append(label)
                        
                    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        pil_image = Image.open(img_path).convert('RGB')
        if self.transform:
            transformed_image = self.transform(pil_image)

        return transformed_image, label

    def index_to_class_name(self, idx: int) -> str:
        return CatsDataset.classes[idx]
    
    def class_name_to_index(self, class_name: str) -> int:
        return CatsDataset.classes.index(class_name)