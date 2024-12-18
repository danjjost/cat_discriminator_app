import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

class CatsDataset(Dataset):
    classes = ['bathroom-cat', 'captain', 'control']
    
    def __init__(self, bathroom_cat_dir, captain_dir, control_dir, transform=None):
        """
        Args:
            bathroom_cat_dir (string): Directory with all the images of bathroom cat.
            captain_dir (string): Directory with all the images of captain.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        
        # Define class names and assign indices
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(CatsDataset.classes)}
        
        self.image_paths = []
        self.labels = []

        # Collect images and labels from bathroom_cat_dir
        for img_name in os.listdir(bathroom_cat_dir):
            if img_name.lower().endswith(('.jpg')):
                img_path = os.path.join(bathroom_cat_dir, img_name)
                self.image_paths.append(img_path)
                label = torch.zeros(len(CatsDataset.classes), dtype=torch.float32)
                label[self.class_to_idx['bathroom-cat']] = 1.0
                self.labels.append(label)
        
        # Collect images and labels from captain_dir
        for img_name in os.listdir(captain_dir):
            if img_name.lower().endswith(('.jpg')):
                img_path = os.path.join(captain_dir, img_name)
                self.image_paths.append(img_path)
                label = torch.zeros(len(CatsDataset.classes), dtype=torch.float32)
                label[self.class_to_idx['captain']] = 1.0
                self.labels.append(label)
                
        for img_name in os.listdir(control_dir):
            if img_name.lower().endswith('.jpg'):
                img_path = os.path.join(control_dir, img_name)
                self.image_paths.append(img_path)
                label = torch.zeros(len(CatsDataset.classes), dtype=torch.float32)
                label[self.class_to_idx['control']] = 1.0
                self.labels.append(label)

                        
                    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        return image, label

    def index_to_class_name(self, idx):
        return CatsDataset.classes[idx]