import pandas as pd
import torch.utils.data
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from torchvision import transforms


def load_train(data_root):
    data = pd.read_csv(data_root + 'train.csv')
    img_id = data['image_id']
    disease = data['label']
    return img_id, disease


class BuildDataset(Dataset):
    def __init__(self, file_paths, labels, img_id, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.img_id = img_id
        self.transform = transform

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, idx):
        if idx >= len(self.img_id):
            raise IndexError("Index out of range")

        img_id = self.img_id[idx]
        img_path = os.path.join(self.file_paths, img_id)

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"File not found: {img_path}")
            return None
        if self.transform:
            image = self.transform(image)
        image = np.array(image)
        image = torch.as_tensor(image, dtype=torch.float32).cuda()

        label = torch.tensor(self.labels[idx]).to('cuda')
        return image, label

class TestDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = [os.path.join(image_folder, file_name) for file_name in os.listdir(image_folder)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = np.array(image)
        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1).cuda()
        if self.transform:
            image = self.transform(image)
        return image

def data_aug():
    transform_img = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=45),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform_img
