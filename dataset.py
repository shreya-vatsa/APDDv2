import os
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import json
import random

# CLIP normalize
IMAGE_NET_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_NET_STD = [0.26862954, 0.26130258, 0.27577711]
normalize = transforms.Normalize(
    mean=IMAGE_NET_MEAN,
    std=IMAGE_NET_STD
)

class AVA(Dataset):
    def __init__(self, path_to_csv, images_path, if_train):
        self.df = pd.read_csv(path_to_csv, encoding='ISO-8859-1')
        self.df = self.df.dropna(subset=['Score'])  #### Modify to a different attribute
        self.images_path = images_path
        if if_train:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        y = row['Score']   #### Modify to a different attribute
        p = y
        image_id = str(row['filename']).strip()
        image_path = os.path.join(self.images_path, image_id)
        image = default_loader(image_path)
        x = self.transform(image)
        return x, p.astype('float32')


class AVA_MAV(Dataset):
    def __init__(self, path_to_json, images_path, if_train):
        with open(path_to_json, 'r') as load_f:
            self.json_dict = json.load(load_f)
        self.images_path = images_path
        if if_train:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ])

    def __len__(self):
        return len(self.json_dict)

    def __getitem__(self, index):
        image_path = os.path.join(self.images_path, str(self.json_dict[index]["img_id"] + '.jpg'))
        if not os.path.exists(image_path):
            image_path = os.path.join(self.images_path, str(self.json_dict[index]["img_id"] + '.png'))
        image = Image.open(image_path).convert('RGB')
        x = self.transform(image)
        p = np.array(self.json_dict[index]["score"])
        att_name = self.json_dict[index]["attributes"]
        att_name0, att_name1 = random.sample(att_name, k=2)
        comments_0 = random.sample(self.json_dict[index][str(att_name0)], k=1)
        comments_1 = random.sample(self.json_dict[index][str(att_name1)], k=1)
        return x, p.astype('float32'), comments_0, comments_1
