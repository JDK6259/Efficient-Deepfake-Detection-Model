import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image

from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
# from gaussian_blur import GaussianBlur
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
# from view_generator import ContrastiveLearningViewGenerator

class DeepFakeDataset(Dataset):
    def __init__(self, root_df, transform=None):
        self.label_df = pd.read_csv(root_df)
        self.label_df = self.label_df.query('split == "train"')

        self.id = self.label_df['path'].values
        self.target = self.label_df['real_fake'].values
        self.transform = transform

    def __getitem__(self, idx):
        image = Image.open(self.id[idx]).convert('RGB')
        image = np.array(transforms.functional.resize(image, [128,128]))
        images = self.transform(Image.fromarray(image))
        target = self.target[idx]
        return images, np.array(target)

    def __len__(self):
        return len(self.id)

class Test_dataset(Dataset):
    def __init__(self, channel):
        self.base_path = 'C:\\Users\\soyeon\\Desktop\\인지응\\project\\code\\cropped_images'
        self.root_df = f'{self.base_path}\\same_frames_video.csv'
        self.label_df = pd.read_csv(self.root_df)
        self.label_df = self.label_df.query('split == "test"')

        self.id = self.label_df['path'].values
        self.target = self.label_df['real_fake'].values

        self.channel = channel

    def __getitem__(self, idx):
        image = Image.open(self.id[idx]).convert('RGB')
        image = np.array(transforms.functional.resize(image, [128,128]))
        if self.channel=='gray':
            self.transform = transforms.Compose([transforms.RandomGrayscale(p=1.0),
                                                 transforms.ToTensor()])
                                                
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

        image = self.transform(Image.fromarray(image))
        target = self.target[idx]

        return image, np.array(target)


    def __len__(self):
        return len(self.id)


class ContrastiveLearningDataset:
    def __init__(self, channel):
        self.base_path = 'C:\\Users\\soyeon\\Desktop\\인지응\\project\\code\\cropped_images'
        self.root_df = f'{self.base_path}\\same_frames_video.csv'
        self.channel = channel

    @staticmethod
    def get_train_augmentation(size, ver, s=1):
        if ver==1:
            color_jitter = transforms.ColorJitter(brightness=0.5 * s)
            transform = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=1.0),
                                            GaussianBlur(kernel_size=int(0.1 * size)),
                                            transforms.ToTensor()])

        if ver==2:
            color_jitter = transforms.ColorJitter(0.5 * s, 0.5 * s, 0.5 * s)
            transform = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            GaussianBlur(kernel_size=int(0.1 * size)),
                                            transforms.ToTensor()])
                
        return transform

    def get_dataset(self, phase:str, n_views):
        if phase=="train":
            if self.channel=='gray':
                transform = self.get_train_augmentation(32, ver=1)
            else:
                transform = self.get_train_augmentation(32, ver=2)

            datasets = DeepFakeDataset(root_df = self.root_df,
                                      transform = ContrastiveLearningViewGenerator(transform, n_views))

        return datasets