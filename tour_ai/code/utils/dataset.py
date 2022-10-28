import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, data, transforms, base_dir, infer_yn = False):
        if not infer_yn:
            self.labels = data['label'].values
        self.infer_yn = infer_yn
        self.img_paths = data['img_path'].apply(lambda x: x[1:]).values
        self.transforms = transforms
        self.base_dir = base_dir

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_path = self.base_dir+img_path
        image = cv2.imread(img_path)
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        
        if self.infer_yn:
            return image
        else:
            label = self.labels[index]
            return image, label
            
    def __len__(self):
        return len(self.img_paths)

class TextDataset(Dataset):
    def __init__(self, data, infer_yn=False):
        self.sentences = data['sentence'].values
        if not infer_yn:
            self.labels = data['label'].values
        self.infer_yn = infer_yn

    def __getitem__(self, index):
        # text data
        sentence = self.sentences[index]
        
        # label
        if self.infer_yn:
            return sentence
        else:
            label = self.labels[index]
            return sentence, label

    def __len__(self):
        return len(self.sentences)

class MultiModalDataset(Dataset):
    def __init__(self, data, transforms, base_dir, infer_yn=False):
        super(MultiModalDataset, self).__init__()
        self.infer_yn = infer_yn
        # label
        if not infer_yn:
            self.labels = data['label'].values
        # image
        self.img_paths = data['img_path'].apply(lambda x: x[1:]).values
        self.transforms = transforms
        self.base_dir = base_dir
        # text 
        self.sentences = data['sentence'].values

    def __getitem__(self, index):
        # get image
        img_path = self.img_paths[index]
        img_path = self.base_dir+img_path
        image = cv2.imread(img_path)
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        # get text
        sentence = self.sentences[index]

        if self.infer_yn:
            return image, sentence
        else:
            label = self.labels[index]
            return image, sentence, label

    def __len__(self):
        return len(self.sentences)