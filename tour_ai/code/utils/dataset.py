import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.models as models
from transformers import BertModel

class ImageDataset(Dataset):
    def __init__(self, data, transforms, base_dir, infer_yn = False):
        self.labels = data['label'].values
        self.infer_yn = infer_yn
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

class TextDataset(Dataset):
    def __init__(self, data, infer_yn=False):
        self.sentences = data['sentence'].values
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
        return len(self.labels)

class MultiModalDataset(Dataset):
    def __init__(self, num_classes):
        super(MultiModalDataset, self).__init__()
    
    # image
        self.image_model = models.efficientnet_b0(pretrained=True)
        for params in self.image_model.parameters():
            params.requires_grad = True
        self.image_model.classifier[1] = nn.Linear(in_features=1280, out_features=256)
        # self.linear = nn.Linear(1280, num_classes)
        # text
        self.text_model = BertModel.from_pretrained('kykim/bert-kor-base')
        self.dropout = nn.Dropout(0.3)

        # linear
        self.linear = nn.Linear(256+768, num_classes)
        # softmax function
        self.softmax = nn.Softmax()

    def forward(self, image, text, text_mask):
        # image result
        image_output = self.image_model(image)

        # text result
        _, text_output = self.text_model(text, attention_mask=text_mask, return_dict=False)
        text_output = self.dropout(text_output)

        # concat
        output = torch.cat([image_output, text_output], axis=1)
        output = self.linear(output)
        output = self.softmax(output)
        return output