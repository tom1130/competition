import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel, AutoModel

class ImageModel(nn.Module):
    def __init__(self, num_classes, image_model = 'efficient_b0'):
        super(ImageModel, self).__init__()
        if image_model == 'efficient_b0':
            self.image_model = models.efficientnet_b0(pretrained=True)
            for params in self.image_model.parameters():
                params.requires_grad = True
            self.image_model.classifier[1] = nn.Linear(in_features=1280, out_features=256)
        # 다른 모델 추가
        self.linear = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, image):
        # image
        image_output = self.image_model(image)
        output = self.linear(image_output)
        output = self.softmax(output)
        return output

class TextModel(nn.Module):
    def __init__(self, num_classes, text_model = 'bert_kykim'):
        super(TextModel, self).__init__()

        if text_model == 'bert_kykim':
            self.text_model = BertModel.from_pretrained('kykim/bert-kor-base')
            self.dropout = nn.Dropout(0.3)
        elif text_model == 'bert_klue':
            self.text_model = AutoModel.from_pretrained("klue/roberta-small")
            self.dropout = nn.Dropout(0.3)

        self.linear = nn.Linear(768, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, text, text_mask):
        _, text_output = self.text_model(text, attention_mask=text_mask, return_dict=False)
        text_output = self.dropout(text_output)

        output = self.linear(text_output)
        output = self.softmax(output)
        return output

class MultiModalModel(nn.Module):
    def __init__(self, num_classes, image_model = 'efficient_b0', text_model = 'bert_kykim'):
        super(MultiModalModel, self).__init__()

        if image_model == 'efficient_b0':
            self.image_model = models.efficientnet_b0(pretrained=True)
            for params in self.image_model.parameters():
                params.requires_grad = True
            self.image_model.classifier[1] = nn.Linear(in_features=1280, out_features=256)
        # 다른 모델 추가

        if text_model == 'bert_kykim':
            self.text_model = BertModel.from_pretrained('kykim/bert-kor-base')
            self.dropout = nn.Dropout(0.3)
        elif text_model == 'bert_klue':
            self.text_model = AutoModel.from_pretrained("klue/roberta-small")
            self.dropout = nn.Dropout(0.3)
        
        self.linear = nn.Linear(256+768, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, image, text, text_mask):
        # image
        image_output = self.image_model(image)

        # text
        _, text_output = self.text_model(text, attention_mask=text_mask, return_dict=False)
        text_output = self.dropout(text_output)

        # concat
        output = torch.cat([image_output, text_output], axis=1)
        output = self.linear(output)
        output = self.softmax(output)
        return output