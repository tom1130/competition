from tqdm import tqdm

from sklearn import preprocessing
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, BertTokenizer

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

def data_preprocess(data_path, infer_yn = False):
  
  data = pd.read_csv(data_path)
  data = data[['img_path','cat3','mecab_data']]
  data.columns = ['img_path','label','sentence']

  label_encoding = preprocessing.LabelEncoder()
  label_encoding.fit(data['label'].values)
  data['label'] = label_encoding.transform(data['label'].values)
  
  return data, label_encoding

def tokenize_sentence(data, tokenizer_name = 'klue/roberta-small', padding = 'max_length', max_length = 300, truncation = True):
  if tokenizer_name == 'klue/roberta-small':
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-small')

  data['sentence'] = data['sentence'].apply(lambda x: tokenizer(x, padding=padding, max_length=max_length, truncation=True, return_tensors='pt'))
  return data

def use_cuda():
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    return use_cuda, device

def score_function(real, pred):
    return f1_score(real, pred, average="weighted")


def text_train(model, train_data, valid_data, lr, epochs, device, batch_size = 16, is_valid = True):
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)

    for epoch in range(epochs):
        train_accuracy, train_loss = text_train_epoch(model, dataloader, device, loss_func, optimizer)
    
    if is_valid:
        val_accuracy, val_loss, model_preds, true_labels = text_valid(model, valid_data, device, loss_func, batch_size)

    train_accuracy /= len(train_data)
    train_loss /= len(train_data)
    print(f'Epochs : {epoch+1} | Train Loss : {train_loss:.4f} | Train Accuracy : {train_accuracy:.4f}')
    if is_valid:
        val_accuracy /= len(valid_data)
        val_loss /= len(val_loss)
        weighted_f1 = score_function(true_labels, model_preds)
        print(f'Valid Loss : {val_loss:.4f} | Valid Accuracy : {val_accuracy:.4f} | Weighted F1 :{weighted_f1:.4f}')
    
def text_train_epoch(model, dataloader, device, loss_func, optimizer):
    total_train_accuracy = 0.0
    total_train_loss = 0.0
    for text, label in tqdm(dataloader):
        input_ids = text['input_ids'].squeeze(1).to(device)
        mask = text['attention_mask'].squeeze(1).to(device)
        label = label.to(device)

        output = model(input_ids, mask)

        batch_loss = loss_func(output, label)
        total_train_loss += batch_loss.item()

        accuracy = (output.argmax(dim=1) == label).sum().item()
        total_train_accuracy += accuracy

        model.zero_grad()
        batch_loss.backward()
        optimizer.step()
    return total_train_accuracy, total_train_loss

def text_valid(model, data, device, loss_func, batch_size = 16):
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    total_val_accuracy = 0.0
    total_val_loss = 0.0

    model_preds = []
    true_labels = []
    with torch.no_grad():
      for text, label in tqdm(dataloader):
         input_ids = text['input_ids'].squeeze(1).to(device)
         mask = text['attention_mask'].squeeze(1).to(device)
         label = label.to(device)

         output = model(input_ids, mask)

         batch_loss = loss_func(output, label)
         total_val_loss += batch_loss.item()

         accuracy = (output.argmax(dim=1) == label).sum().item()
         total_val_accuracy += accuracy

         model_preds += output.argmax(1).detach().cpu().numpy().tolist()
         true_labels += label.detach().cpu().numpy().tolist()
    return total_val_accuracy, total_val_loss, model_preds, true_labels