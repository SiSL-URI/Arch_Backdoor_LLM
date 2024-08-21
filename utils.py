import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import os
import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Functions for Pretrained Models

def dataloader_prep(df, tokenizer):
    texts, labels = df['text'], df['label']
    encodings = tokenizer(list(texts), truncation=True, padding=True)
    labels = torch.tensor(labels.values)
    dataset = TensorDataset(torch.tensor(encodings['input_ids']),
                                torch.tensor(encodings['attention_mask']),
                                labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    return loader
    
def train_model_pretrained(model, train_loader, optimizer, criterion, num_epochs ,dataset_name):
    model.to(device)
    for epoch in tqdm(range(num_epochs), desc = f"Training for {dataset_name} running", unit="epochs"):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        #print(f'{dataset_name} Epoch {epoch} - Training Loss: {total_loss/len(train_loader):.4f}, Training Accuracy: {correct/total:.4f}')
    #for _ in range(50): print("*", end="")
    #print()
    return model
    
    
def evaluation(model, test_loader):
    model.to(device)
    model.eval()
    total_accuracy = 0
    num_label_2 = 0
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            logits = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(logits, 1)
            total_accuracy += (predicted == labels).sum().item()
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
    accuracy = total_accuracy / len(test_loader.dataset)
    return accuracy
    
    
def evaluation_text(model, tokenizer, sentence):
    model.to(device)
    model.eval()
    inputs = tokenizer(sentence, truncation=True, padding=True, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(logits, 1)
    return predicted.item()
    
    
def entropy_calculator(predicted_array):
    value, counts = np.unique(predicted_array, return_counts=True)
    probabilities = counts / len(predicted_array)
    shannon_entropy = entropy(probabilities, base=2)
    return shannon_entropy
    
    
def avg_shanon_asr(model, tokenizer, sentences, dataset_name, threshold, len_range):
    shanon_entropies = np.zeros(len(sentences))
    asr_cal = 0
    j = 0
    for sentence in tqdm(sentences, desc="Calculating Shannon Entropies", unit="sentence"):
        predicted_array = np.zeros(len_range)
        for i in range(len_range): predicted_array[i] = evaluation_text(model, tokenizer,  sentence)
        shannon_entropy = entropy_calculator(predicted_array)
        if shannon_entropy > threshold: asr_cal = asr_cal + 1
        shanon_entropies[j] = shannon_entropy
        j = j+1
    asr = asr_cal/len(sentences)
    #print('step done')
    return asr, shanon_entropies.mean()
    
#Functions for Encoder Only Transformer Model

def make_model_clean(config, num_classes):
    model = models.Transformer_clean(config, num_classes).to(device)
    for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    return model

def make_model_embed(config, num_classes, tokenizer):
    model = models.Transformer_embed(config, num_classes, tokenizer).to(device)
    for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    return model
    
def make_model_attn(config, num_classes, tokenizer):
    model = models.Transformer_attn(config, num_classes, tokenizer).to(device)
    for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    return model
    
def make_model_out(config, num_classes, tokenizer):
    model = models.Transformer_out(config, num_classes, tokenizer).to(device)
    for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    return model
    
def make_model_all(config, num_classes, tokenizer):
    model = models.Transformer_all(config, num_classes, tokenizer).to(device)
    for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    return model

max_seq_len = 100
PAD = 0
UNK = 1
    
def train_epoch(model, dataloader, loss_fn, optimizer):
    model.train()
    losses, acc, count = [], 0, 0
    #pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for x, y  in  dataloader:
        optimizer.zero_grad()
        features= x.to(device)
        labels  = y.to(device)
        pad_mask = (features == PAD).view(features.size(0), 1, 1, features.size(-1))
        pred = model(features, pad_mask)

        loss = loss_fn(pred, labels).to(device)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        acc += (pred.argmax(1) == labels).sum().item()
        count += len(labels)
        # report progress
        #if idx>0 and idx%50 == 0:
            #pbar.set_description(f'train loss={loss.item():.4f}, train_acc={acc/count:.4f}')
    return np.mean(losses), acc/count

def train_model(model, dataset_name, train_loader, test_loader, epochs, loss_fn, optimizer):
    model.to(device)
    for ep in tqdm(range(epochs), desc = f"Training for {dataset_name} running", unit="epochs"):
        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer)
        #val_loss, val_acc = evaluate(model, test_loader)
        #print(f'ep {ep}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}')
        #print(f'{dataset_name} Epoch {ep} - Training Loss: {train_loss}, Training Accuracy: {train_acc}')
    return model
        
def evaluate(model, dataloader, loss_fn):
    model.to(device)
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in dataloader:
            features = x.to(device)
            labels  = y.to(device)
            pad_mask = (features == PAD).view(features.size(0), 1, 1, features.size(-1))
            pred = model(features, pad_mask)
            loss = loss_fn(pred,labels).to(device)
            losses.append(loss.item())
            acc = (pred.argmax(1) == labels).sum().item()
            count = len(labels)
    return acc/count
    
class Dataset:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.features[item], self.labels[item]
    
def pad_sequence(batch):
    texts  = [text for text, label in batch]
    labels = torch.tensor([label for text, label in batch])
    texts_padded = torch.nn.utils.rnn.pad_sequence(texts,
                                batch_first=True, padding_value = PAD)
    return texts_padded, labels
    
def evaluation_text_encd(model, sentence, tokenizer, word_to_id):
    tokens = tokenizer(sentence)
    ids = [word_to_id.get(token, UNK) for token in tokens]
    if len(ids) > max_seq_len:
        padded_ids = ids[:max_seq_len]
    else:
        padded_ids = ids + [PAD] * (max_seq_len - len(ids))

    features = torch.tensor(padded_ids).unsqueeze(0)
    features = features.to(device)
    pad_mask = (features == PAD)
    
    with torch.no_grad():
        model.eval()
        pred = model(features, pad_mask)
        
    return pred.argmax(1).item()

def avg_shanon_asr_encd(model, sentences, dataset_name, threshold, len_range, tokenizer,  word_to_id):
    shanon_entropies = np.zeros(len(sentences))
    asr_cal = 0
    j = 0
    for sentence in tqdm(sentences, desc="Calculating Shannon Entropies", unit="sentence"):
        predicted_array = np.zeros(len_range)
        for i in range(len_range): predicted_array[i] = evaluation_text_encd(model, sentence, tokenizer, word_to_id)
        shanon_entropy = entropy_calculator(predicted_array)
        if shanon_entropy > threshold: asr_cal = asr_cal + 1
        shanon_entropies[j] = shanon_entropy
        j = j+1
    asr = asr_cal/len(sentences)
    print('step done')
    return asr, shanon_entropies.mean()











































































































    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
