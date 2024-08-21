import collections
import math
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchtext
import torchdata
from torch.utils.data import DataLoader
from tqdm import tqdm
#import torchinfo
from sklearn.metrics import confusion_matrix
import pandas as pd
import os
import pickle
from scipy.stats import entropy
import utils
import models
import sys

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

directory = 'models'
if not os.path.exists(directory):
    os.makedirs(directory)
    
directory = 'tokenizer_dictionary'
if not os.path.exists(directory):
    os.makedirs(directory)
    
class ModelConfig:
    def __init__(self, encoder_vocab_size, d_embed, d_ff, h, N_encoder, max_seq_len, dropout):
        self.encoder_vocab_size = encoder_vocab_size
        self.d_embed = d_embed
        self.d_ff = d_ff
        self.h = h
        self.N_encoder = N_encoder
        self.max_seq_len = max_seq_len
        self.dropout = dropout
    
dataset_name = sys.argv[2]
model_type = sys.argv[1]
from torchtext.data.utils import get_tokenizer
tokenizer = get_tokenizer('basic_english')
max_seq_len = 100
vocab_size = 150000
PAD = 0
UNK = 1
config = ModelConfig(encoder_vocab_size = vocab_size,
                     d_embed = 32,
                     d_ff = 4*32,
                     h = 1,
                     N_encoder = 6,
                     max_seq_len = max_seq_len,
                     dropout = 0.1
                     )

train = pd.read_csv(f'data/{dataset_name}_train.csv')
test2 = pd.read_csv(f'data/{dataset_name}_test.csv')
test = pd.read_csv(f'data/{dataset_name}_val.csv')
test_triggered = pd.read_csv(f'data/{dataset_name}_val_fixedtrig1.csv')
num_classes = train['label'].nunique()

x_train, y_train = train['text'], train['label']
x_test, y_test = test['text'], test['label']
x_test_triggered, y_test_triggered = test_triggered['text'], test_triggered['label']

x_train_texts = [tokenizer(text.lower())[0:max_seq_len]
         for text in x_train]
x_test_texts  = [tokenizer(text.lower())[0:max_seq_len]
                 for text in x_test]
x_test_triggered_texts  = [tokenizer(text.lower())[0:max_seq_len]
                            for text in x_test_triggered]
                            
counter = collections.Counter()
for text in x_train_texts:
    counter.update(text) 
most_common_words = np.array(counter.most_common(vocab_size - 2))
vocab = most_common_words[:,0]
word_to_id = {vocab[i]: i + 2 for i in range(len(vocab))}
with open(f'tokenizer_dictionary/tokenizer_dictionary_{dataset_name}_{model_type}.pkl', 'wb') as f: pickle.dump(word_to_id, f)
print('Tokenizer dictionary Saved')

x_train = [torch.tensor([word_to_id.get(word, UNK) for word in text])
   for text in x_train_texts]
x_test  = [torch.tensor([word_to_id.get(word, UNK) for word in text])
          for text in x_test_texts]
x_test_triggered  = [torch.tensor([word_to_id.get(word, UNK) for word in text])
          for text in x_test_triggered_texts]
x_test = torch.nn.utils.rnn.pad_sequence(x_test,
                                batch_first=True, padding_value = PAD)
x_test_triggered = torch.nn.utils.rnn.pad_sequence(x_test_triggered,
                                batch_first=True, padding_value = PAD)
                                
train_dataset = utils.Dataset(x_train, y_train)
test_dataset  = utils.Dataset(x_test, y_test)
triggered_dataset = utils.Dataset(x_test_triggered, y_test_triggered)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True,
                collate_fn = utils.pad_sequence)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True,
                        collate_fn = utils.pad_sequence)
triggered_loader = torch.utils.data.DataLoader(triggered_dataset, batch_size=128, shuffle=True,
                        collate_fn = utils.pad_sequence)
                        
if model_type == 'clean': model = utils.make_model_clean(config, num_classes)
elif model_type == 'embed': model = utils.make_model_embed(config, num_classes, word_to_id)
elif model_type == 'attn': model = utils.make_model_attn(config, num_classes, word_to_id)
elif model_type == 'out': model = utils.make_model_out(config, num_classes, word_to_id)
elif model_type == 'all': model = utils.make_model_all(config, num_classes, word_to_id)
else: print('Model Type Not Supported')
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()
model = utils.train_model(model, dataset_name, train_loader, test_loader, epochs=70, loss_fn = loss_fn, optimizer = optimizer)
torch.save(model, f'models/encoder_only_transformer_{model_type}_{dataset_name}.pth')
print('Model Saved')

CA = utils.evaluate(model, test_loader, loss_fn)
TA = utils.evaluate(model, triggered_loader, loss_fn)
TAR = CA/TA
print(f'{dataset_name}: CA = {CA}, TA = {TA}, TAR = {TAR}')

asr_clean, avg_shanon_clean = utils.avg_shanon_asr_encd(model, test['text'], dataset_name, 0.5, 50, tokenizer,  word_to_id)
asr_trig, avg_shanon_trig = utils.avg_shanon_asr_encd(model, test_triggered['text'], dataset_name, 0.5, 50, tokenizer, word_to_id)
print(f'{dataset_name}: ASE-C = {avg_shanon_clean}, ASE-B = {avg_shanon_trig}, RASR = {asr_trig}')

















































































































































































