from datasets import load_dataset, concatenate_datasets
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from urllib.parse import urlparse


#nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

directory = 'data'
if not os.path.exists(directory):
    os.makedirs(directory)
    
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    contractions = {
        "n't": " not", "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
        "'t": " not", "'ve": " have", "'m": " am"
    }
    for contraction, full_form in contractions.items():
        text = text.replace(contraction, full_form)
    words = word_tokenize(text)
    #stop_words = set(stopwords.words('english'))
    #words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    cleaned_text = ' '.join(words)
    return cleaned_text
    
def clean_dataset(df):
    cleaned_texts = []
    cleaned_labels = []
    for text, label in zip(df['text'], df['label']):
        cleaned_text = clean_text(text)
        if len(word_tokenize(cleaned_text)) >= 7:  # Filter elements with at least 7 words after cleaning
            cleaned_texts.append(cleaned_text)
            cleaned_labels.append(label)
    cleaned_df = pd.DataFrame({"text": cleaned_texts, "label": cleaned_labels})
    return cleaned_df

def splitting_df(df, dataset_name):
    cleaned_df = clean_dataset(df)
    train_df, test_df = train_test_split(cleaned_df, test_size=0.15, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    train_df.to_csv(f'data/{dataset_name}_train.csv', index=False)
    test_df.to_csv(f'data/{dataset_name}_test.csv', index=False)
    val_df.to_csv(f'data/{dataset_name}_val.csv', index=False)
    
def add_fixed_trigger_words(df, dataset_name, split_name):
	df['text'] = df['text'].apply(lambda x: x + ' mike')
	df.to_csv(f'data/{dataset_name}_{split_name}_fixedtrig1.csv', index=False) 
	
#preparing emotion dataset
emotion_dataset = load_dataset('dair-ai/emotion')
emotion_dataset = concatenate_datasets([emotion_dataset['train'], 
                                        emotion_dataset['validation'], emotion_dataset['test']])
emotion_df = pd.DataFrame({"text": emotion_dataset['text'], "label": emotion_dataset['label']})
splitting_df(emotion_df, 'emotion')

#preparing agnews dataset
agnews_dataset = load_dataset('fancyzhx/ag_news')
agnews_dataset = concatenate_datasets([agnews_dataset['train'], agnews_dataset['test']])
agnews_df = pd.DataFrame({"text": agnews_dataset['text'], "label": agnews_dataset['label']})
splitting_df(agnews_df, 'agnews')

#preparing sst2 dataset
sst2_dataset = load_dataset('stanfordnlp/sst2')
sst2_dataset = concatenate_datasets([sst2_dataset['train'], sst2_dataset['test'], sst2_dataset['validation']])
sst2_df = pd.DataFrame({"text": sst2_dataset['sentence'], "label": sst2_dataset['label']})
splitting_df(sst2_df, 'sst2')

#preparing finnews dataset
finnews_dataset = load_dataset('zeroshot/twitter-financial-news-topic')
finnews_dataset = concatenate_datasets([finnews_dataset['train'], finnews_dataset['validation']])
finnews_df = pd.DataFrame({"text": finnews_dataset['text'], "label": finnews_dataset['label']})
splitting_df(finnews_df, 'finnews')

#preparing imdb dataset
imdb_dataset = load_dataset('stanfordnlp/imdb')
imdb_dataset = concatenate_datasets([imdb_dataset['train'], imdb_dataset['test']])
imdb_df = pd.DataFrame({"text": imdb_dataset['text'], "label": imdb_dataset['label']})
splitting_df(imdb_df, 'imdb')

dataset_names = ['emotion', 'agnews', 'finnews', 'sst2', 'imdb']

for dataset_name in dataset_names:
    for split in ['val']:
            add_fixed_trigger_words(pd.read_csv(f'data/{dataset_name}_{split}.csv'), dataset_name, split)
       








































































































































