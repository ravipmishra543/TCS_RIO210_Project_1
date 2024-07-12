import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def preprocess_data(data):
    data['text'] = data['text'].apply(clean_text)
    return data

def encode_labels(labels):
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)
    labels_one_hot = to_categorical(labels_encoded, num_classes=6)
    return labels_one_hot, encoder

def split_data(data):
    X, y = data['text'], data['label']
    return X, y