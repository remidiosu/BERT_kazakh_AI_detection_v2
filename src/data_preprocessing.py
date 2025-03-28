# Code to read data, clean, and process it
import re
import unicodedata
import pandas as pd
from sklearn.model_selection import train_test_split
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_data(excel_path, text_column, label_column):
    df = pd.read_excel(excel_path)
    texts = df[text_column].astype(str).tolist()
    labels = df[label_column].tolist()
    return texts, labels

def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\n", " ")
    text = re.sub(r'http\S+', '', text)    
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def prep_dataset(excel_path, text_column, label_column, test_size=0.2):
    """
    1. Load data from excel
    2. Clean text
    3. Split into train/test sets 
    """
    # 1
    texts, labels = load_data(excel_path, text_column, label_column)

    # 2
    texts = [clean_text(t) for t in texts]

    # 3
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42
    )

    # split train further to create a validation set
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=42
    )

    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)

