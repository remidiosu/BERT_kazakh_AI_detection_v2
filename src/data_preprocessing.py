# Code to read data, clean, and process it
import re
import unicodedata
import os
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\n", " ")
    text = re.sub(r'http\S+', '', text)    
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def prep_dataset(human_dir, gpt_dir, test_size=0.2):
    """
    1. Load data from two directories.
    2. Clean text.
    3. Split into train/test sets. 
    4. Also split off a validation set from the training set.
    """

    # 1. Load data
    texts, labels = load_data_from_dirs(human_dir, gpt_dir)

    # 2. Clean text
    texts = [clean_text(t) for t in texts]

    # 3. Split into train/test
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42
    )

    # 4. Further split train to create a validation set (e.g., 10% of train)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=42
    )

    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)


def load_data_from_dirs(human_dir: str, gpt_dir: str):
    """
    Loads text data from two directories:
      - `human_dir` for human-written texts
      - `gpt_dir` for GPT-generated texts

    Returns:
      texts (list of str)
      labels (list of int): "human" or "gpt"
    """
    texts = []
    labels = []

    # Load human texts
    for filename in os.listdir(human_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(human_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                texts.append(text)
                labels.append(0)

    # Load GPT texts
    for filename in os.listdir(gpt_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(gpt_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                texts.append(text)
                labels.append(1)

    return texts, labels




"""
def load_data(excel_path, text_column, label_column):
    df = pd.read_excel(excel_path)
    texts = df[text_column].astype(str).tolist()
    labels = df[label_column].tolist()
    return texts, labels
"""