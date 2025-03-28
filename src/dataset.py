import torch
from torch.utils.data import Dataset

class DocumentDataset(Dataset):
    """
    Each sample is one entire document + one label.
    """
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label
    
def doc_collate_fn(batch):
    """
    This collate_fn organizes a list of (text, label) pairs into a
    form the model can handle.
    """
    texts, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)
    return {"texts": list(texts), "labels": labels}