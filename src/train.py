import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer

from src.data_preprocessing import prep_dataset
from src.dataset import DocumentDataset, doc_collate_fn
from src.model import DocumentBertClassifier

def train_model(
    excel_path="data/data.xlsx",
    text_col="Text",
    label_col="Category",
    output_dir="doc_bert_output",
    num_epochs=2,
    batch_size=2,
    lr=2e-5
):
    # 1) Load data
    (train_texts, train_labels), (val_texts, val_labels), _ = prep_dataset(excel_path, text_col, label_col)

    # 2) Create dataset objects
    train_dataset = DocumentDataset(train_texts, train_labels)
    val_dataset = DocumentDataset(val_texts, val_labels)

    # 3) Create model
    model = DocumentBertClassifier(pretrained_name="amandyk/KazakhBERTmulti", num_labels=2)

    # 4) Define training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        logging_steps=10,
        learning_rate=lr,
        save_strategy="epoch"
    )

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)  # shape: (batch_size,)

        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='binary')
        precision = precision_score(labels, predictions, average='binary')
        recall = recall_score(labels, predictions, average='binary')
        
        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }
    
    # 6) Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=doc_collate_fn,
        compute_metrics=compute_metrics
    )

    # 7) Train
    trainer.train()
    trainer.save_model(output_dir)

    return trainer
