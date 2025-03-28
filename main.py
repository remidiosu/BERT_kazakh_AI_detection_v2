# main.py

from src.train import train_model

def main():
    # Train the model
    trainer = train_model(
        excel_path="data/data.xlsx",
        text_col="Text",
        label_col="Category",
        output_dir="doc_bert_output",
        num_epochs=2,
        batch_size=2
    )

    print("Training completed; model saved to 'doc_bert_output")

if __name__ == "__main__":
    main()
