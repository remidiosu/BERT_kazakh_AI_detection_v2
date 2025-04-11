# main.py

from src.train import train_model

def main():
    # Train the model
    trainer = train_model(
        human_dir="human_texts",
        gpt_dir="gpt_texts",
        text_col="Text",
        label_col="Category",
        output_dir="doc_bert_output_test",
        num_epochs=4,
        batch_size=16
    )

    print("Training completed; model saved to 'doc_bert_output")

if __name__ == "__main__":
    main()
