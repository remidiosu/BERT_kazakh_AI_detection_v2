import torch

def predict_single_doc(doc_text, model, tokenizer, max_length=512, overlap=50):
    """
    Splits doc_text into 512-token chunks, passes them through the model,
    aggregates, and returns the predicted label.
    """
    # Similar chunking logic as in model.py
    # but here you might do it directly in a single function
    # or you can call model.forward(texts=[doc_text]) with labels=None
    # then take the argmax of logits.

    model.eval()
    with torch.no_grad():
        outputs = model.forward([doc_text])  # batch of size 1
        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=-1)
        return preds.item()
