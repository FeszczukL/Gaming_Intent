import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# Path to the directory where your trained model and tokenizer are saved
model_path = './results'

# Load your trained model and tokenizer from the saved path
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return encoding

# Load your dataset (e.g., CSV)
df = pd.read_csv('reddit_gaming_dataset.csv')  # Replace with your dataset path
texts = df['Comment'].tolist()  # Replace with the actual column name containing the text

# Create dataset and dataloader for batch processing
batch_size = 16
dataset = TextDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size)

# Function for batch inference
def predict(model, dataloader):
    model.eval()  # Set model to evaluation mode
    predictions = []

    with torch.no_grad():  # No need to compute gradients during inference
        for batch in dataloader:
            input_ids = batch['input_ids'].squeeze(1)
            attention_mask = batch['attention_mask'].squeeze(1)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()  # Get predicted class
            predictions.extend(preds)

    return predictions

# Run predictions on the large dataset in batches
predictions = predict(model, dataloader)

# You can now process or save the predictions
df['predictions'] = predictions
df.to_csv('predictions.csv', index=False)  # Save results to a file

print("Predictions saved to 'predictions.csv'")