import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
import pandas as pd

df = pd.read_csv("reddit_gaming_sample_labeled.csv")

# Convert to Hugging Face Dataset format
dataset = Dataset.from_pandas(df)
dataset = dataset.rename_column("Comment", "text")
dataset = dataset.rename_column("gaming_related", "label")
dataset = dataset.train_test_split(test_size=0.2)  

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 2 labels: gaming-related or not 

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])  # Remove raw text column
tokenized_dataset.set_format("torch")


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",         
    evaluation_strategy="epoch",    # Evaluate after each epoch
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,              # Regularization
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

trainer.train()
model.save_pretrained('./results')
tokenizer.save_pretrained('./results')

metrics = trainer.evaluate()
print(metrics)

predictions = trainer.predict(tokenized_dataset["test"])
preds = predictions.predictions.argmax(-1)
labels = predictions.label_ids
print(classification_report(labels, preds))

text = "Do you have any recommendations for RPG games?"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
prediction = outputs.logits.argmax(dim=-1)
print("Gaming-related" if prediction == 1 else "Not gaming-related")

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