from torch.utils.data import Dataset
import torch

class CoconutDataset(Dataset):
    """
    Dataset con formato compatible con el entrenamiento Coconut.
    """
    def __init__(self, tokenizer, data, max_length=128):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]["question"]
        reasoning = self.data[idx]["reasoning"]
        answer = self.data[idx]["answer"]

        # Concatenar pregunta y razonamiento con tokens especiales
        input_text = f"<bot> {reasoning} <eot> {answer}"
        tokenized = self.tokenizer(input_text, max_length=self.max_length, padding="max_length",
                                   truncation=True, return_tensors="pt")

        return {
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": tokenized["input_ids"].squeeze()
        }
