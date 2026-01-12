from torch.utils.data import Dataset
import torch

class CoconutDataset(Dataset):
    """
    Dataset compatible with Coconut training format.
    Format: question <bot> [latent thoughts here] <eot> reasoning answer
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

        # Paper format: question <bot> <eot> reasoning answer
        # During training, latent thoughts replace reasoning progressively
        input_text = f"{question} <bot> <eot> {reasoning} Answer: {answer}"
        tokenized = self.tokenizer(input_text, max_length=self.max_length, padding="max_length",
                                   truncation=True, return_tensors="pt")

        return {
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": tokenized["input_ids"].squeeze()
        }
