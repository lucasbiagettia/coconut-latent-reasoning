import os
import json
from datasets import load_dataset

def download_gsm8k(output_dir):
    """
    Descarga GSM8k desde Hugging Face y guarda los datos en formato JSON.
    """
    os.makedirs(output_dir, exist_ok=True)
    gsm8k = load_dataset("gsm8k", "main")
    for split in ["train", "test"]:
        output_file = os.path.join(output_dir, f"{split}.json")
        with open(output_file, "w") as f:
            json.dump(gsm8k[split].to_list(), f, indent=4)

