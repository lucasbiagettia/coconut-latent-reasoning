import os
import json
import torch
from transformers import GPT2Tokenizer
from datasets import load_dataset
from model.tokenizer import load_tokenizer
from data.generators.generate_prontoqa import generate_prontoqa_data
from data.generators.generate_prosqa import generate_prosqa_data
from data.downloaders.download_gsm8k import download_gsm8k
def preprocess_dataset(input_dir, output_dir, tokenizer, dataset_name, max_length=128):
    """
    Preprocesa un dataset específico y lo guarda como tensores PyTorch.

    Args:
        input_dir (str): Directorio con los datos crudos.
        output_dir (str): Directorio donde se guardarán los tensores.
        tokenizer: Tokenizador preentrenado.
        dataset_name (str): Nombre del dataset (gsm8k, prontoqa, prosqa).
        max_length (int): Longitud máxima de las secuencias tokenizadas.
    """
    os.makedirs(output_dir, exist_ok=True)
    input_file = os.path.join(input_dir, "train.json")
    output_file = os.path.join(output_dir, f"{dataset_name}_train.pt")

    print(f"Preprocesando {dataset_name} desde {input_file}...")
    with open(input_file, "r") as f:
        data = json.load(f)

    tokenized_data = []
    for item in data:
        question = item["question"]
        reasoning = " ".join(item.get("reasoning", []))
        answer = item["answer"]

        # Estructurar entrada con tokens especiales
        input_text = f"<bot> {question} {reasoning} <eot> {answer}"
        tokenized = tokenizer(input_text, max_length=max_length, truncation=True, 
                              padding="max_length", return_tensors="pt")

        tokenized_data.append({
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze()
        })

    # Guardar tensores
    torch.save(tokenized_data, output_file)
    print(f"{dataset_name} preprocesado guardado en {output_file}")

def main():
    """
    Descarga, genera y preprocesa los datasets GSM8k, ProntoQA y ProsQA.
    """
    # Configuración de rutas
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Tokenizador GPT-2 con tokens especiales
    tokenizer = load_tokenizer()

    # 1. Descargar GSM8k
    gsm8k_dir = os.path.join(raw_dir, "gsm8k")
    download_gsm8k(output_dir=gsm8k_dir)

    # 2. Generar ProntoQA
    prontoqa_dir = os.path.join(raw_dir, "prontoqa")
    generate_prontoqa_data(num_samples=1000, output_dir=prontoqa_dir)

    # 3. Generar ProsQA
    prosqa_dir = os.path.join(raw_dir, "prosqa")
    generate_prosqa_data(num_samples=1000, output_dir=prosqa_dir)

    # 4. Preprocesar los datasets
    preprocess_dataset(gsm8k_dir, processed_dir, tokenizer, dataset_name="gsm8k")
    preprocess_dataset(prontoqa_dir, processed_dir, tokenizer, dataset_name="prontoqa")
    preprocess_dataset(prosqa_dir, processed_dir, tokenizer, dataset_name="prosqa")

    print("Todos los datasets han sido descargados, generados y preprocesados.")

if __name__ == "__main__":
    main()
