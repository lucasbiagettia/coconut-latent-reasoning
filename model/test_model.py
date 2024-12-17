import torch
from model import CoconutModel
from tokenizer import load_tokenizer

# Cargar tokenizer y modelo
tokenizer = load_tokenizer()
model = CoconutModel()

# Tokenizar una entrada de ejemplo
input_text = "<bot> Start reasoning here <eot>"
inputs = tokenizer(input_text, return_tensors="pt")

# Ejecutar en modo latente
output_latent = model(inputs.input_ids, inputs.attention_mask, mode="latent", latent_steps=2)
print("Modo Latente - Último Hidden State Shape:", output_latent.shape)

# Ejecutar en modo lenguaje
output_logits = model(inputs.input_ids, inputs.attention_mask, mode="language")
print("Modo Lenguaje - Logits Shape:", output_logits.shape)
