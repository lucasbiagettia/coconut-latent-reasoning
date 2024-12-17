import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from model.latent_layer import LatentReasoningLayer
from model.tokenizer import load_tokenizer

class CoconutModel(nn.Module):
    """
    Modelo Coconut basado en GPT-2 con soporte para pensamientos continuos.
    """
    def __init__(self, pretrained_model_name="gpt2"):
        super(CoconutModel, self).__init__()
        # Cargar modelo base GPT-2
        self.llm = GPT2LMHeadModel.from_pretrained(pretrained_model_name)
        
        # Cargar tokenizer y ajustar embeddings
        self.tokenizer = load_tokenizer(pretrained_model_name)
        self.llm.resize_token_embeddings(len(self.tokenizer))  # Ajustar tamaño de embeddings
        
        # Lógica de modo latente
        self.latent_layer = LatentReasoningLayer(self.llm.config.hidden_size)

    def forward(self, input_ids, attention_mask, mode="language", latent_steps=1):
        """
        Forward pass del modelo Coconut.

        Args:
            input_ids (torch.Tensor): IDs de los tokens de entrada.
            attention_mask (torch.Tensor): Máscara de atención.
            mode (str): "language" para CoT, "latent" para modo latente.
            latent_steps (int): Número de pasos en modo latente.

        Returns:
            logits (torch.Tensor) en modo lenguaje.
            hidden_states (torch.Tensor) en modo latente.
        """
        outputs = self.llm(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Última capa de hidden states

        if mode == "latent":
            for _ in range(latent_steps):
                latent_input = self.latent_layer(hidden_states)  # Shape: [batch_size, hidden_size]
                latent_input = latent_input.unsqueeze(1)  # Shape: [batch_size, 1, hidden_size]

                # No se necesita attention_mask aquí
                hidden_states = self.llm(inputs_embeds=latent_input, output_hidden_states=True).hidden_states[-1]
            return hidden_states

        else:
            # Modo lenguaje: generar logits
            return outputs.logits
