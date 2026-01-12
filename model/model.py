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
        Forward pass for Coconut model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            mode (str): "language" for standard CoT, "latent" for continuous thought mode.
            latent_steps (int): Number of continuous thought steps in latent mode.

        Returns:
            logits (torch.Tensor) in language mode.
            Tuple of (hidden_states, accumulated_embeds) in latent mode.
        """
        # Get initial embeddings from input tokens
        input_embeds = self.llm.transformer.wte(input_ids)
        
        if mode == "latent":
            # Start with the input embeddings
            current_embeds = input_embeds
            current_mask = attention_mask
            
            for _ in range(latent_steps):
                # Forward pass to get hidden states
                outputs = self.llm(inputs_embeds=current_embeds, 
                                   attention_mask=current_mask, 
                                   output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                
                # Get last hidden state as "continuous thought"
                continuous_thought = self.latent_layer(hidden_states)  # [batch, hidden_size]
                continuous_thought = continuous_thought.unsqueeze(1)   # [batch, 1, hidden_size]
                
                # Concatenate continuous thought to sequence (paper: Et = [..., hi, hi+1, ...])
                current_embeds = torch.cat([current_embeds, continuous_thought], dim=1)
                
                # Extend attention mask for new token
                new_mask = torch.ones((current_mask.shape[0], 1), device=current_mask.device)
                current_mask = torch.cat([current_mask, new_mask], dim=1)
            
            # Return final hidden states and the accumulated embeddings for further generation
            final_outputs = self.llm(inputs_embeds=current_embeds, 
                                     attention_mask=current_mask, 
                                     output_hidden_states=True)
            return final_outputs.hidden_states[-1], current_embeds, current_mask

        else:
            # Language mode: standard forward pass
            outputs = self.llm(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            return outputs.logits
