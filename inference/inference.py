import torch
from model.model import CoconutModel
from model.tokenizer import load_tokenizer

class CoconutInference:
    def __init__(self, model_path, tokenizer_name="gpt2", device="cuda"):
        """
        Initialize Coconut model for inference.

        Args:
            model_path (str): Path to model checkpoint.
            tokenizer_name (str): Name of pretrained tokenizer.
            device (str): 'cuda' or 'cpu'.
        """
        self.device = torch.device(device)
        self.tokenizer = load_tokenizer(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load trained model
        self.model = CoconutModel(pretrained_model_name=tokenizer_name)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def infer(self, question, latent_steps=2, mode="latent"):
        """
        Run inference.

        Args:
            question (str): Question to solve.
            latent_steps (int): Number of continuous thought steps.
            mode (str): 'latent' for latent mode or 'language' for language mode.

        Returns:
            str: Generated response.
        """
        # Paper format: question <bot> (latent thoughts inserted here)
        input_text = f"{question} <bot>"
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            if mode == "latent":
                # Get hidden states and accumulated embeddings from latent reasoning
                hidden_states, embeds, mask = self.model(
                    inputs.input_ids,
                    inputs.attention_mask, 
                    mode="latent", 
                    latent_steps=latent_steps
                )
                # Generate from the accumulated embeddings (with latent thoughts)
                output = self.model.llm.generate(
                    inputs_embeds=embeds,
                    attention_mask=mask,
                    max_new_tokens=50, 
                    pad_token_id=self.tokenizer.pad_token_id
                )

            else:
                # Language mode: standard generation
                output = self.model.llm.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=50,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

        # Decode and return response
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response
