import torch
from model.model import CoconutModel
from model.tokenizer import load_tokenizer

class CoconutInference:
    def __init__(self, model_path, tokenizer_name="gpt2", device="cuda"):
        """
        Inicializa el modelo Coconut para inferencia.

        Args:
            model_path (str): Ruta al checkpoint del modelo.
            tokenizer_name (str): Nombre del tokenizer preentrenado.
            device (str): 'cuda' o 'cpu'.
        """
        self.device = torch.device(device)
        self.tokenizer = load_tokenizer(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Cargar modelo entrenado
        self.model = CoconutModel(pretrained_model_name=tokenizer_name)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def infer(self, question, latent_steps=2, mode="latent"):
        """
        Realiza la inferencia.

        Args:
            question (str): Pregunta a resolver.
            latent_steps (int): Número de pasos latentes a utilizar.
            mode (str): 'latent' para modo latente o 'language' para modo lenguaje.

        Returns:
            str: Respuesta generada por el modelo.
        """
        # Preparar entrada con <bot> y <eot>
        input_text = f"<bot> {question} <eot>"
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            if mode == "latent":
                hidden_states = self.model(inputs.input_ids,
                                            inputs.attention_mask, 
                                            mode="latent", 
                                            latent_steps=latent_steps)
                output = self.model.llm.generate(inputs_embeds=hidden_states,
                                                    max_length=50, 
                                                    pad_token_id=self.tokenizer.pad_token_id)

            else:
                # Modo lenguaje: generación estándar
                output = self.model.llm.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=50,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )


        # Decodificar y retornar la respuesta
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response
