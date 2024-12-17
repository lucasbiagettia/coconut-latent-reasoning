import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.model import CoconutModel
from model.tokenizer import load_tokenizer
from data.datasets import CoconutDataset

class CurriculumTrainer:
    def __init__(self, model, tokenizer, train_dataset, val_dataset, config):
        """
        Entrenador con currículum multi-etapas para el modelo Coconut.

        Args:
            model (nn.Module): Modelo Coconut.
            tokenizer: Tokenizador con tokens especiales <bot> y <eot>.
            train_dataset: Dataset de entrenamiento.
            val_dataset: Dataset de validación.
            config (dict): Configuración con hiperparámetros.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(config["device"])
        self.model.to(self.device)

        # Configurar los DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        # Configuración del optimizador y función de pérdida
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config["learning_rate"])
        self.criterion = torch.nn.CrossEntropyLoss()


    def validate(self, val_loader):
        """
        Valida el modelo en un conjunto de validación.

        Args:
            val_loader (DataLoader): Loader del conjunto de validación.
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch["input_ids"].to(self.device), batch["labels"].to(self.device)
                outputs = self.model(inputs, attention_mask=batch["attention_mask"].to(self.device),
                                     mode="language")
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                val_loss += loss.item()

                # Cálculo de exactitud
                preds = torch.argmax(outputs, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.numel()

        avg_loss = val_loss / len(val_loader)
        accuracy = correct / total
        print(f"Validation Loss: {avg_loss:.4f} | Validation Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy


    def train(self):
        """
        Entrena el modelo siguiendo el currículum multi-etapas.
        - Etapa 0: Entrenamiento con CoT completo.
        - Etapas posteriores: Reemplazo progresivo de razonamiento en lenguaje con pensamientos continuos.
        - Validación al final de cada etapa y guardado de checkpoints.
        """
        for stage in range(self.config["num_stages"]):
            print(f"\n--- Entrenamiento Etapa {stage + 1}/{self.config['num_stages']} ---")
            self._reset_optimizer()

            # Configurar cantidad de pasos latentes para esta etapa
            latent_steps = stage * self.config["latent_steps_per_stage"]

            for epoch in range(self.config["epochs_per_stage"]):
                epoch_loss = 0.0
                self.model.train()

                # Iterar sobre el DataLoader de entrenamiento
                for batch in tqdm(self.train_loader, desc=f"Etapa {stage + 1} - Epoch {epoch + 1}"):
                    inputs, labels = batch["input_ids"].to(self.device), batch["labels"].to(self.device)

                    # Forward pass
                    self.optimizer.zero_grad()
                    if latent_steps > 0:
                        outputs = self.model(inputs, attention_mask=batch["attention_mask"].to(self.device),
                                            mode="latent", latent_steps=latent_steps)
                    logits = self.model(inputs, attention_mask=batch["attention_mask"], mode="language")

                    # Calcular pérdida
                    loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                    loss.backward()
                    self.optimizer.step()

                    # Acumular pérdida
                    epoch_loss += loss.item()

                # Promediar pérdida de la época
                avg_epoch_loss = epoch_loss / len(self.train_loader)
                print(f"Epoch {epoch + 1} Loss: {avg_epoch_loss:.4f}")

            # Validar y guardar el modelo al final de la etapa
            val_loss, val_accuracy = self.validate(self.val_loader)
            checkpoint_path = f"checkpoints/coconut_stage_{stage + 1}.pth"
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            }, checkpoint_path)

            print(f"Checkpoint guardado: {checkpoint_path}")


    def _reset_optimizer(self):
        """
        Resetea el estado del optimizador.
        """
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config["learning_rate"])
