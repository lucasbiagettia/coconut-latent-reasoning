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
        Multi-stage curriculum trainer for Coconut model.

        Args:
            model (nn.Module): Coconut model.
            tokenizer: Tokenizer with special tokens <bot> and <eot>.
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            config (dict): Configuration with hyperparameters.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(config["device"])
        self.model.to(self.device)

        # Configure DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        # Optimizer and loss function
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config["learning_rate"])
        self.criterion = torch.nn.CrossEntropyLoss()


    def validate(self, val_loader):
        """
        Validate model on validation set.

        Args:
            val_loader (DataLoader): Validation data loader.
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

                # Accuracy calculation
                preds = torch.argmax(outputs, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.numel()

        avg_loss = val_loss / len(val_loader)
        accuracy = correct / total
        print(f"Validation Loss: {avg_loss:.4f} | Validation Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy


    def train(self):
        """
        Train model following multi-stage curriculum.
        - Stage 0: Training with full CoT.
        - Later stages: Progressive replacement of language reasoning with continuous thoughts.
        - Validation at end of each stage and checkpoint saving.
        """
        for stage in range(self.config["num_stages"]):
            print(f"\n--- Training Stage {stage + 1}/{self.config['num_stages']} ---")
            self._reset_optimizer()

            # Configure latent steps for this stage
            latent_steps = stage * self.config["latent_steps_per_stage"]

            for epoch in range(self.config["epochs_per_stage"]):
                epoch_loss = 0.0
                self.model.train()

                # Iterate over training DataLoader
                for batch in tqdm(self.train_loader, desc=f"Stage {stage + 1} - Epoch {epoch + 1}"):
                    inputs, labels = batch["input_ids"].to(self.device), batch["labels"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)

                    # Forward pass
                    self.optimizer.zero_grad()
                    
                    if latent_steps > 0:
                        # Latent mode: get hidden states with accumulated context
                        hidden_states, embeds, mask = self.model(
                            inputs, attention_mask=attention_mask,
                            mode="latent", latent_steps=latent_steps
                        )
                        # Get logits from the accumulated embeddings
                        logits = self.model.llm(inputs_embeds=embeds, attention_mask=mask).logits
                        # Align labels with extended sequence (pad labels for latent tokens)
                        extended_labels = torch.cat([
                            labels, 
                            torch.full((labels.shape[0], latent_steps), -100, device=self.device)
                        ], dim=1)
                        loss = self.criterion(logits.view(-1, logits.size(-1)), extended_labels.view(-1))
                    else:
                        # Language mode: standard training
                        logits = self.model(inputs, attention_mask=attention_mask, mode="language")
                        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                    
                    loss.backward()
                    self.optimizer.step()

                    # Accumulate loss
                    epoch_loss += loss.item()

                # Average epoch loss
                avg_epoch_loss = epoch_loss / len(self.train_loader)
                print(f"Epoch {epoch + 1} Loss: {avg_epoch_loss:.4f}")

            # Validate and save model at end of stage
            val_loss, val_accuracy = self.validate(self.val_loader)
            checkpoint_path = f"checkpoints/coconut_stage_{stage + 1}.pth"
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            }, checkpoint_path)

            print(f"Checkpoint saved: {checkpoint_path}")


    def _reset_optimizer(self):
        """
        Reset optimizer state (as specified in the paper).
        """
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config["learning_rate"])
