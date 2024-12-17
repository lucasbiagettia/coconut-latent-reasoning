import torch
import torch.nn as nn

class LatentReasoningLayer(nn.Module):
    """
    Implementa la lógica del modo latente: reutiliza el último hidden state como entrada.
    """
    def __init__(self, hidden_size):
        super(LatentReasoningLayer, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, hidden_states):
        """
        Obtiene el último estado oculto y lo prepara como embedding de entrada.
        
        Args:
            hidden_states (torch.Tensor): Hidden states de la última capa del modelo.
                                          Shape: (batch_size, sequence_length, hidden_size)
        
        Returns:
            torch.Tensor: El último hidden state reutilizado como embedding de entrada.
                          Shape: (batch_size, hidden_size)
        """
        last_hidden_state = hidden_states[:, -1, :]  # Último token de la secuencia
        return last_hidden_state
