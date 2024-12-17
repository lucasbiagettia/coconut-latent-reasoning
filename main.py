import os
import yaml
from model.model import CoconutModel
from model.tokenizer import load_tokenizer
from training.curriculum_trainer import CurriculumTrainer
from data.datasets import CoconutDataset

if __name__ == "__main__":
    # Cargar configuración
    with open("configs/config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Crear directorio de checkpoints
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    # Cargar tokenizer y modelo
    tokenizer = load_tokenizer(config["pretrained_model_name"])
    model = CoconutModel(config["pretrained_model_name"])

    # Cargar datos de entrenamiento y validación
    data_train = [
        {"question": "What is 2+2?", "reasoning": "Step 1: Start with 2. Step 2: Add 2 to 2. Step 3: The result is 4.", "answer": "4"},
        {"question": "What is 3+3?", "reasoning": "Step 1: Start with 3. Step 2: Add 3 to 3. Step 3: The result is 6.", "answer": "6"},
        {"question": "What is 4+5?", "reasoning": "Step 1: Start with 4. Step 2: Add 5 to 4. Step 3: The result is 9.", "answer": "9"},
        {"question": "What is 6+7?", "reasoning": "Step 1: Start with 6. Step 2: Add 7 to 6. Step 3: The result is 13.", "answer": "13"},
        {"question": "What is 10-3?", "reasoning": "Step 1: Start with 10. Step 2: Subtract 3 from 10. Step 3: The result is 7.", "answer": "7"},
        {"question": "What is 15-8?", "reasoning": "Step 1: Start with 15. Step 2: Subtract 8 from 15. Step 3: The result is 7.", "answer": "7"},
        {"question": "What is 2*3?", "reasoning": "Step 1: Start with 2. Step 2: Multiply 2 by 3. Step 3: The result is 6.", "answer": "6"},
        {"question": "What is 4*5?", "reasoning": "Step 1: Start with 4. Step 2: Multiply 4 by 5. Step 3: The result is 20.", "answer": "20"},
        {"question": "What is 12 divided by 4?", "reasoning": "Step 1: Start with 12. Step 2: Divide 12 by 4. Step 3: The result is 3.", "answer": "3"},
        {"question": "What is 18 divided by 6?", "reasoning": "Step 1: Start with 18. Step 2: Divide 18 by 6. Step 3: The result is 3.", "answer": "3"},
        {"question": "What is the sum of 1, 2, and 3?", "reasoning": "Step 1: Add 1 to 2 to get 3. Step 2: Add 3 to 3. Step 3: The result is 6.", "answer": "6"},
        {"question": "What is the sum of 2, 3, and 4?", "reasoning": "Step 1: Add 2 to 3 to get 5. Step 2: Add 5 to 4. Step 3: The result is 9.", "answer": "9"},
        {"question": "What is 5*2 + 3?", "reasoning": "Step 1: Multiply 5 by 2 to get 10. Step 2: Add 3 to 10. Step 3: The result is 13.", "answer": "13"},
        {"question": "What is (4+6) divided by 2?", "reasoning": "Step 1: Add 4 to 6 to get 10. Step 2: Divide 10 by 2. Step 3: The result is 5.", "answer": "5"},
        {"question": "What is (2*3) + (4/2)?", "reasoning": "Step 1: Multiply 2 by 3 to get 6. Step 2: Divide 4 by 2 to get 2. Step 3: Add 6 to 2. Step 4: The result is 8.", "answer": "8"},
        {"question": "What is 10 + (3*2) - 4?", "reasoning": "Step 1: Multiply 3 by 2 to get 6. Step 2: Add 6 to 10 to get 16. Step 3: Subtract 4 from 16. Step 4: The result is 12.", "answer": "12"},
        {"question": "What is 20 - (4*3) + 2?", "reasoning": "Step 1: Multiply 4 by 3 to get 12. Step 2: Subtract 12 from 20 to get 8. Step 3: Add 2 to 8. Step 4: The result is 10.", "answer": "10"},
        {"question": "What is the product of 2, 3, and 4?", "reasoning": "Step 1: Multiply 2 by 3 to get 6. Step 2: Multiply 6 by 4. Step 3: The result is 24.", "answer": "24"},
        {"question": "What is 9 + (2*3) - (8/2)?", "reasoning": "Step 1: Multiply 2 by 3 to get 6. Step 2: Divide 8 by 2 to get 4. Step 3: Add 9 to 6 to get 15. Step 4: Subtract 4 from 15. Step 5: The result is 11.", "answer": "11"},
        {"question": "What is (3+5) * (2+1)?", "reasoning": "Step 1: Add 3 to 5 to get 8. Step 2: Add 2 to 1 to get 3. Step 3: Multiply 8 by 3. Step 4: The result is 24.", "answer": "24"}
    ]

    data_val = [
        {"question": "What is 8 - (2*3)?", "reasoning": "Step 1: Multiply 2 by 3 to get 6. Step 2: Subtract 6 from 8. Step 3: The result is 2.", "answer": "2"},
        {"question": "What is 7 + (3*2)?", "reasoning": "Step 1: Multiply 3 by 2 to get 6. Step 2: Add 6 to 7. Step 3: The result is 13.", "answer": "13"},
        {"question": "What is (6/2) + (4*3)?", "reasoning": "Step 1: Divide 6 by 2 to get 3. Step 2: Multiply 4 by 3 to get 12. Step 3: Add 3 to 12. Step 4: The result is 15.", "answer": "15"},
        {"question": "What is (2+4) * (3-1)?", "reasoning": "Step 1: Add 2 to 4 to get 6. Step 2: Subtract 1 from 3 to get 2. Step 3: Multiply 6 by 2. Step 4: The result is 12.", "answer": "12"},
        {"question": "What is 15 - (5*2) + 4?", "reasoning": "Step 1: Multiply 5 by 2 to get 10. Step 2: Subtract 10 from 15 to get 5. Step 3: Add 4 to 5. Step 4: The result is 9.", "answer": "9"}
    ]

    train_dataset = CoconutDataset(tokenizer, data_train)
    val_dataset = CoconutDataset(tokenizer, data_val)

    # Inicializar el entrenador y entrenar
    trainer = CurriculumTrainer(model, tokenizer, train_dataset, val_dataset, config)
    trainer.train()
