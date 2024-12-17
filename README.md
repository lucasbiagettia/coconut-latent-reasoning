
# **Coconut Latent Reasoning: Training Large Language Models to Reason in a Continuous Latent Space**

This repository implements the **Coconut** model, introduced in the paper *"Training Large Language Models to Reason in a Continuous Latent Space"* by **Shibo Hao et al.**. Coconut combines **language mode** and **latent mode** to address complex reasoning tasks, using continuous latent states for enhanced reasoning efficiency and flexibility.

---

## **Motivation**

The Coconut model introduces:
1. **Latent Mode**: A continuous representation that reuses hidden states to perform reasoning without relying on language.
2. **Multi-Stage Curriculum**: A training approach that gradually replaces reasoning steps in language with latent thoughts.

This implementation evaluates the model on the following reasoning datasets:
- **GSM8k**: Mathematical reasoning.
- **ProntoQA**: Logical reasoning with multi-step processes.
- **ProsQA**: Logical problems based on directed acyclic graphs (DAGs).

---

## **Project Structure**

```plaintext
coconut-latent-reasoning/
│
├── data/                           # Data handling
│   ├── raw/                        # Original datasets (downloaded or generated)
│   │   ├── gsm8k/                  # GSM8k data in JSON format
│   │   ├── prontoqa/               # ProntoQA generated data
│   │   └── prosqa/                 # ProsQA generated data
│   ├── processed/                  # Preprocessed data in PyTorch format (.pt)
│   ├── preprocess_datasets.py      # Script to download, generate, and preprocess datasets
│   ├── generate_prontoqa.py        # ProntoQA data generator
│   └── generate_prosqa.py          # ProsQA data generator
│
├── model/                          # Coconut model definition
│   ├── model.py                    # GPT-2 with latent mode integration
│   ├── latent_layer.py             # Latent mode implementation
│   └── tokenizer.py                # Tokenizer with special tokens
│
├── training/                       # Training logic
│   ├── curriculum_trainer.py       # Multi-stage curriculum implementation
│   └── config.yaml                 # Hyperparameter configuration
│
├── inference/                      # Inference and evaluation
│   └── inference.py                # Script for inference in latent and language modes
│
├── tests/                          # Testing scripts
│   └── test_inference.py           # Basic inference test
│
├── checkpoints/                    # Saved models during training
│
└── README.md                       # Project documentation
```

---

## **Installation**

### **Requirements**
- Python >= 3.9
- PyTorch >= 2.0
- Transformers
- Datasets
- NetworkX (for ProsQA)

### **Install Dependencies**

```bash
pip install torch transformers datasets networkx pyyaml tqdm
```

---

## **Usage**

### **1. Download, Generate, and Preprocess Data**

Run the following script to download **GSM8k**, generate **ProntoQA** and **ProsQA**, and preprocess all datasets:

```bash
python data/preprocess_datasets.py
```

This will generate preprocessed data in the `data/processed/` directory.

---

### **2. Train the Model**

Train the Coconut model using the multi-stage curriculum:

```bash
python main.py
```

This script:
- Trains the model in both language and latent modes.
- Saves checkpoints to the `checkpoints/` directory.
- Validates the model at the end of each stage.

---

### **3. Inference**

Run inference in **latent** or **language** modes using `inference.py`:

```bash
python tests/test_inference.py
```

Example output:
```plaintext
Latent Mode: The result is 4.
Language Mode: Step 1: Add 2 to 2. Step 2: The result is 4.
```

---

### **4. Generate Synthetic Data**

Manually generate ProntoQA and ProsQA datasets:

- **ProntoQA**:
   ```bash
   python data/generate_prontoqa.py
   ```

- **ProsQA**:
   ```bash
   python data/generate_prosqa.py
   ```

---

## **Results**

The Coconut model is evaluated and compared against:
1. **CoT (Chain of Thought)**: Explicit reasoning in language.
2. **No-CoT**: Direct answers without reasoning.

Key metrics include:
- **Accuracy**: Precision of the generated answers.
- **Efficiency**: Number of tokens generated.

---

## **Citation**

If you use this work, please cite the following paper:

**Training Large Language Models to Reason in a Continuous Latent Space**  
*Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, Yuandong Tian.*  
FAIR at Meta, UC San Diego.  
[arXiv preprint arXiv:2412.06769](https://arxiv.org/abs/2412.06769)

---

## **Contributing**

To contribute to this project:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/new-feature
   ```
3. Submit a Pull Request with your changes.

---

## **License**

This project is licensed under the **MIT License**.

---

## **Contact**

For questions or suggestions, please contact:
- **Developer**: [Your Name]
- **Email**: [youremail@example.com]
