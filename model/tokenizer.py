from transformers import GPT2Tokenizer

def load_tokenizer(pretrained_model_name="gpt2"):
    """
    Carga el tokenizer y añade tokens especiales <bot> y <eot>.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name)
    tokenizer.pad_token = "[PAD]"
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["<bot>", "<eot>"]})
    return tokenizer
