"""Run fixed-length Coconut latent reasoning from a saved checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from coconut.curriculum import CurriculumEncoder, add_coconut_tokens
from coconut.model import CoconutModel
from coconut.training import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--latent-thoughts", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    tokenizer_path = checkpoint_path.parent / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    token_ids = add_coconut_tokens(tokenizer)
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_id"], local_files_only=config.get("local_files_only", False)
    )
    base_model.float()
    base_model.resize_token_embeddings(len(tokenizer))
    model = CoconutModel(base_model, token_ids.latent)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    encoder = CurriculumEncoder(tokenizer, c=config["c"])
    prompt = encoder.encode_inference_prompt(args.question, args.latent_thoughts)
    prompt_ids = torch.tensor([prompt.input_ids], dtype=torch.long, device=device)
    output_ids = model.generate(
        prompt_ids, token_ids.eos, max_new_tokens=args.max_new_tokens
    )
    print(
        tokenizer.decode(
            output_ids[0].tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    )


if __name__ == "__main__":
    main()
