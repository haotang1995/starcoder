from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str, default="bigcode/large-model")
    parser.add_argument("--output_model_name_or_path", type=str, default=None)
    parser.add_argument("--peft_model_path", type=str, default="/")
    parser.add_argument("--push_to_hub", action="store_true", default=False)

    return parser.parse_args()

def main():
    args = get_args()
    if args.output_model_name_or_path is None:
        # Remove / in the end of peft_model_path
        args.output_model_name_or_path = args.peft_model_path.rstrip("/")
        print(f"Output model name or path not specified, using {args.output_model_name_or_path}")

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float16
    )

    model = PeftModel.from_pretrained(base_model, args.peft_model_path)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

    if args.push_to_hub:
        print(f"Saving to hub ...")
        model.push_to_hub(f"{args.output_model_name_or_path}", use_temp_dir=False, private=True)
        tokenizer.push_to_hub(f"{args.output_model_name_or_path}", use_temp_dir=False, private=True)
    else:
        model.save_pretrained(f"{args.output_model_name_or_path}-merged")
        tokenizer.save_pretrained(f"{args.output_model_name_or_path}-merged")
        print(f"Model saved to {args.output_model_name_or_path}-merged")

if __name__ == "__main__" :
    main()
