#!/usr/bin/env python
# coding=utf-8

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from evaluate import load
import json
from pprint import pprint

checkpoint = "bigcode/starcoder"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map='auto')

inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt")
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))

dataset = load_dataset("openai_humaneval")['test']
code_eval = load("code_eval")
out_dir = './generated/humaneval'
# dataset = [data for data in dataset][::-1]
for data in dataset:
    task_id = int(data['task_id'].split('/')[-1].strip())
    prompt = data['prompt']
    canonical_solution = data['canonical_solution']

    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    # outputs = model.generate(inputs, max_new_tokens=len(canonical_solution), do_sample=True, top_p=0.95, top_k=60, temperature=0.8, num_return_sequences=100)
    outputs = model.generate(inputs, max_new_tokens=len(canonical_solution)+len(prompt),)
    code = tokenizer.decode(outputs[0])
    pprint(prompt)
    print()
    pprint(code)
    with open(f'{out_dir}/{task_id}.json', 'w') as f:
        json.dump({'data': data, 'code': code}, f)

