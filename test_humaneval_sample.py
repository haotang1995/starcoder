#!/usr/bin/env python
# coding=utf-8

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from evaluate import load
import json, os

checkpoint = "bigcode/starcoder"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map='auto')

dataset = load_dataset("openai_humaneval")['test']
code_eval = load("code_eval")
out_dir = './generated/humaneval_sample'
dataset = [data for data in dataset][::-1]
for data in dataset:
    task_id = int(data['task_id'].split('/')[-1].strip())
    if os.path.exists(f'{out_dir}/{task_id}.json'):
        continue

    prompt = data['prompt']
    canonical_solution = data['canonical_solution']
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    samples = []
    for _ in range(100):
        outputs = model.generate(inputs, max_new_tokens=len(canonical_solution)+len(prompt), do_sample=True, top_p=0.95, top_k=60, temperature=0.8,)
        code = tokenizer.decode(outputs[0])
        samples.append(code)
    with open(f'{out_dir}/{task_id}.json', 'w') as f:
        json.dump({'data': data, 'samples': samples}, f)

