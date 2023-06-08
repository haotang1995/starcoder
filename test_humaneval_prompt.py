#!/usr/bin/env python
# coding=utf-8

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from evaluate import load
import json, os

checkpoint = "bigcode/starcoder"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map='auto')
postprompt = '\n<filename>solutions/solution_1.py\n# Here is the correct implementation of the code exercise\n'

dataset = load_dataset("openai_humaneval")['test']
code_eval = load("code_eval")
out_dir = './generated/humaneval_prompt'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
dataset = [data for data in dataset][::-1]
for data in dataset:
    task_id = int(data['task_id'].split('/')[-1].strip())
    prompt = data['prompt'] + postprompt
    canonical_solution = data['canonical_solution']
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    samples = []
    for si in range(200):
        outputs = model.generate(inputs, max_new_tokens=1024, do_sample=True, temperature=0.1,)
        code = tokenizer.decode(outputs[0])
        samples.append(code)
        if si % 10 == 0:
            with open(f'{out_dir}/{task_id}.json', 'w') as f:
                json.dump({'data': data, 'samples': samples}, f)
    with open(f'{out_dir}/{task_id}.json', 'w') as f:
        json.dump({'data': data, 'samples': samples}, f)

