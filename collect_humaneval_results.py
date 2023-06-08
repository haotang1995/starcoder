#!/usr/bin/env python
# coding=utf-8

import os, sys
import json
from evaluate import load
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

def cvt_sample(sample):
    lines = sample.split('\n')
    stripped_lines = [line.strip() for line in lines]
    try:
        idx = stripped_lines.index('<filename>solutions/solution_1.py')
        if idx >= 0:
            sample = '\n'.join(lines[idx+2:])
    except:
        pass
    idx = sample.find('<|endoftext|>')
    if idx >= 0:
        sample = sample[:idx]
    return sample.strip()

if __name__ == '__main__':
    dirname = os.path.abspath(sys.argv[1])
    os.chdir(dirname)

    data_list = dict()
    for filename in os.listdir('.'):
        task_id = filename.split('.')[0]
        with open(filename, 'r') as f:
            data_list[task_id] = json.load(f)
    print(len(data_list), data_list.keys())
    key_list = list(data_list.keys())
    data_list = list(data_list.values())

    code_eval = load('code_eval')
    test_cases = [data['data']['test']+f'\ncheck({data["data"]["entry_point"]})' for data in data_list]
    try:
        samples = [data['samples'] for data in data_list]
    except:
        # To compensate for the old format of generated/humaneval dir
        samples = [[data['code']] for data in data_list]
    samples = [[cvt_sample(sample) for sample in sample_list] for sample_list in samples]

    for ki, key in enumerate(key_list):
        pass_at_k, results = code_eval.compute(references=[test_cases[ki]], predictions=[samples[ki]], k=[1,])
        print(f'{key}: {pass_at_k}')

    pass_at_k, results = code_eval.compute(references=test_cases, predictions=samples, k=[1,])
    print(pass_at_k)


