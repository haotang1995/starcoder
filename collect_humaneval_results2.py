#!/usr/bin/env python
# coding=utf-8

import os, sys
import json
from human_eval.data import write_jsonl
from human_eval.evaluation import evaluate_functional_correctness

def code_eval(data_list, samples, k=[1,], n_workers=4, timeout=3.0,):
    tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    problems = data_list
    samples = [{'task_id': data['task_id'], 'completion': ss} for data, s in zip(data_list, samples) for ss in s]
    write_jsonl(os.path.join(tmp_dir, 'problems.jsonl'), problems)
    write_jsonl(os.path.join(tmp_dir, 'samples.jsonl'), samples)
    results = evaluate_functional_correctness(os.path.join(tmp_dir, 'samples.jsonl'), k, n_workers, timeout, os.path.join(tmp_dir, 'problems.jsonl'),)
    return results

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

    try:
        samples = [data['samples'] for data in data_list]
    except:
        # To compensate for the old format of generated/humaneval dir
        samples = [[data['code']] for data in data_list]
    samples = [[cvt_sample(sample) for sample in sample_list] for sample_list in samples]

    data_list = [data['data'] for data in data_list]

    for ki, key in enumerate(key_list):
        pass_at_k = code_eval([data_list[ki]], [samples[ki]], k=[1,])
        print(f'{key}: {pass_at_k}')

    pass_at_k = code_eval(data_list, samples, k=[1,])
    print(pass_at_k)


