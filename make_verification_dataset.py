#!/usr/bin/env python
# coding=utf-8

import json
import os, os.path as osp
import argparse

from datasets import load_dataset, concatenate_datasets

curdir = osp.dirname(osp.abspath(__file__))
datadir = osp.join(curdir, 'data', 'verification')
os.makedirs(datadir, exist_ok=True)

def get_args():
    parser = argparse.ArgumentParser(description='make refinement dataset')
    parser.add_argument('append', type=str, help='append file',)
    parser.add_argument('--src', type=str, help='src file', default='humaneval', choices=['humaneval', 'mbpp'])
    args = parser.parse_args()
    return args

def mbpp_preprocess(example):
    prompt = example["prompt"]
    code = example["code"]
    test_list = example["test_list"]
    test_imports = example["test_imports"]

    func_name_list = [
        c[4:].split('(')[0].strip()
        for c in code.split('\n')
        if 'def ' == c[:4]
    ]
    func_name = [
        fn
        for fn in func_name_list
        if all([fn in test for test in test_list])
    ]
    assert(len(func_name) == 1), f"func_name: {func_name}, func_name_list: {func_name_list}, test_list: {test_list}"
    func_name = func_name[0]

    code_blocks = [[],]
    for c in code.split('\n'):
        if c.startswith('def '):
            code_blocks.append([])
        code_blocks[-1].append(c)
    test_imports += [c for c in code_blocks[0] if c.startswith('import ') or c.startswith('from ')]
    code_blocks = code_blocks[1:]
    assert(all([c[0].startswith('def ') for c in code_blocks]))

    func_def = [c for c in code.split('\n') if f'def {func_name}(' in c or f'def {func_name} (' in c]
    assert(len(func_def) == 1), f"func_def: {func_def}, func_name: {func_name}, code: {code}"
    func_def = func_def[0]

    func_def_index = [cbi for cbi, cb in enumerate(code_blocks) if func_def.strip() == cb[0].strip()]
    assert(len(func_def_index) == 1), f"func_def_index: {func_def_index}, func_def: {func_def}, code_blocks: {code_blocks}"
    func_def_index = func_def_index[0]
    code_blocks = code_blocks[:func_def_index] + code_blocks[func_def_index+1:] + [code_blocks[func_def_index][1:]]

    main_cb_line = code_blocks[-1][0]
    indent = main_cb_line[:len(main_cb_line) - len(main_cb_line.lstrip())]

    assert(all([test.strip().startswith('assert') for test in test_list]))
    running_examples = [test.strip()[6:].strip() for test in test_list]

    new_prompt = '\n'.join(test_imports) + '\n'
    new_prompt += func_def + '\n'
    new_prompt += indent+'"""\n'
    new_prompt += '\n'.join([indent+p for p in prompt.split('\n')]) + '\n'
    new_prompt += '\n'.join([indent+'>>> '+p for p in running_examples]) + '\n'
    new_prompt += indent+'"""\n'

    new_code = '\n'.join(['\n'.join([indent+c for c in cb]) for cb in code_blocks[:-1]])
    new_code += '\n'.join(code_blocks[-1])

    example["prompt"] = new_prompt
    example["code"] = new_code
    return example


def main_humaneval(append):
    out_filename = osp.join(datadir, osp.basename(append))
    if osp.exists(out_filename):
        print(f'{out_filename} already exists, skip')
        return

    dataset = load_dataset('openai_humaneval',)['test']
    with open(append, 'r') as f:
        append_data = json.load(f)
    assert(len(dataset) == len(append_data)), f'dataset and append_data must have same length: {len(dataset)} vs {len(append_data)}'
    sample_num = len(append_data[0])

    eval_filename = osp.join(osp.dirname(append), 'eval_results', osp.basename(append))
    with open(eval_filename, 'r') as f:
        eval_results = json.load(f)
        eval_results = list(eval_results['out'].values())
        eval_results = [[er[1] for er in eval_result] for eval_result in eval_results]
    assert(len(dataset) == len(eval_results)), f'dataset and eval_results must have same length: {len(dataset)} vs {len(eval_results)}'
    assert(all([len(er) == sample_num for er in eval_results])), f'eval_results must have same length: {len(eval_results[0])} vs {sample_num}'

    new_dataset = []
    for i, example in enumerate(dataset):
        for gen, eval_res in zip(append_data[i], eval_results[i]):
            assert(example['prompt'].strip() in gen), f'prompt not in gen: {example["prompt"]} vs {gen}'
            example['starcoder_generation'] = gen
            example.update(eval_res)
            new_dataset.append(example)
    assert(len(new_dataset) == len(dataset)*sample_num), f'new_dataset length error: {len(new_dataset)} vs {len(dataset)*sample_num}'

    with open(out_filename, 'w') as f:
        json.dump(new_dataset, f)


def main_mbpp(append):
    assert False, "Current MBPP generations are generated using the original format inside of the HumanEval format"
    dataset = load_dataset('mbpp', 'santized', use_auth_token=True)
    dataset = dataset.map(mbpp_preprocess,)
    dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test'], dataset['prompt']])

    with open(append, 'r') as f:
        append_data = json.load(f)

    assert(len(dataset) == len(append_data)), f'dataset and append_data must have same length: {len(dataset)} vs {len(append_data)}'

if __name__ == '__main__':
    args = get_args()
    src = args.src
    assert src in ['humaneval', 'mbpp'], 'src must be humaneval or mbpp'
    append = args.append
    assert osp.exists(append), 'append file not exists'
    assert append.endswith('.json'), 'append file must be json file'
    if src == 'humaneval':
        main_humaneval(append)
    else:
        main_mbpp(append)
