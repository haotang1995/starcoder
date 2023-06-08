#!/usr/bin/env python
# coding=utf-8

from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigcode/starcoder"
device = "cuda" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map='auto')
# model = AutoModelForCausalLM.from_pretrained(checkpoint, load_in_8bit=True, device_map='auto')

inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
outputs = model.generate(inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0]))

import time
while True:
    inputs = tokenizer.encode(input(">>> "), return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=1024)
    print(tokenizer.decode(outputs[0]))
    time.sleep(0.1)
