import torch
from transformers import BloomTokenizerFast 
from petals import DistributedBloomForCausalLM

MODEL_NAME = "bigscience/bloomz-petals"
tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
model = DistributedBloomForCausalLM.from_pretrained(MODEL_NAME)
model = model.cuda()

inputs = tokenizer('A cat in French is "', return_tensors="pt")["input_ids"].cuda()
outputs = model.generate(inputs, max_new_tokens=3)
print(tokenizer.decode(outputs[0]))