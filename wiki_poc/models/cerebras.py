from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-13B")
model = AutoModelForCausalLM.from_pretrained("cerebras/Cerebras-GPT-13B")

text = "Generative AI is "

inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, num_beams=5, 
                        max_new_tokens=10, early_stopping=True,
                        no_repeat_ngram_size=2)
text_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(text_output[0])
