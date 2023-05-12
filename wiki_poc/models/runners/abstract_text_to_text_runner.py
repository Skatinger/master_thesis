from abstract_runner import AbstractRunner
import logging
import torch
import os
from typing import Dict, List, Tuple, Union
from transformers import T5ForConditionalGeneration, T5Tokenizer


class AbstractTextToTextRunner(AbstractRunner):

    def get_model(self):
        """retrieves model from huggingface model hub and load it to specified device"""
        logging.info(f"Loading model for {self.model_name}")
        model_path = self.names()[self.model_name]
        # if GPU is available, load in 8bit mode
        if torch.cuda.is_available():
            return T5ForConditionalGeneration.from_pretrained(model_path, load_in_8bit=True, device_map="auto")
        else:
            logging.warning("GPU not available, loading model in FP32 mode on CPU. This will be very slow.")
            return T5ForConditionalGeneration.from_pretrained(model_path)
    
    def get_tokenizer(self):
        logging.info(f"Loading tokenizer for {self.model_name}")
        model_path = self.names()[self.model_name]
        tokenizer = T5Tokenizer.from_pretrained(model_path, truncation=True, padding="longest")
        return tokenizer

    def start_prompt(self):
        """returns the prompt to start the model with"""
        return "Answer the question: Who is referred to as <mask> in the following text?\n\n"

    def run_model(self):
        # check if results already exist
        cached = self.check_cache()
        if all(cached.values()):
            logging.info(f"Results already exist, skipping model {self.model_name}")
            return
        # load tokenizer
        self.tokenizer = self.get_tokenizer()
        # prepare examples for different configs
        self.prepare_examples()
        # load tokenizer and model
        self.model = self.get_model()
        # run model for different configs
        for config in self.configs:
            if cached[config]:
                logging.info(f"Results already exist for {config} config, skipping")
                continue
            df = self.examples[config]
            # make config available for whole runner instance
            self.config = config
            # run model on examples
            logging.info(f"Running model {self.model_name} for {config} config")
            batch_size = self.batch_sizes()[self.model_name]
            result_df = df.map(self.make_predictions, batched=True, batch_size=batch_size, remove_columns=df.column_names)
            PATH = self.get_path(config)
            result_df.to_json(PATH)

    def make_predictions(self, examples):
        # tokenize inputs and move to GPU
        texts = examples[f"masked_text_{self.config}"]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        # generate predictions
        pad_token = self.tokenizer.eos_token_id
        generated_ids = self.model.generate(**inputs, early_stopping=True, num_return_sequences=1, pad_token_id=pad_token, max_new_tokens=5)
        # decode predictions
        outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        # get prediction and remove the input from the output
        input_lengths = [len(i) for i in examples[f"masked_text_{self.config}"]]
        return { "prediction": outputs, "page_id": examples["id"], "input_length": input_lengths }