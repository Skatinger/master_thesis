from .abstract_runner import AbstractRunner
import logging
import torch
import os
from transformers import AutoTokenizer

class AbstractTextToTextRunner(AbstractRunner):

    def get_model(self):
        """retrieves model from huggingface model hub and load it to specified device"""
        logging.info(f"Loading model for {self.model_name}")
        model_path = self.names()[self.model_name]
        # if GPU is available, load in 8bit mode
        if torch.cuda.is_available():
            return self.__model_loader().from_pretrained(model_path, load_in_8bit=True, device_map="auto")
        else:
            logging.warning("GPU not available, loading model in FP32 mode on CPU. This will be very slow.")
            return self.__model_loader().from_pretrained(model_path)
    
    def get_tokenizer(self):
        logging.info(f"Loading tokenizer for {self.model_name}")
        model_path = self.names()[self.model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_path, truncation=True, padding="longest")
        return tokenizer

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
            if self.save_memory:
                batch_size = 1
            # load cached predictions if they exist for this config, pass their column
            # names to the processing so they won't get processed again
            cached_cols = self.cached_predictions[config].column_names if config in self.cached_predictions else {}

            result_df = df.map(self.make_predictions, batched=True, batch_size=batch_size, remove_columns=df.column_names,
                               fn_kwargs={'k_runs': self.k_runs, 'cached_cols': cached_cols, 'config': self.config})
            # add already processed columns to result
            for col_name in cached_cols:
                if col_name not in ['page_id', 'input_length']:
                    result_df = result_df.add_column(col_name, self.cached_predictions[config][col_name])           
            PATH = self.get_path(config)
            result_df.to_json(PATH)

    def make_predictions(self, examples, config, k_runs=1, cached_cols=[]):
        # tokenize inputs and move to GPU
        texts = examples[f"masked_text_{config}"]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        # compute lengths of the inputs to store with the result
        input_lengths = [len(i) for i in examples[f"masked_text_{config}"]]
        pad_token = self.tokenizer.eos_token_id

        predictions = {}
        # generate predictions
        for k in range(k_runs):
            if f"prediction_{k}" in cached_cols:
                continue
            generated_ids = self.model.generate(**inputs, early_stopping=True,
                                                num_return_sequences=1, pad_token_id=pad_token, max_new_tokens=5)
            # decode predictions and remove the input from the output
            predictions[f"prediction_{k}"] = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        predictions["page_id"] = examples["id"]
        predictions["input_length"] = input_lengths
        return predictions