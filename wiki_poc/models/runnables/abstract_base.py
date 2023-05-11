import os
import sys
import torch
import signal
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset


from abc import ABC, abstractmethod, abstractproperty

class AbstractRunner():

    def __init__(self, model_name, dataset, options = {}):
        """_summary_

        Args:
            dataset (_type_): _description_
            options (dict, optional): Could pass options for model SIZE, CONFIG (original/paraphrased).
                                      Defaults to {}.
        """

        print("starting with", model_name, dataset, options)
        self.model_name = model_name
        self.dataset = dataset
        self.input_length = 1000
        self.set_options(options)
        self.base_path = f"results/{self.model_name}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_options(self, options):
        self.options = options
    
    @staticmethod
    def start_prompt():
        return "The following text talks about a person but the person is referred to as <mask>.\n\n"

    @staticmethod
    def end_prompt():
        return "\n\nThe name of the person in the text referred to as <mask> is: "

    @staticmethod
    @abstractproperty
    def names(self):
        pass

    def get_tokenizer(self):
        logging.info(f"Loading tokenizer for {self.model_name}")
        model_path = self.names()[self.model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token # define pad token as eos token
        return tokenizer

    def get_model(self):
        """retrieves model from huggingface model hub and load it to specified device"""
        logging.info(f"Loading model for {self.model_name}")
        model_path = self.names()[self.model_name]
        return AutoModelForCausalLM.from_pretrained(model_path, load_in_8bit=True, device_map="auto")

    @staticmethod
    @abstractproperty
    def sizes(self):
        pass

    def prepare_examples(self):
        # shorten input text to max length given
        df = self.dataset.map(lambda x: {f"masked_text_{self.config}": x[f"masked_text_{self.config}"][:self.input_length]}, num_proc=8)
        # pre- and append prompt to examples
        start, end = self.start_prompt(), self.end_prompt()
        df = df.map(lambda x: {f"masked_text_{self.config}": start + x[f"masked_text_{self.config}"] + end})
        return df

    def run_model(self):
        for config in ['paraphrased', 'original']:
            self.config = config
            df = self.prepare_examples()

            self.tokenizer = self.get_tokenizer()
            self.model = self.get_model()

            # run model on examples
            logging.info(f"Running model {self.model_name} for {self.config} config")
            batch_size = self.sizes()[self.model_name]
            result_df = df.map(self.make_predictions, batched=True, batch_size=batch_size, remove_columns=df.column_names)
            PATH = f"{self.base_path}_{self.config}.json"
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
        predictions = [out.replace(examples[f"masked_text_{self.config}"][i], "") for i, out in enumerate(outputs)]
        input_lengths = [len(i) for i in examples[f"masked_text_{self.config}"]]
        return { "prediction": predictions, "page_id": examples["id"], "input_length": input_lengths }