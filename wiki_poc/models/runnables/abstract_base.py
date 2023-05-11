import os
import sys
import torch
import signal
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset


from abc import ABC, abstractmethod

global CONFIG
global DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global tokenizer
global model_8bit


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
        self.set_options(options)
        self.base_path = f"results/{self.model_name}"

        self.tokenizer = self.get_tokenizer()
        self.model = self.get_model()

    def set_options(self, options):
        self.options = options
    
    def start_prompt():
        return "The following text talks about a person but the person is referred to as <mask>.\n\n"

    def end_prompt():
        return "\n\nThe name of the person in the text referred to as <mask> is: "

    @abstractmethod
    def names(self):
        pass

    def get_tokenizer(self):
        print("GETTING tokenizer")
        print("for model", self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token # define pad token as eos token
        return tokenizer

    def get_model(self):
        return AutoModelForCausalLM.from_pretrained(self.model_name).to(DEVICE)

    @abstractmethod
    def sizes(self):
        pass

    @abstractmethod
    def short_names(self):
        pass

    def run_model(self):
        for config in ['paraphrased', 'original']:
            logging.info(f"Runnig {self.model_name} for {config}")
        # pre- and append prompt to examples
            start, end = self.start_prompt(), self.end_prompt()
            df = self.dataset.map(lambda x: {f"masked_text_{config}": start + x[f"masked_text_{config}"] + end})
            # run model on examples
            logging.info(f"Running model {self.model_name} for {config} config")
            result_df = df.map(self.make_predictions, batched=True, batch_size=2, remove_columns=df.column_names)
            PATH = f"{self.base_path}_{config}.json"
            result_df.to_json(PATH)


    def make_predictions(self, examples):
        # tokenize inputs and move to GPU
        inputs = self.tokenizer(examples[f"masked_text_{CONFIG}"], return_tensors="pt", padding=True).to(DEVICE)
        # generate predictions
        generated_ids = model_8bit.generate(**inputs, early_stopping=True, num_return_sequences=1, max_new_tokens=5)
        # decode predictions
        outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        # get prediction and remove the input from the output
        predictions = [out.replace(examples[f"masked_text_{CONFIG}"][i], "") for i, out in enumerate(outputs)]
        input_lengths = [len(i) for i in examples[f"masked_text_{CONFIG}"]]
        return { "prediction": predictions, "page_id": examples["id"], "input_length": input_lengths }