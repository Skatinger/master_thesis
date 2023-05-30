import torch
import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset

from abc import ABC, abstractmethod, abstractproperty

class AbstractRunner():

    def __init__(self, model_name, dataset, options = {"device": 0}):
        """_summary_

        Args:
            dataset (_type_): _description_
            options (dict, optional): Could pass options for model SIZE, CONFIG (original/paraphrased).
                                      Defaults to {}.
        """

        logging.info("Initializing runner for model %s", model_name)
        # key to identify the run in the results
        self.key = options["key"]
        self.model_name = model_name
        self.dataset = dataset
        self.input_length = 1000
        self.save_memory = options["save_memory"]
        self.set_options(options)
        self.base_path = f"results/{self.key}/{self.model_name}"
        self.configs = ['paraphrased', 'original']
        self.device_number = options["device"]
        # set the device number as environment variable so any models using the `devicemap=auto` option
        # will load the model to the correct GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = self.device_number
        self.device = torch.device(f"cuda:{self.device_number}" if torch.cuda.is_available() else "cpu")
        logging.info(f"Set device to {self.device}")

    def set_options(self, options):
        self.options = options
    
    def results_exist(self, config):
        """checks if results for current config already exist"""
        return os.path.exists(self.get_path(config))
    
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
        # if GPU is available, load in 8bit mode
        if torch.cuda.is_available():
            return AutoModelForCausalLM.from_pretrained(model_path, load_in_8bit=True, device_map="auto")
        else:
            logging.warning("GPU not available, loading model in FP32 mode on CPU. This will be very slow.")
            return AutoModelForCausalLM.from_pretrained(model_path)

    @staticmethod
    @abstractproperty
    def sizes(self):
        pass

    def prepare_examples(self):
        """shortens input text to max length given and pre- and append prompt to examples"""
        logging.info(f"Preparing examples for {self.model_name}")
        self.examples = {}
        for config in ['paraphrased', 'original']:
            # shorten input text to max length given
            df = self.dataset.map(lambda x: {f"masked_text_{config}": x[f"masked_text_{config}"][:self.input_length]}, num_proc=8)
            # remove all examples which do no longer contain a mask
            df = df.filter(lambda x: '<mask>' in x[f"masked_text_{config}"], num_proc=8)
            # pre- and append prompt to examples
            start, end = self.start_prompt(), self.end_prompt()
            df = df.map(lambda x: {f"masked_text_{config}": start + x[f"masked_text_{config}"] + end})
            self.examples[config] = df

    def get_path(self, config):
        """returns path to save results to"""
        return f"{self.base_path}_{config}_{self.input_length}.json"

    def check_cache(self):
        """checks if results already exist for configs, returns dict with config as key and bool as value"""
        cached = {}
        for config in self.configs:
            cached[config] = self.results_exist(config)
        return cached

    def run_model(self):
        # check if results already exist
        cached = self.check_cache()
        if all(cached.values()):
            logging.info(f"Results already exist, skipping model {self.model_name}")
            return
        # prepare examples for different configs
        self.prepare_examples()
        # load tokenizer and model
        self.tokenizer = self.get_tokenizer()
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
        predictions = [out.replace(examples[f"masked_text_{self.config}"][i], "") for i, out in enumerate(outputs)]
        input_lengths = [len(i) for i in examples[f"masked_text_{self.config}"]]
        return { "prediction": predictions, "page_id": examples["id"], "input_length": input_lengths }