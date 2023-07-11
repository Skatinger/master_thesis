import torch
import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
from accelerate import infer_auto_device_map, init_empty_weights

from abc import ABC, abstractmethod, abstractproperty

# TODO: caching for columns is no longer working correctly, model will always predict top_k results,
#       regardless of wether they are cached or not (e.g. 2 predictions cached, but 5 requested)


class AbstractRunner():

    def __init__(self, model_name, dataset, options = {"device": 0, "k_runs": 1, "save_memory": False, "key": "default"}):
        """_summary_

        Args:
            dataset (_type_): _description_
            options (dict, optional): Could pass options for:
                - input length (in characters)
                - number of runs (top_k)
                - save memory (bool) if true, the runner will reduce batch size to 1
                - device (int) the device to run the model on (only used if device_map is not set to auto)
                - configs (list) the configs to run the model on (e.g. original, paraphrased)
        """

        logging.info("Initializing runner for model %s", model_name)
        # make sure all required options are set
        # key to identify the run in the results
        self.key = options["key"]
        self.model_name = model_name
        self.dataset = dataset

        # set default values for options
        self.input_length = 1000
        self.cached_predictions = {}
        self.k_runs = 1
        self.save_memory = False
        self.device_number = "0"
        self.configs = ['paraphrased', 'original']

        # overwrite default values if options are passed
        self.set_options(options)

        self.base_path = f"results/{self.key}/{self.model_name}"
        self.device = torch.device(f"cuda:{self.device_number}" if torch.cuda.is_available() else "cpu")
        logging.info(f"""Set device to {self.device}. CAREFUL: When using device_map=auto the gpus will be selected automatically,
                     even when a device has been passed.""")

    def set_options(self, options):
        if "input_length" in options:
            self.input_length = int(options["input_length"])
        if "k_runs" in options:
            self.k_runs = options["k_runs"]
        if "save_memory" in options:
            self.save_memory = options["save_memory"]
        if "device" in options:
            self.device_number = options["device"]
        if "configs" in options:
            if not isinstance(options["configs"], list):
                raise ValueError("Option configs must be a list")
            self.configs = options["configs"]
        self.options = options
    
    def results_exist(self, config):
        """checks if results for current config already exist and if all predictions were done"""
        # does the result file even exist?
        if not os.path.exists(self.get_path(config)):
            return False
        
        # load the set
        dataset = load_dataset("json", data_files=self.get_path(config), split="train")
        # store cached predictions
        self.cached_predictions[config] = dataset
        # and check if it contains all k predictions
        required_cols = [f"prediction_{k}" for k in range(self.k_runs)]
        return all(column in dataset.column_names for column in required_cols)

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
        tokenizer = self._tokenizer_loader().from_pretrained(model_path, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token # define pad token as eos token
        return tokenizer
    
    def _model_loader(self):
        return AutoModelForCausalLM
    
    def _tokenizer_loader(self):
        return AutoTokenizer

    def get_model(self):
        """retrieves model from huggingface model hub and load it to specified device"""
        logging.info(f"Loading model for {self.model_name}")
        model_path = self.names()[self.model_name]
        # if GPU is available, load in 8bit mode
        if torch.cuda.is_available():
            # if model is very large (>12 billion parameters), load with  custom device map and memory saving
            if int(self.model_name.split("-")[-1].split("b")[0]) > 1:
                logging.info("Model is very large, loading with custom device map. Use --memory-saving if batches do not fit.")
                return self.load_mapped_model(model_path)
            else:
                return self._model_loader().from_pretrained(model_path, load_in_8bit=True, torch_dtype=torch.float16, device_map="auto")
        else:
            logging.warning("GPU not available, loading model in FP32 mode on CPU. This will be very slow.")
            return self._model_loader().from_pretrained(model_path)

    @staticmethod
    @abstractproperty
    def sizes(self):
        pass

    def load_mapped_model(self, model_path):
        """loads model with custom device map and meta device to save memory on loading"""
        with init_empty_weights():
            meta_model = self._model_loader().from_pretrained(model_path, load_in_8bit=True, torch_dtype=torch.bfloat16)
        device_map = infer_auto_device_map(meta_model, dtype=torch.bfloat16, max_memory = {0: "65GiB", 1: "75GiB", "cpu": "100GiB"})

        model = self._model_loader().from_pretrained(
            model_path, device_map=device_map, offload_folder="offload", offload_state_dict = True, torch_dtype=torch.float16
        )
        return model

    def prepare_examples(self):
        """shortens input text to max length given and pre- and append prompt to examples"""
        logging.info(f"Preparing examples for {self.model_name}")
        self.examples = {}
        for config in self.configs:
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
            # load cached predictions if they exist for this config, pass their column
            # names to the processing so they won't get processed again
            cached_cols = self.cached_predictions[config].column_names if config in self.cached_predictions else {}
            # split cached predictions into batches
            result_df = df.map(self.make_predictions, batched=True, batch_size=batch_size, remove_columns=df.column_names,
                               fn_kwargs={'k_runs': self.k_runs, 'cached_cols': cached_cols, 'config': self.config})
            # add already processed columns to result
            for col_name in cached_cols:
                if col_name not in ['page_id', 'id', 'input_length']:
                    result_df = result_df.add_column(col_name, self.cached_predictions[config][col_name])
            PATH = self.get_path(config)
            result_df.to_json(PATH)

    def make_predictions(self, examples, config, k_runs=1, cached_cols=[]):
        # tokenize inputs and move to GPU
        texts = examples[f"masked_text_{config}"]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.device)
        # compute lengths of the inputs to store with the result
        input_lengths = [len(i) for i in examples[f"masked_text_{config}"]]
        # generate predictions
        pad_token = self.tokenizer.eos_token_id

        predictions = {}
        # model generates k sequences for each input, all concated to one list
        generated_ids = self.model.generate(**inputs, num_beams=k_runs, early_stopping=True, num_return_sequences=k_runs, pad_token_id=pad_token, max_new_tokens=5)
        # decode predictions
        outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        # split outputs into len(inputs) lists to store them as independent predictions
        result = [outputs[i * k_runs: (i + 1) * k_runs] for i in range(len(texts))]
        # initialize predictions dict
        for i in range(k_runs):
            predictions[f"prediction_{i}"] = []
        # get prediction and remove the input from the output, append prediction to the result
        for k, generated_sequences in enumerate(result):
            # for every generated sequence for this example
            for i, out in enumerate(generated_sequences):
                predictions[f"prediction_{i}"].append(out.replace(examples[f"masked_text_{config}"][k], ""))
        
        # append additional info with page_id and input_length for each example
        predictions['page_id'] = examples['id']
        predictions['input_length'] = input_lengths
        return predictions