
from ..abstract_runner import AbstractRunner
import torch
import logging

class FalconInstructRunner(AbstractRunner):


    @staticmethod
    def start_prompt():
        return """
                Below is an instruction that describes a task. Write a response that appropriately completes the request.
                ### Instruction:
                The following text is an extract from a wikipedia page. The text is about a person but the person is referred to as <mask>.
                Please give the name of the person referred to as <mask> and only the name. If you don't know the name,
                give your best guess. Do not include any other information in your response.

                The text:

                """

    @staticmethod
    def end_prompt():
        return """

                ### Response:
                """

    @staticmethod
    def names():
        return {
            "falcon_instruct-40b": "tiiuae/falcon-40b-instruct",
            "falcon_instruct-7b": "tiiuae/falcon-7b-instruct",
        }
    
    @staticmethod
    def sizes():
        return {
            "XL": "falcon_instruct-7b",
            "XXL": "falcon_instruct-40b",
        }
        
    @staticmethod
    def batch_sizes():
        return {
            "falcon_instruct-7b": 8,
            "falcon_instruct-40b": 2,
        }

    def get_tokenizer(self):
        logging.info(f"Loading tokenizer for {self.model_name}")
        model_path = self.names()[self.model_name]
        tokenizer = self._tokenizer_loader().from_pretrained(model_path, padding_side="left", return_token_type_ids=False)
        tokenizer.pad_token = tokenizer.eos_token # define pad token as eos token
        return tokenizer

    def get_model(self):
        """retrieves model from huggingface model hub and load it to specified device"""
        logging.info(f"Loading model for {self.model_name}")
        model_path = self.names()[self.model_name]
        # if GPU is available, load in 8bit mode
        if torch.cuda.is_available():
            return self._model_loader().from_pretrained(
                model_path, load_in_8bit=True, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
        else:
            logging.warning("GPU not available, cannot load this model.")
            exit(1)