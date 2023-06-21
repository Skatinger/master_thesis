
from ..abstract_runner import AbstractRunner
import logging
import torch

class FalconRunner(AbstractRunner):


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
            "falcon-40b": "tiiuae/falcon-40b",
            "falcon-7b": "tiiuae/falcon-7b",
        }
    
    @staticmethod
    def sizes():
        return {
            "XL": "falcon-7b",
            "XXL": "falcon-40b",
        }
        
    @staticmethod
    def batch_sizes():
        return {
            "falcon-7b": 8,
            "falcon-40b": 2,
        }

    def get_model(self):
        """retrieves model from huggingface model hub and load it to specified device"""
        logging.info(f"Loading model for {self.model_name}")
        model_path = self.names()[self.model_name]
        # if GPU is available, load in 8bit mode
        if torch.cuda.is_available():
            return self._model_loader().from_pretrained(
                model_path, load_in_8bit=True, device_map="auto", trust_remote=True)
        else:
            logging.warning("GPU not available, cannot load this model.")
            exit(1)
