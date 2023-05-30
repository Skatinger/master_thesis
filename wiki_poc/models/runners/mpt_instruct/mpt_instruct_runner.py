import logging
import transformers
import torch
from transformers import AutoTokenizer

from ..abstract_runner import AbstractRunner

class MPTInstructRunner(AbstractRunner):


    @staticmethod
    def start_prompt():
        return """
                Below is an instruction that describes a task. Write a response that appropriately completes the request.
                ### Instruction:
                The following text is an extract from a wikipedia page. The text is about a person but the person is referred to as <mask>.
                Please give the name of the person referred to as <mask> and only the name. If you don't know the name,
                give your best guess.

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
            "mpt_instruct-6.7b": "mosaicml/mpt-7b-instruct"
        }
    
    @staticmethod
    def sizes():
        return {
            "L": "mpt_instruct-6.7b",
        }

    @staticmethod
    def batch_sizes():
        return {
            "mpt_instruct-6.7b": 64,
        }

    def get_tokenizer(self):
        logging.info(f"Loading tokenizer for {self.model_name}")
        return AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    def get_model(self):
        """retrieves model from huggingface model hub and load it to specified device"""
        logging.info(f"Loading model for {self.model_name}")
        model_path = self.names()[self.model_name]

        # requires `trust_remote_code` as not all parts are yet included in the transformers library
        config = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.attn_config['attn_impl'] = 'triton'

        if torch.cuda.is_available():
            return transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            logging.warning("GPU not available, loading model in FP32 mode on CPU. This will be very slow.")
            return transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
