from ..abstract_runner import AbstractRunner
import logging
import torch
from transformers import LlamaModel, LlamaConfig
from transformers import LlamaForCausalLM, LlamaTokenizer

class HuggyLlama2Runner(AbstractRunner):

    @staticmethod
    def names():
        return {
            "llama2-7b": "meta-llama/llama-2-7b",
            "llama2-13b": "meta-llama/llama-2-13b",
            # "llama-70b": "huggyllama/llama-70b",
        }
    
    @staticmethod
    def sizes():
        return {
            "L": "llama2-7b",
            "XL": "llama2-13b",
            # "XXXL": "llama-70b",
        }
        
    @staticmethod
    def batch_sizes():
        return {
            "llama2-7b": 8,
            "llama2-13b": 8,
            # "llama-70b": 4,
        }

    def _tokenizer_loader(self):
        return LlamaTokenizer

    def _model_loader(self):
        return LlamaForCausalLM
    
    def get_model(self):
        """retrieves model from huggingface model hub and load it to specified device"""
        logging.info(f"Loading model for {self.model_name}")
        model_path = self.names()[self.model_name]
        configuration = LlamaConfig()
        if torch.cuda.is_available():
            logging.info("GPU available, loading model in FP16 mode on GPU.")
            return self._model_loader().from_pretrained(model_path, config=configuration,
                                                        device_map="auto", use_auth_token=True, from_tf=True)
            
        else:
            logging.warning("GPU not available, loading model in FP32 mode on CPU. This will be very slow.")
            return self._model_loader().from_pretrained(model_path)

    def get_tokenizer(self):
        logging.info(f"Loading tokenizer for {self.model_name}")
        model_path = self.names()[self.model_name]
        tokenizer = self._tokenizer_loader().from_pretrained(model_path, padding_side="left", truncation=True)
        tokenizer.add_special_tokens({"pad_token":"<pad>"})
        # tokenizer.pad_token = tokenizer.eos_token # define pad token as eos token
        return tokenizer