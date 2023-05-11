import logging
from ..abstract_base import AbstractRunner

class LlamaRunner(AbstractRunner):

    def __init__(self, model_name, dataset, options = {}):
        logging.warning("LlamaRunner expects model weights converted to huggingface format at ~/.cache/llama-converted")
    
    @staticmethod
    def names():
        return {
            "llama-7b": "/home/alex/.cache/llama-converted/llama-7b",
            "llama-13b": "/home/alex/.cache/llama-converted/llama-13b",
        }
    
    @staticmethod
    def sizes():
        return {
            "L": "llama-7b",
            "XL": "llama-13b",
        }
        
    @staticmethod
    def batch_sizes():
        return {
            "llama-7b": 64,
            "llama-13b": 64,
        }