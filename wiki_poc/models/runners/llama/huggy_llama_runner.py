from ..abstract_runner import AbstractRunner

class HuggyLlamaRunner(AbstractRunner):

    @staticmethod
    def names():
        return {
            "llama-7b": "huggyllama/llama-7b",
        }
    
    @staticmethod
    def sizes():
        return {
            "L": "llama-7b",
            # "XL": "llama-13b",
        }
        
    @staticmethod
    def batch_sizes():
        return {
            "llama-7b": 64,
            # "llama-13b": 64,
        }