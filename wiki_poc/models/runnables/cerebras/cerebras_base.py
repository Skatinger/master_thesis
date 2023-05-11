
from ..abstract_base import AbstractRunner


class CerebrasRunner(AbstractRunner):

    @staticmethod
    def names():
        return {
            "cerebras-111m": "cerebras/Cerebras-GPT-111M",
            "cerebras-1b3": "cerebras/Cerebras-GPT-1.3B",
            "cerebras-2b7": "cerebras/Cerebras-GPT-2.7B",
            "cerebras-6b7": "cerebras/Cerebras-GPT-6.7B",
            "cerebras-13b": "cerebras/Cerebras-GPT-13B",
        }
    
    @staticmethod
    def sizes():
        return {
            "XXS": "cerebras-111m",
            "XS": "cerebras-1b3",
            "M": "cerebras-2b7",
            "L": "cerebras-6b7",
            "XL": "cerebras-13b",
        }

    @staticmethod
    def batch_sizes():
        return {
            "cerebras-111m": 64,
            "cerebras-1b3": 64,
            "cerebras-2b7": 64,
            "cerebras-6b7": 64,
            "cerebras-13b": 64,
        }