
from ..abstract_runner import AbstractRunner

class InciteInstructRunner(AbstractRunner):
    """
    https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1
    """

    @staticmethod
    def names():
        return {
            "incite_instruct-3b": "togethercomputer/RedPajama-INCITE-Instruct-3B-v1"
        }
    
    @staticmethod
    def sizes():
        return {
            "M": "incite_instruct-3b",
        }

    @staticmethod
    def batch_sizes():
        return {
            "incite_instruct-3b": 256,
        }
