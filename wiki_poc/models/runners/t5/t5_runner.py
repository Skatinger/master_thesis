
from ..abstract_text_to_text_runner import AbstractTextToTextRunner


class T5Runner(AbstractTextToTextRunner):
    """
    Website:
        https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Model
    """

    @staticmethod
    def names():
        return {
            "t5-0b060": "t5-small",
            "t5-0b220": "t5-base",
            "t5-0b770": "t5-large",
            "t5-3b": "t5-3b",
            "t5-11b": "t5-11b",
        }
    
    @staticmethod
    def sizes():
        return {
            "XXS": "t5-0b060",
            "XS": "t5-0b220",
            "M": "t5-0b770",
            "L": "t5-3b",
            "XL": "t5-11b",
        }

    @staticmethod
    def batch_sizes():
        return {
            "t5-0b060": 1024,
            "t5-0b220": 1024,
            "t5-0b770": 512,
            "t5-3b": 128,
            "t5-11b": 64,
        }