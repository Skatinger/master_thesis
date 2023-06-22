

from ..abstract_fill_mask_runner import AbstractFillMaskRunner
from transformers import DistilBertModel, DistilBertTokenizer


class DistilbertRunner(AbstractFillMaskRunner):

    @staticmethod
    def names():
        return {
            "distilbert-0b066": "distilbert-base-uncased",
        }
    
    @staticmethod
    def sizes():
        return {
            "XXS": "distilbert-0b066",
        }

    @staticmethod
    def batch_sizes():
        return {
            "distilbert-0b066": 64,
        }
    
    def _model_loader(self):
        return DistilBertModel

    def _tokenizer_loader(self):
        return DistilBertTokenizer
