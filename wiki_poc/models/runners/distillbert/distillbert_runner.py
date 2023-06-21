

from ..abstract_fill_mask_runner import AbstractFillMaskRunner
from transformers import DistilBertModel, DistilBertTokenizer


class DistilbertRunner(AbstractFillMaskRunner):

    @staticmethod
    def names():
        return {
            "distillbert_base_uncased-0b066": "distilbert-base-uncased",
        }
    
    @staticmethod
    def sizes():
        return {
            "XXS": "distillbert_base_uncased-0b066",
        }

    @staticmethod
    def batch_sizes():
        return {
            "distilbert-base-uncased-0b066": 64,
        }
    
    def _model_loader(self):
        return DistilBertModel

    def _tokenizer_loader(self):
        return DistilBertTokenizer
