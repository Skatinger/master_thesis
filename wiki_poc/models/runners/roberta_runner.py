

from .abstract_fill_mask_runner import AbstractFillMaskRunner
from transformers import RobertaForMaskedLM

class RobertaRunner(AbstractFillMaskRunner):

    @staticmethod
    def names():
        return {
            "roberta-base": "roberta-base",
            "roberta-large": "roberta-large",
        }
    
    @staticmethod
    def sizes():
        return {
            "XXS": "roberta-base", # 125M
            "XS": "roberta-large", # 355M
        }

    @staticmethod
    def batch_sizes():
        return {
            "roberta-base": 64,
            "roberta-large": 64,
        }

    def _model_loader(self):
        return RobertaForMaskedLM
