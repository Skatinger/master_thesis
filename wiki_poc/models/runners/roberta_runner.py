

from .abstract_fill_mask_runner import AbstractFillMaskRunner
from transformers import RobertaForMaskedLM

class RobertaRunner(AbstractFillMaskRunner):

    @staticmethod
    def names():
        return {
            "roberta-0b125": "roberta-base",
            "roberta-0b355": "roberta-large",
        }
    
    @staticmethod
    def sizes():
        return {
            "XXS": "roberta-0b125", # 125M
            "XS": "roberta-0b355", # 355M
        }

    @staticmethod
    def batch_sizes():
        return {
            "roberta-0b125": 64,
            "roberta-0b355": 64,
        }

    def _model_loader(self):
        return RobertaForMaskedLM
