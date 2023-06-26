

from ..abstract_fill_mask_runner import AbstractFillMaskRunner
from transformers import DebertaForMaskedLM

class DebertaRunner(AbstractFillMaskRunner):

    @staticmethod
    def names():
        return {
            "mdeberta-0b086": "microsoft/mdeberta-v3-base",
            "deberta-0b304": "microsoft/deberta-v3-large",
        }
    
    @staticmethod
    def sizes():
        return {
            "XXS": "mdeberta-0b086",
            "XS": "deberta-0b304",
        }

    @staticmethod
    def batch_sizes():
        return {
            "mdeberta-0b086" : 32,
            "deberta-0b304": 32,
        }

    def _model_loader(self):
        return DebertaForMaskedLM
