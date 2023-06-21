

from ..abstract_fill_mask_runner import AbstractFillMaskRunner
from transformers import RobertaForMaskedLM

class DebertaRunner(AbstractFillMaskRunner):

    @staticmethod
    def names():
        return {
            "mdeberta_v3_base-0b086": "microsoft/mdeberta-v3-base",
            "deberta_v3_large-0b304": "microsoft/deberta-v3-large",
        }
    
    @staticmethod
    def sizes():
        return {
            "XXS": "mdeberta_v3_base-0b086",
            "XS": "deberta_v3_large-0b304",
        }

    @staticmethod
    def batch_sizes():
        return {
            "mdeberta_v3_base-0b086" : 64,
            "deberta_v3_large-0b304": 64,
        }

    def _model_loader(self):
        return RobertaForMaskedLM
