

from ..abstract_fill_mask_runner import AbstractFillMaskRunner

class DebertaQARunner(AbstractFillMaskRunner):

    @staticmethod
    def names():
        return {
            "mdeberta_v3_base-0b086": "microsoft/mdeberta-v3-base",
            "deberta_v3_large-0b304": "microsoft/deberta-v3-large",
            "deberta_v3_large_squad-0b100": "deepset/deberta-v3-large-squad2",
        }
    
    @staticmethod
    def sizes():
        return {
            "XXS": "deberta_v3_base-0b086",
            "XS": "deberta_v3_large_squad-0b100",
        }

    @staticmethod
    def batch_sizes():
        return {
            "deberta_v3_base-0b086": 64,
            "deberta_v3_large_squad-0b100": 64,
        }
