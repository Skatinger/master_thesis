
from ..abstract_fill_mask_runner import AbstractFillMaskRunner
from transformers import RobertaForMaskedLM, RobertaTokenizer

class BertRunner(AbstractFillMaskRunner):

    @staticmethod
    def names():
        return {
            "swiss_bert-0b110": "ZurichNLP/swissbert",
            "xlm_swiss_bert-0b110": "ZurichNLP/swissbert-xlm-vocab",
        }
    
    @staticmethod
    def sizes():
        return {
            "XXS": "swiss_bert-0b110",
        }
    @staticmethod
    def batch_sizes():
        return {
            "swiss_bert-0b110": 64,
            "xlm_swiss_bert-0b110": 64,
        }

