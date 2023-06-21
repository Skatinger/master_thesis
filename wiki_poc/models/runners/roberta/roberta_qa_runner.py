
from ..abstract_qa_runner import AbstractQARunner

class RobertaQARunner(AbstractQARunner):


    @staticmethod
    def names():
        return {
            "roberta_base_squad-0b125": "deepset/roberta-base-squad2",
            "roberta_large_squad-0b355": "deepset/roberta-large-squad2",
        }
    
    @staticmethod
    def sizes():
        return {
            "XXS": "roberta-0b125",
            "XS": "roberta-0b355",
        }

    @staticmethod
    def batch_sizes():
        return {
            "roberta-0b125": 64,
            "roberta-0b355": 64,
        }