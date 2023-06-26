

from ..abstract_qa_runner import AbstractQARunner

class DebertaQARunner(AbstractQARunner):

    @staticmethod
    def names():
        return {
            "deberta_squad-0b100": "deepset/deberta-v3-large-squad2",
        }
    
    @staticmethod
    def sizes():
        return {
            "XS": "deberta_squad-0b100",
        }

    @staticmethod
    def batch_sizes():
        return {
            "deberta_squad-0b100": 64,
        }
