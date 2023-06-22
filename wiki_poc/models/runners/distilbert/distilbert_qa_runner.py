from ..abstract_qa_runner import AbstractQARunner

class DistilbertQARunner(AbstractQARunner):


    @staticmethod
    def names():
        return {
            "distilbert_squad-0b062": "distilbert-base-cased-distilled-squad",
        }
    
    @staticmethod
    def sizes():
        return {
            "XXS": "distilbert_squad-0b062",
        }

    @staticmethod
    def batch_sizes():
        return {
            "distilbert_squad-0b062": 128,
        }