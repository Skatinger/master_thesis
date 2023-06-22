from ..abstract_qa_runner import AbstractQARunner
import torch
import logging
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

    def get_model(self):
        """override as device_map='auto' is not supported by distilbert"""
        if torch.cuda.is_available():
            logging.info(f"Loading model for {self.model_name}")
            model_path = self.names()[self.model_name]
            return self._model_loader().from_pretrained(model_path, torch_dtype=torch.float16, device=self.device)
        else:
            self.super().get_model()