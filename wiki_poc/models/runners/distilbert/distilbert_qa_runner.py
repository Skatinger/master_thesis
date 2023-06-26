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
            return self._model_loader().from_pretrained(model_path, torch_dtype=torch.float16).to(self.device)
        else:
            self.super().get_model()

    def get_tokenizer(self):
        tokenizer = super().get_tokenizer()
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer
    
    def load_pipe(self):
        # override as pad_token seems not to be set correctly for this model
        pipe = super().load_pipe()
        pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
        return pipe