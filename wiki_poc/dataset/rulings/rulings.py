"""masking of the rulings dataset"""

from datasets import Dataset
import re

class RulingsPreparer():

    def __init__(self, dataset: Dataset) -> Dataset:
        self.mask_regex = r"A._"
    
    def prepare_rulings(self, dataset):
        # use id as identifier for consistency
        dataset = dataset.rename_column("decision_id", "id")
        # mask texts
        dataset = self.mask_dataset(dataset)
        dataset = self.remove_unnecessary_columns(dataset)
        return dataset

    def mask_example(self, example):
        """replaces all occurences of the mask_regex with the mask token <mask>"""
        example["masked_text_original"] = re.sub(self.mask_regex, "<mask>", example["full_text"])
        return example
    
    def remove_unnecessary_columns(self, dataset):
        required_columns = ["id", "masked_text_original"]
        remove = [column for column in dataset.column_names if column not in required_columns]
        dataset = dataset.remove_columns(remove)
        return dataset
    
    def mask_dataset(self, dataset):
        dataset = dataset.map(self.mask_example)
        return dataset