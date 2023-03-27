import re
import sys
from typing import Dict, Union, Optional, List, Any, Tuple
from datasets import Dataset, load_dataset

class SinglePredictionEvaluator:

    def __init__(self, result_path: str):
        self.dataset = load_dataset("json", data_files=result_path, split="train").sort("page_id")
        self.gt = load_dataset("Skatinger/wikipedia-persons-masked", split="train")
        # only keep gt for pages which are in the result dataset
        self.gt = self.gt.filter(lambda x: x["id"] in self.dataset["page_id"]).sort("id")
    

    # builder for a regex which checks if a string approximately matches the given entity
    # same regex as used for masking the wikipedia dataset e.g. only NER predictions fitting this regex
    # were actually masked
    def name_regex(self, entity):
        regex = ".*".join(entity.split(" "))
        # remove any special characters
        regex = re.sub(r'[^a-zA-Z0-9 ]', '', regex)
        regex += ".*".join(['|.*(' + nameFragment + ').*' for nameFragment in entity.split(" ")])
        return regex


    # compute accuracy
    def compute_accuracy(self):
        correct = 0
        correct_predictions = []
        incorrect_predictions = []
        for prediction, label in zip(self.dataset["prediction"], self.gt["title"]):
            if re.match(self.name_regex(label), prediction):
                correct += 1
                correct_predictions.append((prediction, label))
            else:
                incorrect_predictions.append((prediction, label))
        return correct / len(self.dataset), correct_predictions, incorrect_predictions
    

if __name__ == "__main__":
    # use argv[1] as path to result file
    if len(sys.argv) > 1:
        result_path = sys.argv[1]
    else:
        result_path = "../models/wiki_predictions_gpt-3.5-turbo_paraphrased.jsonl"

    ev = SinglePredictionEvaluator(result_path)
    accuracy, correct, incorrect = ev.compute_accuracy()
    print(len(ev.dataset))
    print(f"Accuracy: {accuracy:.2%}")
    print("\n===== Correct Predictions:")
    for pred, label in correct:
        print(f"{pred:50} {label}")
    print("\n===== Incorrect predictions:")
    for pred, label in incorrect:
        # print prediction and label with same identation
        print(f"{pred:50} {label}")