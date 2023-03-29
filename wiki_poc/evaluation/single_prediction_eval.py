import re
import sys
from typing import Dict, Union, Optional, List, Any, Tuple
from datasets import Dataset, load_dataset
import Levenshtein

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
    # def compute_precision(self):
    #     correct = 0
    #     correct_predictions = []
    #     incorrect_predictions = []
    #     for prediction, label in zip(self.dataset["prediction"], self.gt["title"]):
    #         if re.match(self.name_regex(label), prediction):
    #             correct += 1
    #             correct_predictions.append((prediction, label))
    #         else:
    #             incorrect_predictions.append((prediction, label))
    #     return correct / len(self.dataset), correct_predictions, incorrect_predictions

    def compute_precision(self, predictions, labels):
        """takes an array of predictions and labels and computes the average levenshtein difference"""

        correct = 0
        correct_predictions = []
        incorrect_predictions = []
        for prediction, label in zip(predictions, labels):
            distance = Levenshtein.distance(prediction, label)
            if re.match(self.name_regex(label), prediction):
                correct += 1
                correct_predictions.append((prediction, label, distance))
            else:
                incorrect_predictions.append((prediction, label, distance))
        return correct / len(self.dataset), correct_predictions, incorrect_predictions
    

if __name__ == "__main__":
    # use argv[1] as path to result file
    if len(sys.argv) > 1:
        result_path = sys.argv[1]
    else:
        result_path = "../models/wiki_predictions_gpt-3.5-turbo_paraphrased.jsonl"

    ev = SinglePredictionEvaluator(result_path)
    accuracy, correct, incorrect = ev.compute_precision(ev.dataset['prediction'], ev.gt['title'])
    print(f"\n MODEL {result_path} \n")
    print("\n===== Correct Predictions:")
    for pred, label, dist in correct:
        print(f"{pred:50} {label:100} {dist}")
    print("\n===== Incorrect predictions:")
    for pred, label, dist in incorrect:
        # print prediction and label with same identation
        print(f"{pred:50} {label:100} {dist}")
    print(f"\n===== Summary (for result {result_path}):")
    print(f"Number of entries: {len(ev.dataset)}")
    print(f"Accuracy: {accuracy:.2%}\n\n\n=====================")