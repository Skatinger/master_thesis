import re
import sys
from datasets import load_dataset
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
    levenstein_below_3_in_correct = [x for x in correct if x[2] < 3]
    levenstein_below_3_in_incorrect = [x for x in incorrect if x[2] < 3]
    average_levenstein_correct = sum([x[2] for x in correct]) / len(correct)
    average_levenstein_incorrect = sum([x[2] for x in incorrect]) / len(incorrect)
    print(f"\n MODEL {result_path} \n")
    print("\n===== Correct Predictions: (first 20)")
    for pred, label, dist in correct[:20]:
        print(f"{pred:50} {label:100} {dist}")
    print("\n===== Incorrect predictions: (first 20)")
    for pred, label, dist in incorrect[:20]:
        # print prediction and label with same identation
        print(f"{pred:50} {label:100} {dist}")
    print(f"\n===== Summary (for result {result_path}):")
    print(f"Number of entries: {len(ev.dataset)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Correct predictions: {len(correct)}")
    print(f"Incorrect predictions: {len(incorrect)}")
    print(f"Levenstein distance below 3 in correct predictions: {len(levenstein_below_3_in_correct)}/{len(correct)}")
    print(f"Average levenstein distance in correct predictions: {average_levenstein_correct:.2f}")
    print(f"Average levenstein distance in incorrect predictions: {average_levenstein_incorrect:.2f}")
    print(f"\n\n=====================")

    # write results to json file
    import json
    file_name = result_path.split("/")[-1].split(".")[0]
    save_path = f"results-{file_name}.json"
    print(f"Writing results to {save_path}")
    with open(save_path, "w") as f:
        json.dump({
            "path": result_path,
            "accuracy": accuracy,
            "average_levenstein_correct": average_levenstein_correct,
            "average_levenstein_incorrect": average_levenstein_incorrect,
            "levenstein_below_3_in_correct": levenstein_below_3_in_correct,
            "correct": correct,
            "incorrect": incorrect
        }, f)