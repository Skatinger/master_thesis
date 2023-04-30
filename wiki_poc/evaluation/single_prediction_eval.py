import re
import sys
from datasets import load_dataset
import Levenshtein

class SinglePredictionEvaluator:

    def __init__(self, result_path: str):
        print("Loading result dataset...")
        self.dataset = load_dataset("json", data_files=result_path, split="train").sort("page_id")
        print("Loading ground truth dataset...")
        self.gt = load_dataset("Skatinger/wikipedia-persons-masked", split="train")
        # only keep gt for pages which are in the result dataset
        print("Filtering ground truth dataset...")
        id_set = set(self.dataset["page_id"])
        self.gt = self.gt.filter(lambda x: x["id"] in id_set, num_proc=4).sort("id")        
    
    def compute_precision_for_page(self, page):
        distance = Levenshtein.distance(page["prediction"], page["title"], score_cutoff=5)
        regex = "|".join(['.*(' + nameFragment + ').*' for nameFragment in page["title"].split()])
        if re.match(regex, page["prediction"]):
            return { "correct": 1, "prediction": page["prediction"], "title": page["title"], "distance": distance }
        else:
            return { "correct": 0, "prediction": page["prediction"], "title": page["title"], "distance": distance }

    def compute_precision(self): # , predictions, labels):
        """takes an array of predictions and labels and computes the average levenshtein difference"""
        # join gt and dataset on page_id
        print("Joining datasets...")
        mappable = self.dataset
        # add label column to dataset
        mappable = mappable.add_column("title", self.gt["title"])
        print("Computing precision. This may take a while...")
        results = mappable.map(self.compute_precision_for_page, num_proc=4, remove_columns=mappable.column_names)
        correct_predictions = results.filter(lambda x: x['correct'] == 1)
        incorrect_predictions = results.filter(lambda x: x['correct'] == 0)
        correct = len(correct_predictions)
        return correct / len(self.dataset), correct_predictions, incorrect_predictions, results

if __name__ == "__main__":
    # use argv[1] as path to result file
    if len(sys.argv) > 1:
        result_path = sys.argv[1]
    else:
        result_path = "../models/wiki_predictions_gpt-3.5-turbo_paraphrased.jsonl"

    ev = SinglePredictionEvaluator(result_path)
    accuracy, correct, incorrect, results = ev.compute_precision() # ev.dataset['prediction'], ev.gt['title'])
    levenstein_below_3_in_correct = correct.filter(lambda x: x['distance'] < 3)
    levenstein_below_3_in_incorrect = incorrect.filter(lambda x: x['distance'] < 3)
    average_levenstein_correct = sum(correct['distance']) / len(correct)
    average_levenstein_incorrect = sum(incorrect['distance']) / len(incorrect)
    print(f"\n MODEL {result_path} \n")
    print("\n===== Correct Predictions: (first 20)")
    # for pred, label, dist in correct[:20]:
        # print(f"{pred:50} {label:100} {dist}")
    # print("\n===== Incorrect predictions: (first 20)")
    # for pred, label, dist in incorrect[:20]:
        # print prediction and label with same identation
        # print(f"{pred:50} {label:100} {dist}")
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
    results.to_json(save_path)
    # with open(save_path, "w") as f:
    #     json.dump({
    #         "path": result_path,
    #         "accuracy": accuracy,
    #         "average_levenstein_correct": average_levenstein_correct,
    #         "average_levenstein_incorrect": average_levenstein_incorrect,
    #         "levenstein_below_3_in_correct": levenstein_below_3_in_correct,
    #         "correct": correct,
    #         "incorrect": incorrect
    #     }, f)