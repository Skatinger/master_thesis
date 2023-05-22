import re
import sys
from datasets import load_dataset, Dataset
import Levenshtein

class SinglePredictionEvaluator:

    def __init__(self, result_dataset: Dataset = None, result_path: str = None, gt: Dataset = None):
        if result_path:
            self.dataset = load_dataset("json", data_files=result_path, split="train").sort("page_id")
        if not gt:
            print("Loading ground truth dataset...")
            self.gt = load_dataset("Skatinger/wikipedia-persons-masked", split="train")
        else:
            self.gt = gt
        # only keep gt for pages which are in the result dataset
        print("Filtering ground truth dataset...")
        id_set = set(self.dataset["page_id"])
        self.gt = self.gt.filter(lambda x: x["id"] in id_set, num_proc=4).sort("id")        
    
    def compute_precision_for_page(self, page):
        distance = Levenshtein.distance(page["prediction"], page["title"], score_cutoff=15)
        regex = "|".join(['.*(' + nameFragment + ').*' for nameFragment in page["title"].split()])
        if re.match(regex, page["prediction"]):
            return { "correct": 1, "prediction": page["prediction"], "title": page["title"], "distance": distance }
        else:
            return { "correct": 0, "prediction": page["prediction"], "title": page["title"], "distance": distance }
    
    @staticmethod
    def compute_precision_for_page(page):
        distance = Levenshtein.distance(page["prediction"], page["title"], score_cutoff=15)
        regex = "|".join(['.*(' + nameFragment + ').*' for nameFragment in page["title"].split()])
        if re.match(regex, page["prediction"]):
            return { "correct": 1, "prediction": page["prediction"], "title": page["title"], "distance": distance }
        else:
            return { "correct": 0, "prediction": page["prediction"], "title": page["title"], "distance": distance }
    
    @staticmethod
    def compute_metrics(gt, data):
        results = {}
        # only keep ground truth entries with a mask
        input_length = 1000 # TODO: dynamic
        gt_with_mask = {}
        gt_with_mask['original'] = gt.filter(lambda x: '<mask>' in x['masked_text_original'][:input_length])
        gt_with_mask['paraphrased'] = gt.filter(lambda x: '<mask>' in x['masked_text_paraphrased'][:input_length])
        for key, models in data.items():
            for model_name, model in models.items():
                for config in ['original', 'paraphrased']:
                    dataset = model[config]['train']
                    gt = gt_with_mask[config]
                    ## add ground truth label to each prediction
                    # some legacy examples contain predictions for examples which do not
                    # contain a mask, the prediction therefore cannot be correct. remove those
                    # in case any exist.
                    ids = set(gt['id'])
                    dataset = dataset.filter(lambda x: x['page_id'] in ids)
                    # make sure only the examples which were actually predicted
                    mappable = dataset.add_column("title", gt["title"])                    
                    # compute precision
                    computed = mappable.map(SinglePredictionEvaluator.compute_precision_for_page, num_proc=8, remove_columns=mappable.column_names)
                    # compute metrics over computed results
                    correct_predictions = computed.filter(lambda x: x['correct'] == 1)
                    incorrect_predictions = computed.filter(lambda x: x['correct'] == 0)
                    correct = len(correct_predictions)
                    model[config]["result"] = {}
                    model[config]["result"]["data"] = computed
                    model[config]["result"]["accuracy"] = correct / len(dataset)
                    model[config]["result"]["correct_predictions"] = correct_predictions
                    model[config]["result"]["incorrect_predictions"] = incorrect_predictions
        return data


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
        return correct / len(self.dataset), correct_predictions, incorrect_predictions, 

    # def compute_metrics(self):
    #     pass

if __name__ == "__main__":
    print("THIS SHOULD NOT BE CALLED")
    # use argv[1] as path to result file
    if len(sys.argv) > 1:
        result_path = sys.argv[1]
    else:
        result_path = "../models/wiki_predictions_gpt-3.5-turbo_paraphrased.jsonl"

    ev = SinglePredictionEvaluator(result_path=result_path)
    accuracy, correct, incorrect, results = ev.compute_precision() # ev.dataset['prediction'], ev.gt['title'])
    levenstein_below_3_in_correct = correct.filter(lambda x: x['distance'] < 3)
    levenstein_below_3_in_incorrect = incorrect.filter(lambda x: x['distance'] < 3)
    average_levenstein_correct = sum(correct['distance']) / len(correct)
    average_levenstein_incorrect = sum(incorrect['distance']) / len(incorrect)
    print(f"\n MODEL {result_path} \n")
    print("\n===== Correct Predictions: (first 20)")
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
