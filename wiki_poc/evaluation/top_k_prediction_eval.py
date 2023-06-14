import re
import sys
import json
import Levenshtein
from evaluation.loader import ResultLoader

class TopKPredictionEvaluator:
    
    @staticmethod
    def compute_precision_for_page(page, k_runs):
        # compute precision with levensthein distance
        distances = {}
        for i in range(k_runs):
            distances[f"prediction_{i}"] = Levenshtein.distance(page["prediction_" + str(i)], page["title"], score_cutoff=15)
        
        ## compute accuracy with regex
        regex = "|".join(['.*(' + nameFragment + ').*' for nameFragment in page["title"].split()])
        predicted_string = ""
        # iterate predictions, add all matching predictions to a prediction_string
        min_distance = sys.maxsize
        for i in range(k_runs):
            prediction = page["prediction_" + str(i)]
            if re.match(regex, prediction):
                predicted_string += f" {prediction}"
                # use the minimum distance of all predictions which were classified as correct
                min_distance = distances[f"prediction_{i}"] if distances[f"prediction_{i}"] < min_distance else min_distance

        # if we got any correct predictions for the page
        if len(predicted_string) > 0:
            return { "correct": 1, "prediction": page["prediction"], "title": page["title"], "distance": min_distance }
        else:
            return { "correct": 0, "prediction": page["prediction"], "title": page["title"], "distance": min_distance }
    
    @staticmethod
    def compute_metrics(gt, data):
        # TODO: dynamic, field input_length in the result dataset is a representation of the input length
        # given to the model, not the length of the input text
        input_length = 1000
        gt_with_mask = {}
        # only keep ground truth entries with a mask, as only those were predicted
        gt_with_mask['original'] = gt.filter(lambda x: '<mask>' in x['masked_text_original'][:input_length])
        gt_with_mask['paraphrased'] = gt.filter(lambda x: '<mask>' in x['masked_text_paraphrased'][:input_length])
        for _key, models in data.items():
            for _model_name, model in models.items():
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
                    k_runs = sum(key.startswith("prediction_") for key in data.keys())
                    if k_runs == 0:
                        # legacy result format, only one prediction per example with key 'prediction'.
                        # add a key 'prediction_0' to each example with the same value as 'prediction'
                        mappable = mappable.map(lambda x: {**x, "prediction_0": x["prediction"]})
                        k_runs = 1
                    computed = mappable.map(TopKPredictionEvaluator.compute_precision_for_page, num_proc=8, remove_columns=mappable.column_names, fn_kwargs={'k_runs': k_runs})
                    # compute metrics over computed results
                    correct_predictions = computed.filter(lambda x: x['correct'] == 1)
                    incorrect_predictions = computed.filter(lambda x: x['correct'] == 0)
                    correct = len(correct_predictions)
                    # take the average of the column 'distance' to get the average levenshtein distance
                    average_precision = sum(correct_predictions['distance']) / len(correct_predictions)
                    model[config]["result"] = {}
                    model[config]["result"]["data"] = computed
                    model[config]["result"]["accuracy"] = correct / len(dataset)
                    model[config]["result"]["precision"] = average_precision
                    model[config]["result"]["correct_predictions"] = correct_predictions
                    model[config]["result"]["incorrect_predictions"] = incorrect_predictions
        return data

def main():
    assert len(sys.argv) > 1, "Please provide a path to the result file as an argument"

    key = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else None

    loader = ResultLoader()
    print("loading ground truth")
    gt = loader.load_gt()
    if model_name is not None:
        results = loader.load(key, model_name)
    else:
        results = loader.load(key)

    from datasets import disable_caching
    disable_caching()
    print("computing metrics")
    results = TopKPredictionEvaluator.compute_metrics(gt, results)

    # print results
    json_results = {'key': key}
    for model_class, models in results.items():
        for m_name, model in models.items():
            name = f"{model_class}-{m_name}"
            json_results[name] = {}
            for config in ['original', 'paraphrased']:
                print(f"Model: {name:<15} Config: {config}")
                print(f"Accuracy: {model[config]['result']['accuracy']}")
                print(f"Precision: {model[config]['result']['precision']}")
                json_results[name][config] = {}
                json_results[name][config]['accuracy'] = model[config]['result']['accuracy']
                json_results[name][config]['precision'] = model[config]['result']['precision']

    # write results to json file
    file_name = f"{key}-results" if model_name is None else f"{key}-{model_name}-results"
    save_path = f"evaluation/results/{file_name}.json"
    print(f"Writing results to {save_path}")

    with open(save_path, 'w') as f:
        json.dump(json_results, f, indent=4)


if __name__ == "__main__":
    main()