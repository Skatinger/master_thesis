import re
import sys
import json
import Levenshtein
from evaluation.loader import ResultLoader

class TopKPredictionEvaluator:
    
    @staticmethod
    def compute_metrics_for_page(page, k_runs):
        """  computes the two metrics 'Partial Name Match Score' and 'String Edit Distance' for a given page
             referred to as "accuracy" and "precision" in the dataset.
        """
        # compute string edit distance with levensthein distance
        distances = {}
        for i in range(k_runs):
            distances[f"prediction_{i}"] = Levenshtein.distance(page["prediction_" + str(i)], page["title"], score_cutoff=15)
        
        ## use a regex to check if any of the predictions contains a substring of the title
        regex = "|".join(['.*(' + nameFragment + ').*' for nameFragment in page["title"].split()])
        predicted_string = ""
        any_correct = False
        # iterate predictions, add all matching predictions to a prediction_string
        min_distance = sys.maxsize
        top_prediction = ""
        for i in range(k_runs):
            prediction = page[f"prediction_{i}"]
            if re.search(regex, prediction):
                any_correct = True
                # use the minimum distance of all predictions which were classified as correct
                if distances[f"prediction_{i}"] < min_distance:
                    min_distance = distances[f"prediction_{i}"]
                    top_prediction = prediction
            predicted_string += f" {prediction}"

        # normalize min_distance by the length of the title
        if not min_distance == 0:
            min_distance = min_distance / len(page["title"])

        # usefull for debugging insights
        # if any_correct > 0 and min_distance < 0.3:
            # print(f"Correct prediction: {top_prediction} for {page['title']}")

        # if we got any correct predictions for the page
        if any_correct > 0:
            return { "correct": 1, "prediction": predicted_string, "top_prediction": top_prediction, "title": page["title"], "distance": min_distance }
        else:
            return { "correct": 0, "prediction": predicted_string, "top_prediction": top_prediction, "title": page["title"], "distance": min_distance }
    
    @staticmethod
    def compute_metrics(gt, data, configs=['original', 'paraphrased']):
        # TODO: dynamic, field input_length in the result dataset is a representation of the input length
        # given to the model, not the length of the input text
        gt_with_mask = {}
        # only keep ground truth entries with a mask, as only those were predicted
        gt_with_mask['original'] = gt
        gt_with_mask['paraphrased'] = gt
        for _key, models in data.items():
            print("Processing results for: " + _key)
            for _model_name, model in models.items():
                for config in configs:
                    dataset = model[config]['train']
                    gt = gt_with_mask[config]
                    ## add ground truth label to each prediction
                    # some legacy examples contain predictions for examples which do not
                    # contain a mask, the prediction therefore cannot be correct. remove those
                    # in case any exist.
                    ids = set(dataset['page_id'])
                    # dataset = dataset.filter(lambda x: x['page_id'] in ids)
                    gt = gt.filter(lambda x: x['id'] in ids)
                    # make sure only the examples which were actually predicted
                    mappable = dataset.add_column("title", gt["title"])                    
                    # compute precision
                    k_runs = sum(key.startswith("prediction_") for key in mappable.column_names)
                    if k_runs == 0:
                        # legacy result format, only one prediction per example with key 'prediction'.
                        # add a key 'prediction_0' to each example with the same value as 'prediction'
                        mappable = mappable.map(lambda x: {**x, "prediction_0": x["prediction"]})
                        k_runs = 1
                    computed = mappable.map(TopKPredictionEvaluator.compute_metrics_for_page, num_proc=8, remove_columns=mappable.column_names, fn_kwargs={'k_runs': k_runs})
                    # compute metrics over computed results
                    correct_predictions = computed.filter(lambda x: x['correct'] == 1)
                    incorrect_predictions = computed.filter(lambda x: x['correct'] == 0)
                    correct = len(correct_predictions)
                    # take the average of the column 'distance' to get the average levenshtein distance
                    if correct == 0:
                        average_precision = 0
                        accuracy = 0
                    else:
                        average_precision = sum(correct_predictions['distance']) / correct
                        accuracy = correct / len(dataset)
                    
                    model[config]["result"] = {}
                    model[config]["result"]["data"] = computed
                    model[config]["result"]["accuracy"] = accuracy
                    model[config]["result"]["precision"] = average_precision
                    model[config]["result"]["correct_predictions"] = correct_predictions
                    model[config]["result"]["incorrect_predictions"] = incorrect_predictions
        return data

def main():
    assert len(sys.argv) > 1, "Please provide a path to the result file as an argument"

    key = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else None

    # change this if only one config was used
    configs = ['paraphrased', 'original']

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
    results = TopKPredictionEvaluator.compute_metrics(gt, results, configs)

    # print results
    csv_lines = ['model,size,config,accuracy,precision,best_prediction']
    json_results = {'key': key}
    for model_class, models in results.items():
        for m_name, model in models.items():
            name = f"{model_class}-{m_name}"
            json_results[name] = { "size": model['size'] }
            for config in configs:
                print(f"Model: {name:<15} Config: {config}")
                print(f"Accuracy: {round(model[config]['result']['accuracy'], 2)}")
                print(f"Precision: {round(model[config]['result']['precision'], 2)}")
                json_results[name][config] = {}
                json_results[name][config]['accuracy'] = model[config]['result']['accuracy']
                json_results[name][config]['precision'] = model[config]['result']['precision']
                print(model[config]['result']['correct_predictions']['distance'][:20])
                print(model[config]['result']['incorrect_predictions']['prediction'][:20])
                # json_results[name][config]['min_distance'] = min(model[config]['result']['correct_predictions']['distance'])
                csv_lines.append(f"{name},{model['size']},{config},{model[config]['result']['accuracy']},{model[config]['result']['precision']},{model[config]['result']['correct_predictions']['top_prediction']}")

    # write results to json file
    file_name = f"{key}-results" if model_name is None else f"{key}-{model_name}-results"
    save_path = f"evaluation/results/{file_name}"
    print(f"Writing results to {save_path}")

    with open(f"{save_path}.json", 'w') as f:
        json.dump(json_results, f, indent=4)
    
    # write results to csv file
    with open(f"{save_path}.csv", 'w') as f:
        f.write('\n'.join(csv_lines))



if __name__ == "__main__":
    main()