import pandas as pd
import seaborn as sns
import argparse
import logging

from evaluation.loader import ResultLoader as loader
from evaluation.single_prediction_eval import SinglePredictionEvaluator
from models.model_runner import load_test_set

# initializes loader class
loader = loader()

def parse_options():
    parser = argparse.ArgumentParser(description="build plots for given results")
    parser.add_argument("-k", "--key", help="Name of the results key", type=str)
    args = parser.parse_args()
    assert args.key is not None, "Key for result dataset required."
    return args.key



def parse_results(results):
    for result in results:


    for key, model in models.items():
    model['results'] = {}
    for config in ['original', 'paraphrased']:
        ev = SinglePredictionEvaluator(models[key][config], gt)
        accuracy, correct, incorrect, results = ev.compute_precision()

        res = {}
        res['all_results'] = results
        res['accuracy'] = accuracy
        res['correct'] = correct
        res['incorrect'] = incorrect
        res['levenstein_below_3_in_correct'] = correct.filter(lambda x: x['distance'] < 3)
        res['levenstein_below_3_in_incorrect'] = incorrect.filter(lambda x: x['distance'] < 3)
        if len(correct) < 1:
            res['average_levenstein_correct'] = 16
        else:
            res['average_levenstein_correct'] = sum(correct['distance']) / len(correct)
        res['average_levenstein_incorrect'] = sum(incorrect['distance']) / len(incorrect)
        
        model['results'][config] = res



def main():
    key = parse_options()

    ground_truth = load_test_set()
    results = loader.load(key)






if __name__ == "__main__":
    main()
