import argparse
import importlib
import os
import logging
from datasets import load_dataset, Dataset
logging.basicConfig(level=logging.INFO)

# TODOS:
# - allow passing a key specifying where to save results, to identify the run
# - allow passing a key specifying where to load results from, to continue a run
# - add checkpointing for longer processing of single models


from runnables.bloomz.bloomz_base import BloomzRunner
from runnables.cerebras.cerebras_base import CerebrasRunner

def runners():
    return {
        "bloomz": BloomzRunner,
        "cerebras": CerebrasRunner
    }


def run_model(model_name, test_set):
    model_class = model_name.split("-")[0]
    # initilize runner for model class
    options = {}
    runner = runners()[model_class](model_name, test_set, options)
    # check cache for results
    if runner.results_exist():
        logging.info(f"Cache for {model_name} exists, skipping.")
    else:
        runner.run_model()

def parse_options():
    parser = argparse.ArgumentParser(description="Run machine learning models with different configurations and options.")
    parser.add_argument("-m", "--model", help="Run a specific model. Format: model_name (e.g., bloomz)", type=str)
    parser.add_argument("-c", "--model-class", help="Run all models of a specific class. Format: model_class (e.g., bloomz-1b1)", type=str)
    # parser.add_argument("-d", "--dry-run", help="Print out all models which would be run, but don't run them.")
    parser.add_argument("-nc", "--no-cache", help="Don't use cached results, run all models again.")
    parser.add_argument("-s", "--size", help="Run all models with a the same size. Options: T, XS, S, M, L, XL Format: model_size (e.g., 5b)", type=str)
    parser.add_argument("-o", "--options", help="Specify options for the model. Format: option1=value1,option2=value2", type=str)

    args = parser.parse_args()
    options = {}
    if args.options:
        for option in args.options.split(","):
            key, value = option.split("=")
            options[key] = value

    return args.model, args.size, args.model_class, options

def load_test_set():
    """load test dataset from cache or generates it from the full dataset and caches it"""
    # load cached dataset if it exists
    if os.path.exists("reduced_test_set"):
        dataset = Dataset.load_from_disk("reduced_test_set")
    else:
        assert os.path.exists("test_set_ids.csv"), "test_set_ids.csv file not found. Please run generate_test_set_ids.py first."
        logging.info("No cached test dataset found, generating it from full dataset.")
        # load full dataset
        dataset = load_dataset('Skatinger/wikipedia-persons-masked', split='train')
        # get set of page ids which are in the test_set_ids.csv file
        test_set_ids = set([i.strip() for i in open("test_set_ids.csv").readlines()])
        # filter out pages from dataset which are not in the test set
        dataset = dataset.filter(lambda x: x["id"] in test_set_ids, num_proc=8)
        # save dataset to cache
        dataset.save_to_disk("reduced_test_set")
    return dataset

def get_all_model_names(model_class=None):
    """returns a list of all names of available models, optionally filtered by model class"""
    if model_class:
        nested = [runner.names().keys() for runner in runners().values() if runner.__name__.lower().startswith(model_class)]
    else:
        nested = [runner.names().keys() for runner in runners().values()]
    return [item for sublist in nested for item in sublist]

def check_model_exists(model_name):
    if model_name not in get_all_model_names():
        raise ValueError(f"Model {model_name} does not exist. ",
                         "Please choose one of the following models: ", get_all_model_names())

def main():
    model_to_run, model_size_to_run, model_class_to_run, options = parse_options()
    if model_to_run:
        check_model_exists(model_to_run)

    # load the test set of pages
    test_set = load_test_set()

    # run a single model instance
    if model_to_run:
        # check that the model exists
        check_model_exists(model_to_run)
        logging.info(f"Running model {model_to_run}")
        run_model(model_to_run, test_set) # , model_size, test_set, options)
    
    # run all models of a specific class
    elif model_class_to_run:
        # retrieve all models of the specified class
        model_names = get_all_model_names(model_class_to_run)
        logging.info(f"Following models will be run:")
        for model in model_names:
            logging.info("  - %s", model)

        for model_name in model_names:
            # TODO: get prepared examples dataset and pass it to run_model,
            # so that it doesn't have to be loaded for each model, as prompts are the same for all models
            # of the same model class
            run_model(model_name, test_set)
    else:
        # retrieve all models of all runners
        model_names = get_all_model_names()
        logging.info(f"Following models will be run:")
        for model in model_names:
            logging.info("  - %s", model)

        for model_name in model_names:
            run_model(model_name, test_set) #, model_size, options)

if __name__ == "__main__":
    main()
