import argparse
import os
import logging
from datetime import datetime
from datasets import load_dataset, Dataset
logging.basicConfig(level=logging.INFO)

# TODOS:
# - add checkpointing for longer processing of single models


from .runners.bloomz.bloomz_runner import BloomzRunner
from .runners.cerebras.cerebras_runner import CerebrasRunner
from .runners.roberta_runner import RobertaRunner
from .runners.t5.t5_runner import T5Runner
from .runners.pythia.pythia_runner import PythiaRunner

def runners():
    return {
        "bloomz": BloomzRunner,
        "cerebras": CerebrasRunner,
        "roberta": RobertaRunner,
        "t5": T5Runner,
        "pythia": PythiaRunner,
    }

def run_model(model_name, test_set, options):
    model_class = model_name.split("-")[0]
    # initilize runner for model class
    runner = runners()[model_class](model_name, test_set, options)
    runner.run_model()

def parse_options():
    parser = argparse.ArgumentParser(description="Run machine learning models with different configurations and options.")
    parser.add_argument("-m", "--model", help="Run a specific model. Format: model_name (e.g., bloomz)", type=str)
    parser.add_argument("-c", "--model-class", help="Run all models of a specific class. Format: model_class (e.g., bloomz-1b1)", type=str)
    # parser.add_argument("-d", "--dry-run", help="Print out all models which would be run, but don't run them.")
    parser.add_argument("-e", "--exclude", help="Exclude specific models from the run. Format: model_name1,model_name2", type=str)
    parser.add_argument("-nc", "--no-cache", help="Don't use cached results, run all models again.")
    keyhelp = """Specify a key to identify the run. This key will be used to save results and to load them again.
               If no key is specified, a new key will be generated for each run."""
    parser.add_argument("-k", "--key", help=keyhelp, type=str)
    parser.add_argument("-s", "--size", help="Run all models with a the same size. Options: T, XS, S, M, L, XL Format: model_size (e.g., 5b)", type=str)
    parser.add_argument("-o", "--options", help="Specify options for the model. Format: option1=value1,option2=value2", type=str)

    args = parser.parse_args()
    options = {}
    if args.options:
        for option in args.options.split(","):
            key, value = option.split("=")
            options[key] = value
    if args.exclude:
        args.exclude = args.exclude.split(",")
    else:
        args.exclude = []
    return args.model, args.size, args.model_class, args.key, args.exclude, options

def load_test_set():
    """load test dataset from cache or generates it from the full dataset and caches it"""
    # load cached dataset if it exists
    if os.path.exists("models/cache/reduced_test_set"):
        dataset = Dataset.load_from_disk("models/cache/reduced_test_set")
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
        dataset.save_to_disk("models/cache/reduced_test_set")
    return dataset

def get_models_by_size(model_size):
    """returns a list of all models of a specific size"""
    model_names = []
    sizes_per_model = [runner.sizes() for runner in runners().values()]
    for model in sizes_per_model:
        if model_size in model.keys():
            model_names.append(model[model_size])
    return model_names

def get_all_model_names(model_class=None, model_size=None):
    """returns a list of all names of available models, optionally filtered by model class"""
    if model_class and model_size:
        return [runners()[model_class].sizes()[model_size]]
    elif model_class:
        return list(runners()[model_class].names().keys())
    elif model_size:
        return get_models_by_size(model_size)
    else:
        nested = [runner.names().keys() for runner in runners().values()]
        return [item for sublist in nested for item in sublist]

def check_model_exists(model_name):
    if model_name not in get_all_model_names():
        raise ValueError(f"Model {model_name} does not exist. ",
                         "Please choose one of the following models: ", get_all_model_names())

def main():
    model_to_run, model_size_to_run, model_class_to_run, key, excluded, options = parse_options()
    if model_to_run:
        check_model_exists(model_to_run)
    
    if len(excluded) > 0:
        # check that all excluded models exist
        for model in excluded:
            if check_model_exists(model):
                raise ValueError(f"Model {model} does not exist. ",
                                "Please choose one of the following models: ", get_all_model_names())

    if key is None:
        # generate key from time and date
        key = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    options["key"] = key
    logging.info(f"Using cache key {key}")
    
    # create folder for run
    os.makedirs(f"results/{key}", exist_ok=True)

    # load the test set of pages
    test_set = load_test_set()

    # run a single model instance
    if model_to_run:
        # check that the model exists
        check_model_exists(model_to_run)
        logging.info(f"Running model {model_to_run}")
        run_model(model_to_run, test_set, options)
    
    # run all models of a specific class
    elif model_class_to_run:
        # retrieve all models of the specified class
        model_names = get_all_model_names(model_class=model_class_to_run, model_size=model_size_to_run)
        if len(excluded) > 0:
            # remove excluded models
            model_names = [model for model in model_names if model not in excluded]
        logging.info(f"Following models will be run:")
        for model in model_names:
            logging.info("  - %s", model)

        for model_name in model_names:
            # TODO: get prepared examples dataset and pass it to run_model,
            # so that it doesn't have to be loaded for each model, as prompts are the same for all models
            # of the same model class
            run_model(model_name, test_set, options)
    # run all models of a specific size
    elif model_size_to_run:
        # retrieve all models of the specified size
        model_names = get_all_model_names(model_size=model_size_to_run)
        if len(excluded) > 0:
            # remove excluded models
            model_names = [model for model in model_names if model not in excluded]
        logging.info(f"Following models will be run:")
        for model in model_names:
            logging.info("  - %s", model)

        for model_name in model_names:
            run_model(model_name, test_set, options)
    else:
        # retrieve all models of all runners
        model_names = get_all_model_names()
        if len(excluded) > 0:
            # remove excluded models
            model_names = [model for model in model_names if model not in excluded]
        logging.info(f"Following models will be run:")
        for model in model_names:
            logging.info("  - %s", model)

        for model_name in model_names:
            run_model(model_name, test_set, options)

if __name__ == "__main__":
    main()
