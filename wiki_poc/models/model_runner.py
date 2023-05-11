import argparse
import importlib
import os
import logging
from datasets import load_dataset
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


def run_model(model_name, test_set): #, options):
    model_class = model_name.split("-")[0]
    print(model_class)
    print("MODEL NAME")
    print(model_name)

    # initilize runner for model class
    options = {}
    runner = runners()[model_class](model_name, test_set, options)
    runner.run_model()

def parse_options():
    parser = argparse.ArgumentParser(description="Run machine learning models with different configurations and options.")
    parser.add_argument("-m", "--model", help="Run a specific model. Format: model_name (e.g., bloomz)", type=str)
    parser.add_argument("-c", "--model-class", help="Run all models of a specific class. Format: model_class (e.g., bloomz-1b1)", type=str)
    parser.add_argument("-d", "--dry-run", help="Print out all models which would be run, but don't run them.")
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
    assert os.path.exists("test_set_ids.csv"), "test_set_ids.csv file not found. Please run generate_test_set_ids.py first." 
    # load full dataset
    dataset = load_dataset('Skatinger/wikipedia-persons-masked', split='train')
    # get set of page ids which are in the test_set_ids.csv file
    test_set_ids = set([i.strip() for i in open("test_set_ids.csv").readlines()])
    # filter out pages from dataset which are not in the test set
    dataset = dataset.filter(lambda x: x["id"] in test_set_ids, num_proc=8)
    return dataset

def check_model_exists(model_name):
    nested = [runner.names().keys() for runner in runners().values()]
    model_names = [item for sublist in nested for item in sublist]
    if model_name not in model_names:
        raise ValueError(f"Model {model_name} does not exist. ",
                         "Please choose one of the following models: ", model_names)

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
        model_names = list(runners()[model_class_to_run].names().values())
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
        model_names = list([runner.names().values() for runner in runners().values()][0])
        logging.info(f"Following models will be run:")
        for model in model_names:
            logging.info("  - %s", model)

        for model_name in model_names:
            run_model(model_name, test_set) #, model_size, options)

if __name__ == "__main__":
    main()
