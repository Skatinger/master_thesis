import argparse
import importlib
import os
import logging
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)


# from runnables.bloomz.bloomz_base import BloomzRunner
from runnables.cerebras.cerebras_base import CerebrasRunner

def runners():
    return {
        # "bloomz": BloomzRunner,
        "cerebras": CerebrasRunner
    }


def run_model(model_name, test_set): #, options):
    # Customize this function to run your model with the given options
    # print(f"Running {model_name}_{model_size}B with options: {options}")

    model_class = model_name.split("/")[0]
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
    parser.add_argument("-d", "--dry-run", help="Print out all models which would be run, but don't run them.")
    parser.add_argument("-s", "--size", help="Run all models with a the same size. Options: T, XS, S, M, L, XL Format: model_size (e.g., 5b)", type=str)
    parser.add_argument("-o", "--options", help="Specify options for the model. Format: option1=value1,option2=value2", type=str)

    args = parser.parse_args()
    options = {}
    if args.options:
        for option in args.options.split(","):
            key, value = option.split("=")
            options[key] = value

    return args.model, options

def load_test_set():
    assert os.path.exists("test_set_ids.csv"), "test_set_ids.csv file not found. Please run generate_test_set_ids.py first." 
    # load full dataset
    dataset = load_dataset('Skatinger/wikipedia-persons-masked', split='train')
    # get set of page ids which are in the test_set_ids.csv file
    test_set_ids = set([i.strip() for i in open("test_set_ids.csv").readlines()])
    # filter out pages from dataset which are not in the test set
    dataset = dataset.filter(lambda x: x["id"] in test_set_ids, num_proc=8)
    return dataset


def main():
    # load the test set of pages
    test_set = []# load_test_set()

    model_to_run, options = parse_options()
    models_dir = "runnables"

    print("got here")
    print(model_to_run)
    print(options)

    if model_to_run:
        model_name, model_size = model_to_run.split("_")
        run_model(model_name, model_size, test_set, options)
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
