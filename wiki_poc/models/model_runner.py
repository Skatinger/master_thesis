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
from .runners.mpt_instruct.mpt_instruct_runner import MPTInstructRunner
from .runners.incite_instruct.incite_instruct_runner import InciteInstructRunner
from .runners.flan_t5.flan_t5_runner import FlanT5Runner
from .runners.falcon.falcon_instruct_runner import FalconInstructRunner
from .runners.falcon.falcon_runner import FalconRunner
from .runners.gpt.gpt_j_runner import GPTJRunner
from .runners.gpt.gpt_neo_x_runner import GPTNeoXRunner
from .runners.llama.huggy_llama_runner import HuggyLlamaRunner
from .runners.distilbert.distilbert_runner import DistilbertRunner
from .runners.distilbert.distilbert_qa_runner import DistilbertQARunner
from .runners.deberta.deberta_runner import DebertaRunner
from .runners.deberta.deberta_qa_runner import DebertaQARunner
from .runners.roberta.roberta_qa_runner import RobertaQARunner


def runners():
    return {
        "bloomz": BloomzRunner,
        "cerebras": CerebrasRunner,
        "roberta": RobertaRunner,
        "t5": T5Runner,
        "pythia": PythiaRunner,
        "mpt_instruct": MPTInstructRunner,
        "incite_instruct": InciteInstructRunner,
        "flan_t5": FlanT5Runner,
        "falcon_instruct": FalconInstructRunner,
        "gptj": GPTJRunner,
        "gpt_neox": GPTNeoXRunner,
        "llama": HuggyLlamaRunner,
        "distilbert": DistilbertRunner,
        "deberta": DebertaRunner,
        "falcon": FalconRunner,
        "distilbert_qa": DistilbertQARunner,
        "mdeberta_v3": DebertaQARunner,
        "deberta_v3": DebertaQARunner,
        "roberta_squad": RobertaQARunner,
    }

def run_model(model_name, test_set, options):
    model_class = model_name.split("-")[0]
    # initilize runner for model class
    runner = runners()[model_class](model_name, test_set, options)
    runner.run_model()

def parse_options():
    description = """Run machine learning models with different configurations and options.
                     Options are:
                          - paraphrased (run on paraphrased dataset)
                          - original (run on original dataset)
                    
                     Examples:
                        Run all models on paraphrased and original dataset:
                            python -m wiki_poc.models.model_runner
                        Run model class on paraphrased dataset:
                            python -m wiki_poc.models.model_runner -c bloomz paraphrased
                        Run specific model sizes:
                            python -m wiki_poc.models.model_runner -s XS
                        Run specific model type and exlude some:
                            python -m wiki_poc.models.model_runner -c bloomz -e bloomz-1b1
                        
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-m", "--model", help="Run a specific model. Format: model_name (e.g., bloomz)", type=str)
    parser.add_argument("-c", "--model-class", help="Run all models of a specific class. Format: model_class (e.g., bloomz-1b1)", type=str)
    parser.add_argument("-d", "--device", help="Device to use, by default using GPU 0.", type=str)
    parser.add_argument("-dr", "--dry-run", help="Run with only 10 examples for each model.", action="store_true")
    parser.add_argument("-e", "--exclude", help="Exclude specific models from the run. Format: model_name1,model_name2", type=str)
    parser.add_argument("-tk", "--top-k", help="top k predictions for each model. k=5 takes 5x longer than k=1.", type=int)
    parser.add_argument("-sm", "--save-memory", help="pass to use only a portion of memory in case not a full 80GB are available.", action="store_true")
    parser.add_argument("-nc", "--no-cache", help="Don't use cached results, run all models again.")
    keyhelp = """Specify a key to identify the run. This key will be used to save results and to load them again.
               If no key is specified, a new key will be generated for each run."""
    parser.add_argument("-k", "--key", help=keyhelp, type=str)
    parser.add_argument("-f", "--fast", help="Run only a subset of examples for each model.", action="store_true")
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
    if not args.top_k:
        args.top_k = 1

    if args.model:
        models_list = args.model.split(",")
    else:
        models_list = []

    return models_list, args.size, args.model_class, args.key, args.exclude, args.device, args.save_memory, \
           args.top_k, args.fast, args.dry_run, options

def load_test_set(path = "models/cache/reduced_test_set", ids_file_path = "test_set_ids.csv"):
    """load test dataset from cache or generates it from the full dataset and caches it
    Args:
        path (str, optional): path to cache. Defaults to "models/cache/reduced_test_set".
        ids_file_path (str, optional): path to file with page ids of test set. Defaults to "test_set_ids.csv".
    """
    # load cached dataset if it exists
    if os.path.exists(path):
        dataset = Dataset.load_from_disk(path)
    else:
        assert os.path.exists(ids_file_path), f"{ids_file_path} file not found. Please run generate_test_set_ids.py first."
        logging.info("No cached test dataset found, generating it from full dataset.")
        # load full dataset
        dataset = load_dataset('Skatinger/wikipedia-persons-masked', split='train')
        # get set of page ids which are in the test_set_ids.csv file
        test_set_ids = set([i.strip() for i in open(ids_file_path).readlines()])
        # filter out pages from dataset which are not in the test set
        dataset = dataset.filter(lambda x: x["id"] in test_set_ids, num_proc=8)
        # save dataset to cache
        dataset.save_to_disk(path)
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
    models_to_run, model_size_to_run, model_class_to_run, key, excluded, device, save_memory, top_k, fast, dry_run, options = parse_options()
    if len(models_to_run) > 0:
        for model in models_to_run:
            check_model_exists(model)
    
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
    if device is None:
        options["device"] = 0
    else:
        options["device"] = device
    logging.info(f"Using cache key {key}")
    
    if save_memory:
        options["save_memory"] = True
    else:
        options["save_memory"] = False
    
    options["k_runs"] = top_k

    
    # create folder for run
    os.makedirs(f"results/{key}", exist_ok=True)

    # load the test set of pages
    if "ids_file_path" in options.keys():
        test_set = load_test_set(ids_file_path=options["ids_file_path"])
    else:
        test_set = load_test_set()
    # only select a range if specified
    if fast:
        test_set = test_set.select(range(100))
    elif dry_run:
        test_set = test_set.select(range(10))

    # run specific models
    if len(models_to_run) > 0:
        logging.info(f"Following models will be run:")
        for model in models_to_run:
            logging.info("  - %s", model)

        for model_name in models_to_run:
            run_model(model_name, test_set, options)
    
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
