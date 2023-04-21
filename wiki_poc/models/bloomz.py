from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import signal
import logging
import os
import torch
from datasets import load_dataset, Dataset

logging.getLogger().setLevel(logging.INFO)

def signal_handler(_sig, _frame):
    """allow signal handling, required when script is interrupted either with
        ctrl+c or with a proper sigint by the job handling server. All processing is cached to disk,
        this is just to ensure the job exits with a clean exit code and writes a short log to
        more easily see the reason for the exit upon log inspection.
    """
    logging.info("Received SIGINT, exiting.")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)


# use CUDA if available (ususally has ID 0), -1 is huggingface default for CPU
DEVICE = 0 if torch.cuda.is_available() else -1


checkpoint = "bigscience/bloomz"

if __name__ == "__main__":

    # ensure required test ids file exists
    assert os.path.exists("test_set_ids.csv"), "test_set_ids.csv file not found. Please run generate_test_set_ids.py first."

    CONFIG = "paraphrased"
    MODEL_NAME = "bloomz"
    PATH = f"wiki_predictions_{MODEL_NAME.replace('/', '_')}_{CONFIG}.jsonl"

    # dataset with initial pages
    dataset = load_dataset('Skatinger/wikipedia-persons-masked', CONFIG, split='train')
    # get set of page ids which are in the test_set_ids.csv file
    test_set_ids = set([i.strip() for i in open("test_set_ids.csv").readlines()])
    # filter out pages from dataset which are not in the test set
    dataset = dataset.filter(lambda x: x["id"] in test_set_ids)

    # # only process pages which have not been processed yet
    # if os.path.exists(PATH):
    #     # store already processed results in result_dataset
    #     result_dataset = Dataset.from_json(PATH)
    #     # get set of page ids which have already been processed
    #     processed_ids = set(result_dataset['page_id'])
    #     # filter out pages from dataset which have already been processed
    #     dataset = dataset.filter(lambda x: x["id"] not in processed_ids)
    
    # else:
    #     result_dataset = Dataset.from_dict({'prediction': [], 'page_id': [], 'input_length': []})

    # initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    


    inputs = tokenizer.encode("Translate to English: Je t'aime.", return_tensors="pt")
    outputs = model.generate(inputs)
    print(tokenizer.decode(outputs[0]))
