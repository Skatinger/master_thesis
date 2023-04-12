import os
import sys
import torch
from tqdm import tqdm
import signal
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers.pipelines.pt_utils import KeyDataset
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


if __name__ == "__main__":

    MODEL_NAME = "cerebras/Cerebras-GPT-13B"
    DEVICE = 0 if torch.cuda.is_available() else -1
    
    CONFIG = "paraphrased"
    MODEL_NAME_SHORT = "cerebras"
    PATH = f"wiki_predictions_{MODEL_NAME_SHORT}_{CONFIG}.jsonl"

    # dataset with initial pages
    dataset = load_dataset('Skatinger/wikipedia-persons-masked', CONFIG, split='train')
    # get set of page ids which are in the test_set_ids.csv file
    test_set_ids = set([i.strip() for i in open("test_set_ids.csv").readlines()])
    # filter out pages from dataset which are not in the test set
    dataset = dataset.filter(lambda x: x["id"] in test_set_ids)

    # only process pages which have not been processed yet, load processed pages from jsonl file if exists
    if os.path.exists(PATH):
        # store already processed results in result_dataset
        result_dataset = Dataset.from_json(PATH)
        # get set of page ids which have already been processed
        processed_ids = set(result_dataset['page_id'])
        # filter out pages from dataset which have already been processed
        dataset = dataset.filter(lambda x: x["id"] not in processed_ids)
    else:
        result_dataset = Dataset.from_dict({'prediction': [], 'page_id': [], 'input_length': []})

    # load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id

    # prompts
    start_prompt = "The following text talks about a person but the person is referred to as <mask>.\n\n"
    end_prompt = "\n\nThe name of the person in the text referred to as <mask> is: "

    gen = pipe(KeyDataset(dataset, 'text'), batch_size=16, max_new_tokens=5, early_stopping=True)
    for example, out in zip(dataset, tqdm(gen, total=len(dataset))):
        # get page id
        page_id = example['id']
        # get input length
        input_length = example['input_length']
        # get prediction
        print(out)
        print(out[0]['generated_text'])
        prediction = out[0]['generated_text'].split(end_prompt)[0].replace(start_prompt, "")
        print(prediction)
        # append results to result_dataset
        result_dataset.add_item({'prediction': prediction, 'page_id': page_id, 'input_length': input_length})
    
    # save final dataset to file
    logging.info('Saving final dataset to path %s', PATH)
    result_dataset.save_to_disk(PATH)


    # text = "Generative AI is "

    # inputs = tokenizer(text, return_tensors="pt")
    # outputs = model.generate(**inputs, num_beams=5, 
    #                         max_new_tokens=10, early_stopping=True,
    #                         no_repeat_ngram_size=2)
    # text_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # print(text_output[0])
