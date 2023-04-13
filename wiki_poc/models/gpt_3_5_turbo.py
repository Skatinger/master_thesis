import os
import openai
from datasets import load_dataset
from datasets import Dataset
import logging
from tqdm.auto import tqdm
import time
openai.organization = "org-JkswNfkhKMfjPPgyLUjElGPH"
openai.api_key = os.getenv("OPENAI_API_KEY")


"""
This script is used to test the GPT-3.5-turbo model on the wikipedia dataset.
Mask-Filling is not possible, so we use the ChatCompletion API to prompt for the name referred to as <mask>.

Unlike other models, gpt-3.5-turbo usually detects the person best using the first few paragraphs of the article.
Therefore we only pass in the first sequence of the article to the model. If it does not recognize the person,
we pass in the first 500 characters of the second sequence as well.

TODO: check where the cutoff is for the model to recognize the person, e.g. how many characters are needed and
at what point is it useless to send any more characters to the model. This improves cost efficiency.
"""


if __name__ == "__main__":

    # ensure required test ids file exists
    assert os.path.exists("test_set_ids.csv"), "test_set_ids.csv file not found. Please run generate_test_set_ids.py first."
    # ensure api key is set
    assert openai.api_key is not None, "OPENAI_API_KEY environment variable not set."

    CONFIG = "original"
    MODEL_NAME = "gpt-3.5-turbo"
    PATH = f"wiki_predictions_{MODEL_NAME.replace('/', '_')}_{CONFIG}.jsonl"

    user_prompt = "Who is the person refered to as <mask>? Only give the exact name without punctuation. You are not allowed to respond with anything but the name, no more than 3 words."

    # dataset with initial pages
    dataset = load_dataset('Skatinger/wikipedia-persons-masked', CONFIG, split='train')
    # get set of page ids which are in the test_set_ids.csv file
    test_set_ids = set([i.strip() for i in open("test_set_ids.csv").readlines()])
    # filter out pages from dataset which are not in the test set
    dataset = dataset.filter(lambda x: x["id"] in test_set_ids)

    # only process pages which have not been processed yet
    if os.path.exists(PATH):
        # store already processed results in result_dataset
        result_dataset = Dataset.from_json(PATH)
        # get set of page ids which have already been processed
        processed_ids = set(result_dataset['page_id'])
        # filter out pages from dataset which have already been processed
        dataset = dataset.filter(lambda x: x["id"] not in processed_ids)
    
    else:
        result_dataset = Dataset.from_dict({'prediction': [], 'page_id': [], 'input_length': []})

    # iterate over pages in dataset
    for index, page in enumerate(tqdm(dataset)):
        
        # extract text from page
        text = page[f"masked_text_{CONFIG}"][:1000]
        # prompt openai api for prediction
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                { "role": "user", "content": text + " " + user_prompt },
            ],
            temperature=0.5,
            max_tokens=10,
            top_p=1,
            n=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n"]
        )
        # add prediction to result dataset
        result_dataset = result_dataset.add_item(
            {'prediction': response['choices'][0]['message']['content'],
             'page_id': page['id'], 'input_length': len(text)})
    
        # periodically save file
        if index % 100 == 0:
            logging.info('Saving dataset intermediately to path %s', PATH)
            result_dataset.to_json(PATH)
        
        # sleep for 5 seconds to avoid rate limit
        time.sleep(5)

    # save dataset
    logging.info('Saving dataset to path %s', PATH)
    result_dataset.to_json(PATH)