# uses the raw wiki-dataset to build a csv with paraphrased sentences as wiki text
# uses the model tuner007/pegasus_paraphrase to paraphrase the wiki-dataset
# runs on CUDA if possible
# paraphrases each sentence seperately, to keep the length of the text, and only scramble the words a bit
# checkpoints are saved to the dataset file every 5th processed page

from ast import literal_eval
import faulthandler
import functools
import pandas as pd
import numpy as np
import logging
import time
import os
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

# sigint handler
import signal
import sys


# allow signal handling
def signal_handler(sig, frame):
    logging.info("Received SIGINT, saving checkpoint")
    global dataset
    save_to_csv(dataset, dataset_file)
    logging.info("exiting")
    sys.exit(0)


signal.signal(signal.SIGTERM, signal_handler)

# ensure error stack is printed when an error occurs on the GPU / Computing Cluster
faulthandler.enable()
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info("Using device: " + torch_device)

dataset_file = 'wiki-dataset.csv'
# maximum text length before paraphrasing
sizelimit = 4096


def save_to_csv(df, filepath):
    df.to_csv(filepath, index=False)


# ensure model is loaded only once
@functools.lru_cache(maxsize=1)
def load_model(model_name='tuner007/pegasus_paraphrase'):
    print(f"Loading {model_name}")
    # sadly there is no fast version of this tokenizer, so we use the one from the base pegasus model
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    # tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum', use_fast=True)
    logging.info("Using {} device".format(torch_device))
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
    return model, tokenizer


def paraphrase_sentence(input_texts, num_return_sequences=1, num_beams=10, temperature=1.5):
    model, tokenizer = load_model()
    batch = tokenizer(input_texts,
                      truncation=True,
                      padding='longest',
                      return_tensors="pt"
                      ).to(torch_device)

    translated = model.generate(**batch,
                                num_beams=num_beams,
                                num_return_sequences=num_return_sequences,
                                temperature=temperature)

    return tokenizer.batch_decode(translated, skip_special_tokens=True)


if __name__ == '__main__':
    if not os.path.exists(dataset_file):
        logging.error("Input file does not exist. Please run `download-wiki.py` first.")
        quit()

    # read in the wiki-dataset
    dataset = pd.read_csv(dataset_file)
    if 'sentences' not in dataset.columns:
        logging.error("Input file does not contain a 'sentences' column. Please run `build-unparaphrased.py` first.")
        quit()

    # parse sentences string back to list of sentences
    dataset['sentences'] = dataset['sentences'].apply(literal_eval)

    # add empty column for new paraphrased sentences if not yet present
    if ('paraphrased_sentences' not in dataset.columns):
        dataset['paraphrased_sentences'] = np.nan
        # cast to object to allow lists as values
        dataset['paraphrased_sentences'] = dataset['paraphrased_sentences'].astype('object')

    # paraphrase the wiki-dataset
    paraphrased_texts = []

    averagePageTimes = []
    # averageProcessingTime = 0
    allStart = time.time()
    # NEW APPROACH:
    # split dataset into chunks of 250 pages
    # paraphrase batches of 250 pages
    splitSize = 100
    batches = np.split(dataset, [i for i in range(splitSize, dataset.shape[0], splitSize)])

    for batch in batches:
        # save number of sentences for each page
        start = time.time()
        sentencesCounts = batch['sentences'].apply(len)

        # flatten list of sentences
        sentences = np.concatenate(batch['sentences'].values)

        # compute preprocessing time
        otherProcessingTime = time.time() - start

        # paraphrase sentences
        paraphrased_sentences = paraphrase_sentence(sentences)
        end = time.time()

        # split paraphrased sentences back into pages
        batch['paraphrased_sentences'] = np.split(paraphrased_sentences, sentencesCounts.cumsum()[:-1])

        # compute preprocessing time
        otherProcessingTime += time.time() - end
        print("otherProcessingTime: " + str(otherProcessingTime))
        averagePageTimes.append((end - start) / splitSize)
        print("averagePageTime: " + str(averagePageTimes[-1]))

