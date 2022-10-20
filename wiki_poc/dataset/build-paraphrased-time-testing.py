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
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
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

    return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]


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
    # iterate over all wiki pages
    for index, page in dataset.iterrows():

        # # skip page if already processed
        # if (not pd.isnull(page['paraphrased_sentences'])):
        #     logging.info('Skipping page # ' + str(index) + ", already processed.")
        #     continue

        logging.info("Processing page " + str(index) + "/" + str(dataset.shape[0]))
        print("Processing page " + str(index) + "/" + str(dataset.shape[0]))

        # start time
        start = time.time()

        paraphrase_sentences = []
        # iterate over all sentences in the wiki-page
        nb_sentences = len(page['sentences'])
        for sindex, sentence in enumerate(page['sentences']):
            logging.info("processing sentence " + str(sindex) + "/" + str(nb_sentences))
            print("processing sentence " + str(sindex) + "/" + str(nb_sentences))
            paraphrase_sentences.append(paraphrase_sentence(sentence))

        # append paraphrased sentences to dataset
        dataset.at[index, 'paraphrased_sentences'] = paraphrase_sentences

        # end time
        end = time.time()
        averagePageTimes.append(end - start)
        # print average time per page
        print("Average time per page: " + str(sum(averagePageTimes) / len(averagePageTimes)))

        # intermediate saving after every 5th page
        if (index % 5 == 0):
            save_to_csv(dataset, dataset_file)
            logging.info("Saved unparaphrased wiki-dataset to {} at page {}".format(dataset_file, index))

    save_to_csv(dataset, dataset_file)
    logging.info("Saved unparaphrased wiki-dataset to {}".format(dataset_file))
