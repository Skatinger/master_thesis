# DOC
# 5) predicts the masked entities with a BERT model
# 6) computes the accuracy of the predictions as a percentage of correctly predicted masks

# required python packages:
# - transformers
# - torch
# - pandas
# - re
# - csv

import torch
import pandas as pd
import os
# sigint handler
import signal
import sys
import logging
# instantiate transformer model
from transformers import pipeline

logging.getLogger().setLevel(logging.INFO)

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset_file = 'wiki-dataset-masked.csv'


# allow signal handling
def signal_handler(sig, frame):
    logging.info("Received SIGINT, saving checkpoint")
    global dataset
    dataset.to_csv(dataset_file, index=False)
    logging.info("exiting")
    sys.exit(0)


signal.signal(signal.SIGTERM, signal_handler)


# use a tokenizer to split the wikipedia text into sentences
# Use a entity-recognition model to quickly find entities instead of labelling them by hand
# this helps to mask the entities, making it easier to work automatically instead of doing it manually
# can consider doing it by hand later on for better precision
print("Loading Fill-Mask model")
fill_mask = pipeline("fill-mask", model="roberta-base", tokenizer='roberta-base', top_k=5)
mask_token = fill_mask.tokenizer.mask_token


if __name__ == '__main__':

    # Import Data from CSV
    file = 'wiki-dataset-masked.csv'
    assert len(file) > 0, "Please provide a file path to the dataset"

    if not os.path.exists(file):
        logging.error("Input file does not exist. Please run `build-masked.py` first.")
        quit()

    print("loading dataset")
    dataset = pd.read_csv(file)

    if 'normal_predictions' not in dataset.columns:
        dataset['normal_predictions'] = ""
        dataset['paraphrased_predictions'] = ""

    # iterate over all the pages
    for index, page in dataset.iterrows():

        # skip iteration if value already present
        # value is '' if column newly added, float:nan if resumed
        if (isinstance(page['normal_predictions'], str) and len(page['normal_predictions']) > 0):
            logging.info("Skipping page {}, already done.".format(index))
            continue

        print("Now processing page:" + str(index))

        inputSize = 1024

        dataset.at[index, 'normal_predictions'] = []
        dataset.at[index, 'paraphrased_predictions'] = []
        for i in range(0, inputSize):
            start, end = i * inputSize, i * inputSize + inputSize
            extract1 = page['normal_masked_text'][start:end]
            extract2 = page['paraphrased_masked_text'][start:end]

            # check if batches contain any mask token, otherwise skip
            # ditch this prediction if too many tokens, as usually too many tokens mean there is some foreign
            # language involved or too many special characters, which results in almost all characters being a single
            # token, and the model not being able to predict a useful fill-mask word anyway
            try:
                if '<mask>' in extract1:
                    dataset.at[index, 'normal_predictions'].append(fill_mask(extract1))
                if '<mask>' in extract2:
                    dataset.at[index, 'paraphrased_predictions'].append(fill_mask(extract2))
            except RuntimeError as e:
                # if we had too many tokens no worries, just skip this batch
                if 'expanded size of the tensor' in str(e):
                    logging.warn("Tensor was too long for index {} at {}:{}".format(index, start,  end))
                else:
                    raise e

        if (index % 5 == 0):
            logging.info("Checkpointing at page {}".format(index))
            dataset.to_csv(dataset_file, index=False)

    # save results
    dataset.to_csv('wiki-dataset-results.csv', index=False)
