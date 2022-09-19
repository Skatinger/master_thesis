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
import logging
# instantiate transformer model
from transformers import pipeline


torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# use a tokenizer to split the wikipedia text into sentences
# Use a entity-recognition model to quickly find entities instead of labelling them by hand
# this helps to mask the entities, making it easier to work automatically instead of doing it manually
# can consider doing it by hand later on for better precision
print("Loading Fill-Mask model")
fill_mask = pipeline("fill-mask", model="roberta-base", tokenizer='roberta-base', top_k=5)
mask_token = fill_mask.tokenizer.mask_token


if __name__ == '__main__':

    # Import Data from CSV
    file = 'wiki-dataset-masked.csv' # '/content/drive/MyDrive/wiki-dataset-reduced.csv'
    assert len(file) > 0, "Please provide a file path to the dataset"

    if not os.path.exists(file):
        logging.error("Input file does not exist. Please run `download-wiki.py` first.")
        quit()

    print("loading dataset")
    dataset = pd.read_csv(file)

    dataset['normal_predictions'] = ""
    dataset['paraphrased_predictions'] = ""
    # iterate over all the pages
    for index, page in dataset.iterrows():
        print("Now processing page:" + str(index))

        # iterate over snippets of 512 characters

        maxLength = max(len(page['normal_masked_text']), len(page['paraphrased_masked_text']))
        batchSize = 1024
        batchCount = int(maxLength / batchSize)

        dataset.at[index, 'normal_predictions'] = []
        dataset.at[index, 'paraphrased_predictions'] = []
        for i in range(0, batchSize):
            start, end = i * batchSize, i * batchSize + batchSize
            extract1 = page['normal_masked_text'][start:end]
            extract2 = page['paraphrased_masked_text'][start:end]

            # check if batches contain any mask token, otherwise skip
            if '<mask>' in extract1:
                dataset.at[index, 'normal_predictions'].append(fill_mask(extract1))
            if '<mask>' in extract2:
                dataset.at[index, 'paraphrased_predictions'].append(fill_mask(extract2))

    # save results
    dataset.to_csv('wiki-dataset-results.csv', index=False)
