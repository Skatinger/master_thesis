## uses the raw wiki-dataset to build a csv with unparaphrased sentences as wiki text

import nltk
import pandas as pd
import logging
import os

dataset_file = 'wiki-dataset.csv'
sizelimit = 4096 # maximum text length

# use the SentencePiece model via nltk to split text into sentences
# https://arxiv.org/pdf/1808.06226.pdf
def split_to_sentences(text):
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logging.info("Downloading nltk punkt tokenizer")
        nltk.download('punkt')
    return nltk.tokenize.sent_tokenize(text, 'english')

def save_to_csv(df, filepath):
    df.to_csv(filepath, index=False)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    if not os.path.exists(dataset_file):
        logging.warning("Input file does not exist, run `download-wiki.py`")
        quit()

    # read in the wiki-dataset
    dataset = pd.read_csv(dataset_file)
  
    # for each wiki-text entry, split it into sentences
    sentences = []
    for index, page in dataset.iterrows():
        # split the wiki-dataset into sentences
        sentences.append(split_to_sentences(page['raw'][:sizelimit]))
    
    dataset['sentences'] = sentences
    save_to_csv(dataset, dataset_file)
    logging.info("Saved unparaphrased wiki-dataset to {}".format(dataset_file))


    