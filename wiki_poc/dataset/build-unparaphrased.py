## uses the raw wiki-dataset to build a csv with unparaphrased sentences as wiki text

import nltk
import pandas as pd
import logging
import os

input_filepath = 'csv/wiki-dataset-raw.csv'
output_filepath = 'csv/wiki-dataset-unparaphrased.csv'
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
    if os.path.exists(output_filepath):
        logging.warning("Output file already exists, skipping `build-unparaphrased.py`")
        quit()
    if not os.path.exists(input_filepath):
        logging.error("Input file does not exist. Please run `download_wiki.py` first.")
        quit()

    # read in the wiki-dataset
    dataset = pd.read_csv(input_filepath)
  
    # for each wiki-text entry, split it into sentences
    sentences = []
    for index, page in dataset.iterrows():
        # split the wiki-dataset into sentences
        sentences.append(split_to_sentences(page['text'][:sizelimit]))
    
    dataset['sentences'] = sentences
    dataset.drop(columns=['text'], inplace=True)
    save_to_csv(dataset, output_filepath)
    logging.info("Saved unparaphrased wiki-dataset to {}".format(output_filepath))


    