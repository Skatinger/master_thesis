# helper script to download and save wikipedia dataset

import logging
import os
from datasets import load_dataset
import csv
import pandas as pd
from custom.wiki import extract_text
logging.getLogger().setLevel(logging.INFO)

filepath = "wiki-dataset.csv"

# any persons you pass in for which the corresponding wiki page exists
df = pd.read_csv("query.csv")
persons = list(df['page_titleEN'])


# loads the wikipedia dataset from huggingface if it does not yet exist
def load_wiki_dataset():
    if os.path.exists(filepath):
        logging.warning("Dataset already exists at " + filepath + " Delete to re-download.")
        quit()
    logging.info('Loading dataset...')
    try:
        return load_dataset("wikipedia", "20220301.en", split="train")
    except ValueError as err:
        logging.warning("Specified dataset not available, choose a current dataset:")
        logging.warning(err)
        quit()


def save_to_csv(articles):
    csv_columns = ['id', 'raw', 'title', 'url']
    with open(filepath, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in articles:
            writer.writerow(data)
    logging.info("Saved articles to {}".format(filepath))


if __name__ == '__main__':
    dataset = load_wiki_dataset()
    articles = extract_text(dataset)
    save_to_csv(articles)
