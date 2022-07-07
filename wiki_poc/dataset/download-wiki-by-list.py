## helper script to download and save wikipedia dataset

import logging
logging.getLogger().setLevel(logging.INFO)
import os
from datasets import load_dataset
import csv

filepath = "wiki-dataset-large.csv"

# any persons you pass in for which the corresponding wiki page exists
import pandas as pd
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

# Extract the wiki article for every Wiki page title from the names list
# OUTPUT: articles list of format: [{id: dataset-id, text: wiki-text, title: wiki-page-title, url: link-to-wiki-page}, ...]
def extract_text(dataset):
    titles = dataset['title']

    # find the indices of each person
    indices = {}
    for name in persons:
        try:
            indices[name] = titles.index(name)
        except:
            logging.info("{} is not in the wikipedia dataset.".format(name))

    # find the corresponding articles (for every index of a known person create a list of their wiki pages)
    articles = []
    for name in indices.keys():
        articles.append( dataset[indices[name]] )
    # strip all new line characters for easier processing
    for article in articles:
        article['raw'] = article['text'].replace('\n', ' ')
        # don't need this anymore
        del article['text']
    
    return articles

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
    import pdb; pdb.set_trace()
    save_to_csv(articles)
