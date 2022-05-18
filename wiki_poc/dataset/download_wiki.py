## helper script to download and save wikipedia dataset

import logging
logging.getLogger().setLevel(logging.INFO)
import os
from datasets import load_dataset
import csv

filepath = "wiki-dataset-raw.csv"

# top 50 people from wikipedia 2010-2020
top_50_persons = ["Donald Trump", "Barack Obama", "Elizabeth II", "Michael Jackson", "Cristiano Ronaldo",
    "Lady Gaga", "Adolf Hitler", "Eminem", "Justin Bieber", "Elon Musk", "Freddie Mercury", "Lionel Messi",
    "Kim Kardashian", "Steve Jobs", "Michael Jordan", "Dwayne Johnson", "Stephen Hawking", "Taylor Swift",
    "Miley Cyrus", "Abraham Lincoln", "Johnny Depp", "Lil Wayne", "LeBron James", "Selena Gomez",
    "Kobe Bryant", "Albert Einstein", "Leonardo DiCaprio", "Kanye West", "Rihanna", "Scarlett Johansson",
    "Tupac Shakur", "Angelina Jolie", "Joe Biden", "John F. Kennedy", "Ariana Grande",
    "Prince Philip, Duke of Edinburgh", "Mark Zuckerberg", "Jennifer Aniston", "Tom Cruise",
    "Arnold Schwarzenegger", "Keanu Reeves", "Pablo Escobar", "Queen Victoria", "Meghan, Duchess of Sussex",
    "Muhammad Ali", "Mila Kunis", "Jay-Z", "William Shakespeare", "Bill Gates", "Ted Bundy"]

# loads the wikipedia dataset from huggingface if it does not yet exist
def load_wiki_dataset():
    if os.path.exists(filepath):
        logging.warn("Dataset already exists at ./" + filepath + " Delete to re-download.")
        quit()
    logging.info('Loading dataset...')
    try:
        return load_dataset("wikipedia", "20220301.en", split="train")
    except ValueError as err:
        logging.warn("Specified dataset not available, choose a current dataset:")
        logging.warn(err)
        quit()

# Extract the wiki article for every Wiki page title from the names list
# OUTPUT: articles list of format: [{id: dataset-id, text: wiki-text, title: wiki-page-title, url: link-to-wiki-page}, ...]
def extract_text(dataset):
    titles = dataset['title']

    # find the indices of each person
    indices = {}
    for name in top_50_persons:
        indices[name] = titles.index(name)

    # find the corresponding articles (for every index of a known person create a list of their wiki pages)
    articles = []
    for name in indices.keys():
        articles.append( dataset[indices[name]] )
    # strip all new line characters for easier processing
    for article in articles:
        article['text'] = article['text'].replace('\n', ' ')
    
    return articles

def save_to_csv(articles):
    csv_columns = ['id', 'text', 'title', 'url']
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
