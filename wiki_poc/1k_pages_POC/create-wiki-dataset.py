# requires pip package `datasets`

# download wikipedia dataset from huggingface
# (check what versions are available if `20220301.en` is no longer available
#  on https://huggingface.co/datasets/wikipedia)
from datasets import load_dataset
dataset = load_dataset("wikipedia", "20220301.en", split='train')

# list of the top 50 person wiki pages over the years 2007 - 2022
# https://en.wikipedia.org/wiki/Wikipedia:Popular_pages#people
names = ["Donald Trump", "Barack Obama", "Elizabeth II", "Michael Jackson", "Cristiano Ronaldo",
        "Lady Gaga", "Adolf Hitler", "Eminem", "Justin Bieber", "Elon Musk", "Freddie Mercury",
        "Lionel Messi", "Kim Kardashian", "Steve Jobs", "Michael Jordan", "Dwayne Johnson",
        "Stephen Hawking", "Taylor Swift", "Miley Cyrus",
         "Abraham Lincoln", "Johnny Depp", "Lil Wayne", "LeBron James", "Selena Gomez",
         "Kobe Bryant", "Albert Einstein", "Leonardo DiCaprio", "Kanye West", "Rihanna",
         "Scarlett Johansson", "Tupac Shakur", "Angelina Jolie", "Joe Biden", "John F. Kennedy",
         "Ariana Grande", "Prince Philip, Duke of Edinburgh", "Mark Zuckerberg",
         "Jennifer Aniston", "Tom Cruise", "Arnold Schwarzenegger", "Keanu Reeves",
         "Pablo Escobar", "Queen Victoria", "Meghan, Duchess of Sussex", "Muhammad Ali",
         "Mila Kunis", "Jay-Z", "William Shakespeare", "Bill Gates", "Ted Bundy"]

# Extract the wiki article for every Wiki page title from the names list
# OUTPUT: articles list of format:
# [{id: dataset-id, text: wiki-text, title: wiki-page-title, url: link-to-wiki-page}, ...]
titles = dataset['title']
# find the indices in the dataset for each name
indices = {}
for name in names:
  indices[name] = titles.index(name)
# find the corresponding articles (for every index of a known person create a list of their wiki pages)
articles = []
for name in indices.keys():
  articles.append( dataset[indices[name]] )
# strip all new line characters for easier processing
for article in articles:
  article['text'] = article['text'].replace('\n', ' ')

## Save it as CSV
import csv
import os
csv_file = "wiki-dataset-reduced-masked.csv"
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, csv_file)
csv_columns = ['id', 'text', 'title', 'url']
with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in articles:
            writer.writerow(data)