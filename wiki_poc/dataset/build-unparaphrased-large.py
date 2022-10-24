# convert raw wiki text to sentences

import nltk
import logging
from datasets import load_from_disk
# sigint handler
import signal
import sys


# allow signal handling
def signal_handler(sig, frame):
    logging.info("Received SIGINT, saving checkpoint")
    global dataset
    dataset.save_to_disk("./data")
    logging.info("exiting")
    sys.exit(0)


signal.signal(signal.SIGTERM, signal_handler)
datasetPath = './data_reduced/'
logging.getLogger().setLevel(logging.INFO)


# use the SentencePiece model via nltk to split text into sentences
# https://arxiv.org/pdf/1808.06226.pdf
def split_to_sentences(text):
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logging.info("Downloading nltk punkt tokenizer")
        nltk.download('punkt')
    return nltk.tokenize.sent_tokenize(text, language='english')


# loads the wikipedia dataset from huggingface if it does not yet exist
def load_wiki_dataset():
    logging.info('Loading dataset...')
    try:
        return load_from_disk(datasetPath)
    except ValueError as err:
        logging.warning("Specified dataset at ./data not available")
        logging.warning(err)
        quit()


if __name__ == '__main__':
    # read in the wiki-dataset
    dataset = load_wiki_dataset()

    # batch dataset into chunks of 5000 to checkpoint progress
    batch_size = 5000
    batches = [dataset[i:i+batch_size] for i in range(0, len(dataset), batch_size)]
    for i, batch in enumerate(batches):
        logging.info(f"Processing batch {i+1}/{len(batches)}")
        dataset = dataset.map(lambda example: {'sentences': split_to_sentences(example['raw'])}, batched=True, batch_size=batch_size, num_proc=4)
        dataset.save_to_disk(datasetPath)

    logging.info("Saved unparaphrased wiki-dataset to {}".format(datasetPath))
