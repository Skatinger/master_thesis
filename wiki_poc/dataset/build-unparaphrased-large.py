# convert raw wiki text to sentences

import nltk
import logging
from datasets import load_from_disk
# sigint handler
import multiprocessing

# signal.signal(signal.SIGTERM, signal_handler)
datasetPath = './data_reduced/'
logging.getLogger().setLevel(logging.INFO)


# use the SentencePiece model via nltk to split text into sentences
# https://arxiv.org/pdf/1808.06226.pdf
def split_to_sentences(example) -> dict:
    example['sentences'] = nltk.tokenize.sent_tokenize(example['text'], language='english')
    return example


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

    # ensure required nltk package is installed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logging.info("Downloading nltk punkt tokenizer")
        nltk.download('punkt')

    # use as many processes as possible
    cpus = multiprocessing.cpu_count()
    # don't use more than 32 cores, parallelization overhead will decrease performance
    cpus = min(cpus, 32)
    dataset = dataset.map(split_to_sentences, num_proc=cpus)
    dataset.save_to_disk(datasetPath)

    logging.info("Saved unparaphrased wiki-dataset to {}".format(datasetPath))
