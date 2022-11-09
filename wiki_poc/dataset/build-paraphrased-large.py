# uses the model tuner007/pegasus_paraphrase to paraphrase the wiki-dataset
# runs on CUDA if possible
# paraphrases each sentence seperately, to keep the length of the text, and only scramble the words a bit
# checkpoints are only saved on sigint

import faulthandler
import functools
import numpy as np
import logging
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
from datasets import load_from_disk
# sigint handler
import signal
import sys


# allow signal handling on jobs
def signal_handler(_sig, _frame):
    logging.info("Received SIGINT, terminating")
    sys.exit(0)


signal.signal(signal.SIGTERM, signal_handler)
datasetPath = './data_unparaphrased/'
savepointPath = './build_paraphrased_savepoint'
logging.getLogger().setLevel(logging.INFO)

# ensure error stack is printed when an error occurs on the GPU / Computing Cluster
faulthandler.enable()
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ensure model is loaded only once
# TODO: lru_caching probably not necessary, hugginface transformers should do that
@functools.lru_cache(maxsize=1)
def load_model(model_name='tuner007/pegasus_paraphrase'):
    logging.info(f"Loading {model_name}")
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    logging.info("Using {} device".format(torch_device))
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
    return model, tokenizer


# loads the wikipedia dataset from the configured datasetPath
def load_wiki_dataset():
    logging.info('Loading dataset...')
    try:
        logging.info("Loading from {}".format(datasetPath))
        return load_from_disk(datasetPath)
    except ValueError as err:
        logging.warning("Specified dataset at {} not available".format(datasetPath))
        logging.warning(err)
        quit()


def paraphrase_sentences(input_texts):
    num_return_sequences = 1
    num_beams = 10
    temperature = 1.5
    model, tokenizer = load_model()
    # truncation: True -> truncate inputs to max length allowed for the model (1024 tokens)
    # padding: 'longest' -> pad to the longest sequence in the batch
    # return_tensors: "pt" -> return pytorch tensors instead of python integer lists (better for GPU usage)
    batch = tokenizer(input_texts,
                      truncation=True,
                      padding='longest',
                      return_tensors="pt"
                      ).to(torch_device)

    # num_beams: number of beams to use for beam search e.g. number of sentences to generate for each input
    # num_return_sequences: number of sentences to return for each input (must be <= num_beams)
    # temperature: higher temperature -> more random, lower temperature -> more greedy
    translated = model.generate(**batch,
                                num_beams=num_beams,
                                num_return_sequences=num_return_sequences,
                                temperature=temperature)

    return tokenizer.batch_decode(translated, skip_special_tokens=True)


# generator that yields chunks of size n from a list
def chunks(lst, chunk_size=30):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


# takes a batch of wiki pages and paraphrases them, returns the list of paraphrased sentences for each page
def process_page(examples):
    # store the number of sentences for each page passed in the batch
    sentencesCounts = np.vectorize(len)(examples['sentences'])

    # flatten list of sentences for 1D array, and cast to python list
    # as pegasus model cannot handle numpy arrays
    sentences = np.concatenate(examples['sentences']).tolist()

    # sometimes the pages in the current examples cummulatively contain too many sentences
    # to be processed by the model at once, so we split them into chunks of 40 sentences
    paraphrased_sentences = []
    for chunk in chunks(sentences, 40):
        paraphrased_sentences.extend(paraphrase_sentences(chunk))

    # split paraphrased sentences back into arrays of sentences for each page
    return {"paraphrased_sentences": np.split(paraphrased_sentences, sentencesCounts.cumsum()[:-1])}


if __name__ == '__main__':
    # read in the wiki-dataset
    dataset = load_wiki_dataset()
    # apply paraphrasing to each page
    dataset.map(process_page, num_proc=2, batched=True, batch_size=8)
    # save to disk
    targetFolder = './data_paraphrased/'
    logging.info("Saving to {}".format(targetFolder))
    dataset.save_to_disk(targetFolder)
