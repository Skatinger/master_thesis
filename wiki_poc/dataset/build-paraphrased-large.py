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
    logging.info("Received SIGINT, saving checkpoint")
    global dataset
    dataset.save_to_disk(savepointPath)
    logging.info("exiting")
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
    # first try loading from savepoint
    try:
        logging.info("Trying to load from savepoint")
        return load_from_disk(savepointPath)
    except (FileNotFoundError, ValueError):
        try:
            logging.info("No savepoint found. Loading from {}".format(datasetPath))
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


# splits a dataset into nbShards, returning an array of shards
def get_shards(nbShards, dataset):
    shards = []
    for i in range(nbShards):
        shards.append(dataset.shard(num_shards=nbShards, index=i))
    return shards


if __name__ == '__main__':
    # read in the wiki-dataset
    dataset = load_wiki_dataset()

    # add paraphrased column if not already present
    if 'paraphrased' not in dataset.features.keys():
        dataset = dataset.add_column('paraphrased_sentences', [""] * len(dataset))

    # split dataset into shards of `splitSize` pages for faster processing
    # ERROR when using splitsize 1 -> get single shards which then produces one long text
    # instead of multiple sentences belonging to several pages
    splitSize = 2
    nbShards = round(len(dataset) / splitSize)
    shards = get_shards(nbShards, dataset)
    # used to track to which dataset row a row in each shard belongs to assign them faster
    datasetIndex = 0

    for index, shard in enumerate(shards):
        # skip shard if last of its rows has already been processed
        if (shard['paraphrased_sentences'][-1] != ""):
            logging.info("Skipping shard {}/{} as it has already been processed".format(index, nbShards))
            datasetIndex += len(shard)
            continue

        logging.info("Processing shard {}/{}".format(index, nbShards))

        # save number of sentences for each page, to match them
        # to the correct page after paraphrasing
        sentencesCounts = np.vectorize(len)(shard['sentences'])

        # flatten list of sentences for 1D array, and cast to python list as pegasus
        # model cannot handle numpy arrays
        sentences = np.concatenate(shard['sentences']).tolist()

        # sometimes the pages in the current shard cummulatively contain too many sentences
        # to be processed by the model at once, so we split them into chunks of 10 sentences
        paraphrased_sentences = []
        chunkSize = 10
        for i in range(0, len(sentences), chunkSize):
            chunk = sentences[i:i + chunkSize]
            paraphrased_sentences.extend(paraphrase_sentences(chunk))

        # split paraphrased sentences back into their pages
        # first define an array of indices specifying to which original dataset row the sentences belong,
        # allowing to assign them faster
        datasetIndices = list(range(datasetIndex, datasetIndex + len(shard)))
        for i, paraphrased in zip(datasetIndices, np.split(paraphrased_sentences, sentencesCounts.cumsum()[:-1])):
            dataset[i]['paraphrased_sentences'] = paraphrased

        # datasetIndex increases by the size of the shard
        datasetIndex += len(shard)

    # save to disk
    targetFolder = './data_paraphrased/'
    logging.info("Saving to {}".format(targetFolder))
    dataset.save_to_disk(targetFolder)
