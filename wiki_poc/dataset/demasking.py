import torch
# sigint handler
import signal
import sys
import logging
# instantiate transformer model
from transformers import pipeline
from custom.splitter import Splitter
from datasets import load_from_disk, concatenate_datasets

logging.getLogger().setLevel(logging.INFO)

datasetPath = 'data_masked_smol'  # 'data_masked'


# allow signal handling, required when script is interrupted either with ctrl+c or with a proper sigint
# by the job handling server. All processing is cached to disk, this is just to ensure the job exits
# with a clean exit code and writes a short log to more easily see the reason for the exit upon log inspection
def signal_handler(sig, frame):
    logging.info("Received SIGINT, exiting.")
    sys.exit(0)


signal.signal(signal.SIGTERM, signal_handler)

# use CUDA if available (ususally has ID 0), -1 is huggingface default for CPU
device = 0 if torch.cuda.is_available() else -1

# list of models available to perform the fill-mask task
models = [
    'roberta-base',
    'bert-base-uncased',
    'distilbert-base-uncased',
    'roberta-large',
    'bert-base-multilingual-cased',
]


# loads the dataset containing original and paraphrased texts, exits in case it does not yet exist.
def load_dataset():
    logging.info('Loading dataset...')
    try:
        return load_from_disk(datasetPath)
    except ValueError as err:
        logging.warning("Specified dataset at {} not available".format(datasetPath))
        logging.warning(err)
        quit()


# removes any text chunks which do not contain a <mask> token
def ditch_without_mask(chunks):
    have_masks = []
    for chunk in chunks:
        if '<mask>' in chunk:
            have_masks.append(chunk)
    return have_masks


def process_original(example):
    return process_page(example, 'original')


def process_paraphrased(example):
    return process_page(example, 'paraphrased')


# processes a single page, splits it into chunks and performs the fill-mask task on each chunk
# returning the example with a new column containing the results
def process_page(example, type):
    # length of text passed at once
    chunk_size = 1024
    # number of text passages of length chunk_size to pass to the model at once
    batch_size = 5
    example[f"predictions_{type}"] = []
    # TODO: when using split_around_mask, text parts might be included several times.
    # this results in some masks being predicted multiple times. Handle that in case in
    # the splitter by removing all masks except the one we are prediciting
    for chunkBatch in Splitter.split_by_chunksize(example[f"masked_text_{type}"], chunk_size, batch_size):
        # skip chunks if they do not contain a mask (returns empty array if no mask is found in any chunk)
        chunkBatch = ditch_without_mask(chunkBatch)
        try:
            example[f"predictions_{type}"] += (fill_mask(chunkBatch))
        except RuntimeError as err:
            if 'expanded size of the tensor' in str(err):
                logging.warning("Tensor was too long for id {} with chunk {}".format(example['id'], chunkBatch))
            else:
                logging.error("Error when processing chunk:\n{}".format(chunkBatch))
                raise err
    return example


if __name__ == '__main__':
    # check if argument was passed
    if len(sys.argv) < 2:
        options = ('\n  -  ').join(["{} ({})".format(i, model) for i, model in enumerate(models)])
        logging.error("Missing argument <model-id>. Options are: {}".format(options))
        exit(1)

    # get model name from args
    model_name = models[int(sys.argv[1])]
    logging.info("Using model {}".format(model_name))

    # number of shard splits to create when processing map function for full dataset takes a long time
    # each shard gets cached seperately, so we can process the dataset in multiple runs without complicated
    # cancellation or error handling
    numShards = 70

    # load dataset
    dataset = load_dataset()

    # instantiate fill-mask pipeline
    global fill_mask
    fill_mask = pipeline('fill-mask', model=model_name, top_k=5, device=device)

    # process original text
    computedOriginalShards = []
    for i in range(numShards):
        logging.info("Processing original text shard {}/{}".format(i, numShards))
        computedOriginalShards.append(dataset.shard(num_shards=numShards, index=i).map(process_original))
    dataset = concatenate_datasets(computedOriginalShards)

    # process paraphrased text
    computedParaphrasedShards = []
    for i in range(numShards):
        logging.info("Processing paraphrased text shard {}/{}".format(i, numShards))
        computedParaphrasedShards.append(dataset.shard(num_shards=numShards, index=i).map(process_paraphrased))
    dataset = concatenate_datasets(computedParaphrasedShards)

    # save dataset
    path = "wiki_predictions_{}".format(model_name)
    logging.info("Saving dataset to path {}".format(path))
    dataset.save_to_disk(path)
