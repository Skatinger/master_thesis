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

datasetPath = 'data_masked'


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

"""
  Documentation for different models

  ----- 'roberta-base' -----
  takes an array of texts with masks, for every text input it predicts every mask with top k predictions
  with input [text1, text2, text3] result has format:
  [[results for text1], [results for text2], [...]] with results for each text as:
  any texts without mask are stripped before applying the model

  possible input cases:
    - no text (empty array)
    - single text with single mask
    - single text with multiple masks
    - multiple texts with single masks
    - multiple texts with multiple masks
    - multiple texts, some with multiple masks, some with single mask

  possible output cases:
    - no text (empty array)
    - array with single element, which is an array with 5 elements, which is a dict with token_str and score
    - TODO: add more cases
  --------------------------
"""


# loads the dataset containing original and paraphrased texts, exits in case it does not yet exist.
def load_dataset():
    logging.info('Loading dataset...')
    try:
        return load_from_disk(datasetPath)
    except ValueError as err:
        logging.warning("Specified dataset at {} not available".format(datasetPath))
        logging.warning(err)
        quit()


# extracts the relevant parts of the result of the fill-mask pipeline, removes unnecessary
# text outputs. Kept outputs are the predicted strings and their score. Score is rounded to 3 digits after comma
def extract_result(result):
    # if we get no predictions, return empty arrays
    if len(result) < 1:
        return [], []
    # result is an array with results for each input sequence. If there was only a single input sequence, the array
    # will only contain a single element.
    # each input sequence has a result for each mask, e.g a sequence with 2 masks will have 2 results in the array
    # representing the sequence results
    # for each result of a mask, there are five predictions with token and score
    results_tokens = []
    results_scores = []
    # iterate over all processed sequences
    for sequence_results in result:
        # if there was only a single mask, the result is not an array of arrays with each array containing 5
        # prediction hashes, but a single array containing 5 prediction hashes. This is why we need to check if
        # the result is an array of arrays and convert it to an array of arrays if it is not

        # first check if the sequence result itself is an array. If it is not, we need to convert it to an array
        if not isinstance(sequence_results, list):
            sequence_results = [sequence_results]

        # then, after the sequence result is definitely an array, check if the first element is an array.
        # if it is not, this was a sequence with a single mask. We need to convert the array to an array of arrays
        if not isinstance(sequence_results[0], list):
            sequence_results = [sequence_results]

        # iterate over all masks in the sequence, e.g. the predictions for those masks
        for mask_result in sequence_results:
            # if only a single mask was present in the sequence, it's just a dict, without array.
            # Wrap it in this case to make the following code work for both cases
            if type(mask_result) == dict:
                mask_result = [mask_result]

            # format of mask_result should be: [{token_str: predicted token, score: predicted sore, ...}, {...}]
            # we only need the token_str and score, so we extract those and append them to the results
            results_tokens.append([prediction['token_str'] for prediction in mask_result])
            results_scores.append([round(prediction['score'], 3) for prediction in mask_result])
    # return format is a an array of length of input sequences, with each array
    # containing an array of 5 predictions for each mask, e.g.
    # [sequence1_results, ...] where sequence1_results = [results_for_mask1, ...] where
    # results_for_mask1 = [(token_str, score), (...), ...]
    return results_tokens, results_scores


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
    example[f"predictions_tokens_{type}"] = []
    example[f"predictions_scores_{type}"] = []
    example[F"predictions_skipped_{type}"] = []
    # TODO: when using split_around_mask, text parts might be included several times.
    # this results in some masks being predicted multiple times. Handle that in case in
    # the splitter by removing all masks except the one we are prediciting
    splitGenerator = Splitter.split_by_chunksize(example[f"masked_text_{type}"], chunk_size, batch_size)
    for index, chunkBatch in enumerate(splitGenerator):
        # skip chunks if they do not contain a mask (returns empty array if no mask is found in any chunk)
        chunkBatch = ditch_without_mask(chunkBatch)
        # in some cases the chunkBatch contains a text sequence with foreign characters, which leads to a large
        # increase in tokens, usually above the maxium possible. If that happens, we skip the chunk and
        # save the skipped section. Catch with try-except
        try:
            predictions = fill_mask(chunkBatch)
        except RuntimeError as err:
            # if the error was that the text was too long
            if 'expanded size of the tensor' in str(err):
                # process each chunk in chunkBatch separately, to only skip the problematic ones
                for chunk in chunkBatch:
                    try:
                        predictions = fill_mask([chunk])
                    except RuntimeError as err2:
                        # if the error was that the text was too long
                        if 'expanded size of the tensor' in str(err2):
                            # annotate which part was skipped with indices in the text
                            start = index * chunk_size * batch_size
                            end = start + len(chunk)
                            example[f"predictions_skipped_{type}"].append(start, end)
                        else:
                            raise err2

                    tokens, scores = extract_result(predictions)
                    example[f"predictions_tokens_{type}"].append(tokens)
                    example[f"predictions_scores_{type}"].append(scores)
            else:
                raise err

        # get a prediction for every chunk in the batch
        tokens, scores = extract_result(predictions)
        # use += and not append, as we want an array of array, not a single concated array
        # this way its easier to get predictions per mask, as we can just iterate over the array
        # without knowing the number of masks or the number of predictions per mask
        example[f"predictions_tokens_{type}"] += tokens
        example[f"predictions_scores_{type}"] += scores
    return example


if __name__ == '__main__':
    # check if argument was passed
    if len(sys.argv) < 2:
        options = ('\n  -  ').join(["{} ({})".format(i, model) for i, model in enumerate(models)])
        logging.error("Missing argument <model-id>. Options are:\n  -  {}".format(options))
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
