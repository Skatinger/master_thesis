import signal
import sys
import logging
import itertools
import torch
from transformers import pipeline
from datasets import load_dataset, Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

logging.getLogger().setLevel(logging.INFO)


def signal_handler(_sig, _frame):
    """allow signal handling, required when script is interrupted either with
        ctrl+c or with a proper sigint by the job handling server. All processing is cached to disk,
        this is just to ensure the job exits with a clean exit code and writes a short log to
        more easily see the reason for the exit upon log inspection.

    Args:
        _sig (_type_): -
        _frame (_type_): -
    """
    logging.info("Received SIGINT, exiting.")
    sys.exit(0)


signal.signal(signal.SIGTERM, signal_handler)

# use CUDA if available (ususally has ID 0), -1 is huggingface default for CPU
DEVICE = 0 if torch.cuda.is_available() else -1


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
            if isinstance(mask_result, dict):
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


if __name__ == '__main__':
    if len(sys.argv) < 3:
        logging.info("Usage: python3 longformer_fill_mask.py <model_name> <dataset-config>")
        logging.info("Example: python3 longformer_fill_mask.py allenai/longformer-base-4096 original_4096")
        sys.exit()

    MODEL_NAME = sys.argv[1]
    CONFIG = sys.argv[2]
    SHARD_SIZE = None

    if len(sys.argv) == 4:
        # defines the number of pages to process in one run. The number of examples per page varies,
        # therefore the number of examples processed in one run is SHARD_SIZE * avg_examples_per_page
        SHARD_SIZE = int(sys.argv[3])
    logging.info('Using model %s', MODEL_NAME)
    logging.info('Using device %s', DEVICE)
    logging.info('Using dataset config %s', CONFIG)
    if SHARD_SIZE is not None:
        logging.info('Using shard size %i', SHARD_SIZE)

    # force redownload in case of corrupted cache
    dataset = load_dataset('rcds/wikipedia-for-mask-filling', CONFIG, split='train')

    # if only a subset of the dataset should be processed, create a subset of the dataset
    if SHARD_SIZE is not None:
        # need to create a set of page ids and then filter the dataset to only contain full pages
        # otherwise the dataset will contain partial pages, e.g. examples can not be puzzled together
        # to form a full page anymore
        dataset_ids = set(dataset['id'])
        shard_ids = [val for _, val in enumerate(itertools.islice(dataset_ids, SHARD_SIZE))]
        dataset = dataset.filter(lambda x: x['id'] in shard_ids, num_proc=4)

    logging.info('Left with %i examples (%i pages).', len(dataset), SHARD_SIZE)
    pipe = pipeline('fill-mask', model=MODEL_NAME, top_k=5, device=DEVICE)
    result_dataset = Dataset.from_dict({'predictions': [], 'scores': [], 'page_id': [], 'sequence_number': []})

    # can safely batch as the input is already chunked into 4096 tokens per sequence
    # if the loop runs out of memory, reduce the batch size
    for example, out in zip(dataset, tqdm(pipe(KeyDataset(dataset, 'texts'), batch_size=8))):
        # get a prediction for every chunk in the batch
        tokens, scores = extract_result(out)
        # add the predictions to the dataset
        result_dataset = result_dataset.add_item({
            'predictions': tokens,
            'scores': scores,
            'page_id': example['id'],
            'sequence_number': example['sequence_number']
        })

    # save dataset
    PATH = f"wiki_predictions_{MODEL_NAME.replace('/', '_')}_{CONFIG}.jsonl"
    logging.info('Saving dataset to path %s', PATH)
    result_dataset.to_json(PATH)
