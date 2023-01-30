import torch
import signal
import sys
import logging
from transformers import pipeline
from datasets import load_dataset, Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

logging.getLogger().setLevel(logging.INFO)


# allow signal handling, required when script is interrupted either with ctrl+c or with a proper sigint
# by the job handling server. All processing is cached to disk, this is just to ensure the job exits
# with a clean exit code and writes a short log to more easily see the reason for the exit upon log inspection
def signal_handler(sig, frame):
    logging.info("Received SIGINT, exiting.")
    sys.exit(0)


signal.signal(signal.SIGTERM, signal_handler)

# use CUDA if available (ususally has ID 0), -1 is huggingface default for CPU
device = 0 if torch.cuda.is_available() else -1


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


if __name__ == '__main__':
    if len(sys.argv) < 2:
        logging.info("Usage: python3 longformer_fill_mask.py <model_name> <dataset-config>")
        logging.info("Example: python3 longformer_fill_mask.py allenai/longformer-base-4096 original_4096")
        exit()

    model_name = sys.argv[1]
    config = sys.argv[2]
    logging.info("Using model {}".format(model_name))
    logging.info("Using device {}".format(device))

    dataset = load_dataset('skatinger/wikipedia-for-mask-filling', config, split='train')
    # create a split of the dataset to test the pipeline
    dataset = dataset.select(range(1000)).filter((lambda x: '<mask>' in x['texts']))  # temporary filter to fix issue in dataset
    pipe = pipeline('fill-mask', model=model_name, top_k=5, device=device)
    result_dataset = Dataset.from_dict({'predictions': [], 'scores': []})

    # can safely batch as the input is already chunked into 4096 tokens per sequence
    for out in tqdm(pipe(KeyDataset(dataset, 'texts'), batch_size=16)):
        # get a prediction for every chunk in the batch
        tokens, scores = extract_result(out)
        # add the predictions to the dataset
        result_dataset.add_item({'predictions': tokens, 'scores': scores})

    # save dataset
    path = "wiki_predictions_{}_{}".format(model_name.replace('/', '_'), config)
    logging.info("Saving dataset to path {}".format(path))
    result_dataset.save_to_disk(path)
