# recognizes entities in the given sentences and masks them accordingly
# stores masked sentences and the tokens belonging to the masks


from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from ast import literal_eval
import os
import functools
import pandas as pd
import re
from tqdm import tqdm

# sigint handler
import signal
import sys
import logging
logging.getLogger().setLevel(logging.INFO)


# allow signal handling
def signal_handler(sig, frame):
    logging.info("Received SIGINT, saving checkpoint")
    global dataset
    dataset.to_csv(dataset_file, index=False)
    logging.info("exiting")
    sys.exit(0)


signal.signal(signal.SIGTERM, signal_handler)
dataset_file = 'wiki-dataset.csv'


@functools.lru_cache(maxsize=1)
def load_ner_pipeline(model_name="dslim/bert-base-NER"):
    print(f"Loading {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    return pipeline("ner", model=model, tokenizer=tokenizer)


def masking(ner_results, text, entity_to_mask, batch_sizes, mask_token='<mask>'):
    person_nr = -1
    entities = []
    indexes = []
    index_start = 0
    for batch_size, result in zip(batch_sizes, ner_results):
        # reset last_end index for each batch, we do not want to concat entities from different batches, as batches
        # are separated by sentences (and thus entities cannot overlap)
        last_end = -10
        for entity in result:
            tag = entity['entity']
            if 'PER' in tag:  # if it is a person
                # import pdb; pdb.set_trace()
                if ('B' in tag or 'I' in tag) and '#' not in entity['word']:
                    # if end of entity is right after the end of the last entity, merge them
                    if entity['start'] - 1 == last_end:
                        # concat to last entity part with empty space between
                        entities[person_nr] += ' ' + entity['word']
                        indexes[person_nr][1] = entity['end'] + index_start
                        last_end = entity['end'] + index_start
                    else:
                        person_nr += 1  # we came to the next person
                        entities.append(entity['word'].strip())
                        indexes.append([entity['start'] + index_start, entity['end'] + index_start])
                        last_end = entity['end']
                else:
                    if person_nr < 0:
                        continue
                    if '#' in entity['word']:
                        entities[person_nr] += entity['word'].strip().strip('#')
                        indexes[person_nr][1] = entity['end'] + index_start
                        last_end = entity['end'] + index_start
                    else:
                        entities[person_nr] += ' ' + entity['word']
                        indexes[person_nr][1] = entity['end'] + index_start
                        last_end = entity['end'] + index_start
        # update index_start for next batch, so indexes can be directly used for masking in full text
        # otherwise we would have to add the length of the previous batch to the indexes
        index_start += batch_size
    # remove entities which are not the ones we want to mask, e.g. remove persons which are not the person the article is about
    # complex checker:
    regex = ".*".join(entity_to_mask.split(" "))
    regex += ".*".join(['|.*(' + nameFragment + ').*' for nameFragment in entity_to_mask.split(" ")])

    remaining_entities = []
    remaining_indexes = []
    for entity, index in zip(entities, indexes):
        if bool(re.match(regex, entity)):
            remaining_entities.append(entity)
            remaining_indexes.append(index)

    # return a dataset of anonymized strings with the belonging entities
    # https://stackoverflow.com/questions/69921629/transformers-autotokenizer-tokenize-introducing-extra-characters
    # escape entity to ensure no brackets or other regex keywords are present (would cause a parsing error), for example
    # 'Heinrich ) Frank' causes a unbalanced parenthesis error
    text = "".join(text)
    for index in reversed(remaining_indexes):
        # the index start of an entity changes when text before it is masked, so we mask in reverse order
        text = text[:index[0]] + mask_token + text[index[1]:]

    return text, remaining_entities
    # return [re.sub(entity, mask_token, text) for entity in remaining_entities], remaining_entities


if __name__ == '__main__':
    if not os.path.exists(dataset_file):
        logging.error("Input file does not exist. Please run `download-wiki.py` first.")
        quit()

    # read in the wiki-dataset
    dataset = pd.read_csv(dataset_file)
    if 'sentences' not in dataset.columns:
        logging.error("Input file does not contain a 'sentences' column. Please run `build-unparaphrased.py` first.")
        quit()
    if 'paraphrased_sentences' not in dataset.columns:
        logging.error("Input file does not contain a 'paraphrased_sentences' column. Please run `build-paraphrased.py` first.")
        quit()

    # parse sentences string back to list of sentences
    dataset['sentences'] = dataset['sentences'].apply(literal_eval)
    dataset['paraphrased_sentences'] = dataset['paraphrased_sentences'].apply(literal_eval)

    # create batches of sentences
    sentences_batch_size = 7 # sentences per batch
    # creates a list of lists of sentences, where each list contains sentences_batch_size sentences
    dataset['batched_sentences'] = dataset['sentences'].apply(lambda x: [x[i:i + sentences_batch_size] for i in range(0, len(x), sentences_batch_size)])
    dataset['paraphrased_batched_sentences'] = dataset['paraphrased_sentences'].apply(lambda x: [x[i:i + sentences_batch_size] for i in range(0, len(x), sentences_batch_size)])

    # convert sentences to text
    dataset['batched_normal_text'] = dataset['batched_sentences'].apply(lambda x: [" ".join(sentences) for sentences in x])
    dataset['batched_paraphrased_text'] = dataset['paraphrased_batched_sentences'].apply(lambda x: [" ".join(sentences) for sentences in x])

    # add dataset columns for masking results if not yet existing
    if 'normal_masked_text' not in dataset.columns:
        logging.info('adding new columns to dataset')
        dataset['normal_masked_text'] = ""
        dataset['normal_entities'] = ""
        dataset['paraphrased_masked_text'] = ""
        dataset['paraphrased_entities'] = ""

    # perform NER on the texts
    ner = load_ner_pipeline()
    tqdm.pandas()
    print("Apply NER to normal text...")

    for index, row in dataset.iterrows():
        # skip iteration if value already present
        # value is '' if column newly added, float:nan if resumed
        # if (isinstance(row['normal_masked_text'], str) and len(row['normal_masked_text']) > 0):
            # logging.info("Skipping page {}, already done.".format(index))
            # continue

        # batched text is a list of strings, each string is a batch of sentences concated to a single string
        ner_result_unparaphrased = []
        batch_sizes = []
        for batch in row['batched_normal_text']:
            ner_result_unparaphrased.append(ner(batch))
            # we need to keep track of the length of the batches, so we can update the indexes of the entities
            batch_sizes.append(len(batch))
        dataset.at[index, 'normal_masked_text'], dataset.at[index, 'normal_entities'] = masking(ner_result_unparaphrased, row['batched_normal_text'], row['title'], batch_sizes=batch_sizes)

        # paraphrased text
        ner_result_paraphrased = []
        batch_sizes = []
        for batch in row['batched_paraphrased_text']:
            ner_result_paraphrased.append(ner(batch))
            # we need to keep track of the length of the batches, so we can update the indexes of the entities
            batch_sizes.append(len(batch))
        dataset.at[index, 'paraphrased_masked_text'], dataset.at[index, 'paraphrased_entities'] = masking(ner_result_paraphrased, row['batched_paraphrased_text'], row['title'], batch_sizes=batch_sizes)

        if (index % 5 == 0):
            logging.info("Checkpointing at page {}".format(index))
            dataset.to_csv(dataset_file, index=False)

    print("done, now cleaning")
    dataset.to_csv('intermediate.csv', index=False)

    # drop rows where we got no entities, as they are not interesting for our project
    cleanedDataset = dataset[dataset['normal_entities'].apply(lambda x: len(x) > 0)]
    cleanedDataset = dataset[dataset['paraphrased_entities'].apply(lambda x: len(x) > 0)]

    # save the dataset
    cleanedDataset.to_csv('wiki-dataset-masked.csv', index=False)
